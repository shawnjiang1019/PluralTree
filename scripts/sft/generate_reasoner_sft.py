"""Generate SFT data for the contrastive Reasoner — no hand-authoring.

Pipeline (per QA item):
    query + gold answer + anchor entity
      -> Scout: K structural cousins from h_all (structurally similar, semantically distant)
      -> verbalize each cousin into a caption + domain label
      -> distill a multi-path contrastive <think> trace with Claude (answer given:
         rationalization, so the trace is correct by construction)
      -> filter (references all K domains, has holds/breaks, faithful to provided domains)
      -> emit {query, anchor, node_ids, domains, gold_trace, answer}

Each record stores ``node_ids`` so the matching latents are injected at train time;
``gold_trace`` is the full ``<think>...</think><answer>...</answer>`` target.

QA input: JSONL, one object per line, e.g.
    {"question": "...", "answer": "...", "anchor": "<entity name>"}
``anchor`` is optional — if absent we try to link ``answer`` to an entity by name.

Usage:
    export ANTHROPIC_API_KEY=...
    python scripts/generate_reasoner_sft.py \
        --dataset wn18rr --data_dir data/wn18rr \
        --qa data/reasoner_qa.jsonl --embeddings runs/h_all.pt \
        --k 3 --out data/reasoner_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.sft.scout import scout
from pluraltree.sft.verbalize import verbalize_subtree
from evaluation.intrinsic.branch_divergence import divergence_anchors

MODEL = "claude-sonnet-4-6"

SYSTEM = (
    "You write contrastive, multi-strategy reasoning traces over a knowledge graph. "
    "You are given a question, its correct answer, and several STRUCTURAL ANALOGIES — "
    "subtrees that share the query's structure but come from different domains. "
    "Produce one reasoning path per analogy, then a contrastive analysis identifying where "
    "the structural analogy HOLDS and where it BREAKS down due to semantic differences. "
    "Reason toward the given correct answer. Use each analogy's domain label as the path source."
)

# Structured-output schema: one path per analogy + a holds/breaks contrast.
SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning_paths": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "the analogy's domain label"},
                    "reasoning": {"type": "string"},
                },
                "required": ["source", "reasoning"],
                "additionalProperties": False,
            },
        },
        "contrast": {
            "type": "object",
            "properties": {
                "holds": {"type": "string", "description": "where the structural analogy holds"},
                "breaks": {"type": "string", "description": "where it breaks down semantically"},
            },
            "required": ["holds", "breaks"],
            "additionalProperties": False,
        },
    },
    "required": ["reasoning_paths", "contrast"],
    "additionalProperties": False,
}


def load_graph(args):
    if args.dataset == "wn18rr":
        from data.loaders.wordnet import load_wn18rr
        return load_wn18rr(data_dir=args.data_dir, split_seed=args.seed, leakage_safe=True)
    from data.loaders.culturalbench import load_culturalbench
    return load_culturalbench(split_seed=args.seed, leakage_safe=True)


def load_embeddings(args, graph) -> torch.Tensor:
    """Trained hyperbolic h_all from --embeddings, else frozen text features (warned)."""
    if args.embeddings:
        h = torch.load(args.embeddings, map_location="cpu")
        return h if isinstance(h, torch.Tensor) else h["h_all"]
    print("  [warn] no --embeddings: using frozen text features (NOT trained hyperbolic "
          "embeddings). Scout geometry will be approximate.")
    from data.loaders.culturalbench import compute_text_embeddings
    return compute_text_embeddings(graph)


def link_anchor(item: dict, graph) -> int | None:
    """Resolve the QA item's anchor entity to a node id."""
    if item.get("anchor_id") is not None:          # synthesized items carry the id
        return item["anchor_id"]
    for key in ("anchor", "answer"):
        name = item.get(key)
        if name and name in graph.entity_vocab:
            return graph.entity_vocab[name]
    return None


def _descendant(branch_root: int, children_indices, rng, max_steps: int = 12) -> int:
    """Walk down a random path from a branch root to a (near-)leaf descendant."""
    v = branch_root
    for _ in range(max_steps):
        kids = children_indices[v]
        if not kids:
            break
        v = kids[rng.randrange(len(kids))]
    return v


def make_branch_question(anchor: int, graph, children_indices, rng) -> dict | None:
    """Synthesize a verifiable contrastive question at a divergence anchor.

    Branch-attribution: pick two of the anchor's child branches, sample a
    descendant of one, and ask which branch it falls under. The gold answer is
    the containing branch (known from the tree) — verifiable, and meaningful
    precisely because the branches diverge (that's why the anchor was selected).
    """
    kids = children_indices[anchor]
    if len(kids) < 2:
        return None
    i, j = rng.sample(range(len(kids)), 2)
    ca, cb = kids[i], kids[j]
    pick_a = rng.random() < 0.5
    x = _descendant(ca if pick_a else cb, children_indices, rng)
    if x in (ca, cb):                              # need a genuine descendant
        return None
    P = graph.id_to_entity[anchor]
    na, nb, nx = (graph.id_to_entity[ca], graph.id_to_entity[cb], graph.id_to_entity[x])
    q = (f"Within the category '{P}', does '{nx}' belong under the "
         f"'{na}' branch or the '{nb}' branch?")
    return {"question": q, "answer": na if pick_a else nb, "anchor_id": anchor}


def distill_trace(client, question, captions, domains, answer):
    """Ask Claude to rationalize a contrastive trace toward the given answer.

    Returns the structured dict, or None on failure.
    """
    analogies = "\n".join(f"- [{d}] {c}" for d, c in zip(domains, captions))
    user = (
        f"Question: {question}\n"
        f"Correct answer: {answer}\n\n"
        f"Structural analogies (use each domain label as a path source):\n{analogies}\n\n"
        f"Write one reasoning path per analogy, then the holds/breaks contrast."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def assemble_trace(parsed: dict, answer: str) -> str:
    lines = ["<think>"]
    for i, p in enumerate(parsed["reasoning_paths"], 1):
        lines.append(f"Path {i} ({p['source']}): {p['reasoning']}")
    c = parsed["contrast"]
    lines.append(f"Contrast: holds — {c['holds']}; breaks — {c['breaks']}")
    lines.append("</think>")
    lines.append(f"<answer>{answer}</answer>")
    return "\n".join(lines)


def passes_filter(parsed: dict, domains: list[str], k: int) -> bool:
    paths = parsed.get("reasoning_paths", [])
    if len(paths) < min(2, k):
        return False
    dom_lower = {d.lower() for d in domains}
    for p in paths:
        if not p.get("reasoning", "").strip():
            return False
        if not any(s in p["source"].lower() or p["source"].lower() in s for s in dom_lower):
            return False  # source not one of the provided analogy domains
    c = parsed.get("contrast", {})
    return bool(c.get("holds", "").strip()) and bool(c.get("breaks", "").strip())


def main():
    p = argparse.ArgumentParser(description="Generate contrastive Reasoner SFT data")
    p.add_argument("--dataset", choices=["culturalbench", "wn18rr"], default="wn18rr")
    p.add_argument("--data_dir", default="data/wn18rr")
    p.add_argument("--anchors", choices=["qa", "divergence"], default="qa",
                   help="qa = read anchors/questions from --qa file; divergence = "
                        "synthesize verifiable branch questions at Wasserstein "
                        "Divergence Anchors (integration #2).")
    p.add_argument("--qa", default=None, help="JSONL of {question, answer, anchor?} "
                   "(required for --anchors qa)")
    p.add_argument("--embeddings", default=None, help=".pt of trained h_all on the ball")
    p.add_argument("--out", required=True, help="output JSONL")
    p.add_argument("--k", type=int, default=3, help="number of structural cousins")
    p.add_argument("--candidate_pool", type=int, default=100)
    p.add_argument("--n_anchors", type=int, default=256,
                   help="parents to score for --anchors divergence")
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0, help="cap items (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.anchors == "qa" and not args.qa:
        p.error("--qa is required when --anchors qa")
    if args.anchors == "divergence" and not args.embeddings:
        p.error("--anchors divergence needs --embeddings (anchors are ranked from h_all)")

    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    print(f"Loading {args.dataset}...")
    graph = load_graph(args)
    h_all = load_embeddings(args, graph)
    manifold = PoincareBall(c=args.curvature)

    if args.anchors == "divergence":
        import random as _random
        rng = _random.Random(args.seed)
        print(f"  ranking divergence anchors (n_anchors={args.n_anchors})...")
        ranked = divergence_anchors(h_all, graph.children_indices, manifold=manifold,
                                    n_anchors=args.n_anchors, seed=args.seed)
        items = []
        for anchor, score in ranked:                 # highest-divergence first
            q = make_branch_question(anchor, graph, graph.children_indices, rng)
            if q is not None:
                q["divergence"] = round(score, 4)
                items.append(q)
        print(f"  {len(items)} branch questions from {len(ranked)} anchors")
    else:
        with open(args.qa, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        print(f"  {len(items)} QA items")
    if args.limit:
        items = items[: args.limit]

    n_ok, n_skip = 0, 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, item in enumerate(items):
            anchor = link_anchor(item, graph)
            if anchor is None:
                n_skip += 1
                continue
            cousins = scout(anchor, h_all, graph.children_indices, k=args.k,
                            manifold=manifold, candidate_pool=args.candidate_pool)
            if len(cousins) < min(2, args.k):
                n_skip += 1
                continue

            domains, captions = [], []
            for c in cousins:
                d, cap = verbalize_subtree(c, h_all, graph, manifold=manifold)
                domains.append(d)
                captions.append(cap)

            parsed = distill_trace(client, item["question"], captions, domains, item["answer"])
            if parsed is None or not passes_filter(parsed, domains, args.k):
                n_skip += 1
                continue

            record = {
                "query": item["question"],
                "answer": item["answer"],
                "anchor": graph.id_to_entity[anchor],
                "node_ids": cousins,
                "domains": domains,
                "gold_trace": assemble_trace(parsed, item["answer"]),
            }
            if "divergence" in item:
                record["divergence"] = item["divergence"]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            n_ok += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(items)} | kept {n_ok} | skipped {n_skip}")

    print(f"Done. Wrote {n_ok} records to {args.out} (skipped {n_skip}).")


if __name__ == "__main__":
    main()
