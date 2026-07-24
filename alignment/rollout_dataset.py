"""Build the GRPO prompt dataset from the graph's survey questions.

Each training example is one graph ``question`` node (id_to_entity 'q:...'):
  - the user question text,
  - the scout-injected chat prompt (retrieval baked in at build time, so the
    graph encoder / scout stay FROZEN during RL -- this phase trains only the
    reasoner's behavior given retrieval),
  - the graph-grounded ``positions`` for the reward (positions_from_subtree at
    that question node), serialized as plain dicts so they ride the dataset.

LEAKAGE GUARD: the 60 OvertonBench questions are held-out eval. build_prompts
takes ``eval_holdout_texts`` and drops any graph question whose normalized text
matches one of them, so an eval question can never enter training.

Usage (materialize to JSONL for inspection):
    python -m alignment.rollout_dataset --embeddings embeddings_opinionqa.pt \
        --text_feat feats_opinionqa.pt --out grpo_prompts.jsonl --max_questions 50
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict

from alignment.reward import positions_from_subtree


def _norm(q: str) -> str:
    """Normalize a question for holdout matching (lowercase alnum tokens)."""
    return " ".join(re.findall(r"[a-z0-9]+", q.lower()))


def load_eval_holdout_texts(split: str = "full") -> set[str]:
    """Normalized text of the OvertonBench eval questions (never train on these)."""
    from evaluation.overton.eval_overtonbench import load_questions
    return {_norm(q) for _, q in load_questions(split)}


def build_prompts(graph, h_all, text_feat, manifold, *, cfg, instruction: str,
                  eval_holdout_texts: set[str] | None = None,
                  min_positions: int = 2, max_questions: int = 0,
                  q_emb_fn=None) -> list[dict]:
    """One record per usable graph question. Returns dicts with keys:
    question_id, question, prompt (chat messages), positions (list of dicts).

    A question is usable iff scout returns >=1 fork AND the subtree yields
    >= min_positions graph-grounded positions AND it is not a held-out eval
    question. ``cfg`` is a ScoutConfig; ``instruction`` the system prompt.
    """
    from retrieval.answer import build_prompt
    from retrieval.scout import embed_question, scout

    holdout = eval_holdout_texts or set()
    q_nodes = [nid for nid, name in enumerate(graph.id_to_entity)
               if name.startswith("q:")]

    records = []
    for nid in q_nodes:
        question = graph.entity_text.get(nid) or graph.id_to_entity[nid]
        if _norm(question) in holdout:
            continue
        positions = positions_from_subtree(graph, nid)
        if len(positions) < min_positions:
            continue
        q_emb = (q_emb_fn or embed_question)(question)
        forks = scout(question, graph, h_all, text_feat, manifold, cfg=cfg, q_emb=q_emb)
        if not forks:
            continue
        messages = build_prompt(question, forks, graph, instruction)
        records.append({
            "question_id": nid,
            "question": question,
            "prompt": messages,
            "positions": [asdict(p) for p in positions],
        })
        if max_questions and len(records) >= max_questions:
            break
    return records


def _main():
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import torch
    from pluraltree.manifolds.poincare import PoincareBall
    from retrieval.answer import INSTRUCTION_BY_CONDITION, PLURALISM_ROUTE
    from retrieval.scout import ScoutConfig, load_or_compute_text_feat

    ap = argparse.ArgumentParser(description="Build the GRPO prompt dataset")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--text_feat", default=None)
    ap.add_argument("--dataset", choices=["opinionqa", "globalopinionqa"], default="opinionqa")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--no_holdout", action="store_true",
                    help="skip the OvertonBench leakage guard (NOT for real runs)")
    ap.add_argument("--out", default="grpo_prompts.jsonl")
    args = ap.parse_args()

    if args.dataset == "opinionqa":
        from data.loaders.opinionqa import load_opinionqa
        graph = load_opinionqa(split_seed=args.seed, leakage_safe=True)
    else:
        from data.loaders.globalopinionqa import load_globalopinionqa
        graph = load_globalopinionqa(split_seed=args.seed, leakage_safe=True)

    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)
    text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat)
    holdout = set() if args.no_holdout else load_eval_holdout_texts()

    recs = build_prompts(graph, h_all, text_feat, manifold,
                         cfg=ScoutConfig(tau=args.tau, alpha=args.alpha),
                         instruction=PLURALISM_ROUTE,
                         eval_holdout_texts=holdout,
                         max_questions=args.max_questions)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} prompts to {args.out}  "
          f"(holdout {'off' if args.no_holdout else len(holdout)})")


if __name__ == "__main__":
    _main()
