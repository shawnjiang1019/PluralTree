"""What each retriever hands persona drafting, per OvertonBench question. No LLM.

Run once per relevance encoder, then compare the CSVs before spending GPU time
on generation (scripts/analysis/compare_retrieval_diag.py). For every question
and retriever (``scout`` = persona_merge's Wasserstein scout, ``axis`` =
persona_axis's geometry-free axis_forks) it records:

  items_passing  how many survey items clear tau -- the gate was set for MiniLM
                 cosines (0.25); a larger encoder's cosines sit on another
                 scale, so this is where a kept tau shows its effect
  top_item_*     the survey item the top fork belongs to, and its relevance
                 (the first thing to read: is it about the question at all?)
  top_*          the top fork: anchor node, its type, the divergence it was
                 ranked by (W for scout, 0.5 JS + 0.5 EMD for axis), relevance
  personas       whether persona drafting gets two subgroups or falls back to
                 the plain draft (scout: only when the top fork is an axis)
  pair_*         the persona pair's JS, EMD and D on their REAL answers,
                 computed the same way for both retrievers, so "which retriever
                 finds groups that disagree more" is a like-for-like question
  scale          ordinal / nominal, from the release's option_ordinal

    python scripts/analysis/retrieval_diag.py --dataset opinionqa \
        --embeddings embeddings_opinionqa.pt --text_feat feats_opinionqa.pt \
        --out retrieval_diag_minilm.csv
    python scripts/analysis/retrieval_diag.py --dataset opinionqa \
        --embeddings embeddings_opinionqa.pt --text_feat feats_opinionqa_bge.pt \
        --embed_model BAAI/bge-m3 --out retrieval_diag_bge.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

COLS = ["question_id", "question", "retriever", "embed_model", "items_passing",
        "n_forks", "top_item", "top_item_rel", "top_anchor_type", "top_anchor",
        "top_relevance", "top_divergence", "personas", "scale", "pair_js",
        "pair_emd", "pair_d", "persona_a", "persona_b"]


def _parents(children_indices: list[list[int]]) -> dict[int, int]:
    return {c: p for p, kids in enumerate(children_indices) for c in kids}


def _item_of(node: int, types: dict, parent: dict) -> int | None:
    """The survey question a node sits under (itself, if it is one)."""
    v = node
    while v is not None:
        if types.get(v) == "question":
            return v
        v = parent.get(v)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline retrieval diagnostic")
    ap.add_argument("--dataset", default="opinionqa")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--text_feat", required=True)
    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--curvature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42,
                    help="must match the embed job's train.py --seed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from data.loaders.graphs import load_graph
    from evaluation.overton.eval_overtonbench import load_questions
    from pluraltree.manifolds.poincare import PoincareBall
    from retrieval.answer import CONDITIONS, pick_personas
    from retrieval.scout import (_emd_norm, _js_bits, _option_cost, axis_forks,
                                 describe_node, embed_question,
                                 load_or_compute_text_feat, node_relevance, scout)

    graph = load_graph(args.dataset, split_seed=args.seed, leakage_safe=True)
    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)
    text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat,
                                          model_name=args.embed_model)
    probe = embed_question("probe", args.embed_model)
    if text_feat.shape[1] != probe.shape[0]:
        raise SystemExit(f"--text_feat is {text_feat.shape[1]}-d but "
                         f"{args.embed_model} embeds to {probe.shape[0]}-d")
    types = graph.entity_types
    parent = _parents(graph.children_indices)
    dists = graph.opinion_dist
    n_ord = len(getattr(graph, "opinion_ordinal", {}))
    print(f"graph: {len(dists)} subgroup leaves, {n_ord} with option_ordinal")

    retrievers = {
        "scout": (scout, replace(CONDITIONS["persona_merge"],
                                 embed_model=args.embed_model)),
        "axis": (axis_forks, replace(CONDITIONS["persona_axis"],
                                     embed_model=args.embed_model)),
    }
    lam = retrievers["axis"][1].lam
    tau = retrievers["axis"][1].tau

    rows = []
    for qid, question in load_questions():
        q_emb = embed_question(question, args.embed_model)
        rel = node_relevance(q_emb, text_feat)
        n_items = sum(1 for v, t in types.items()
                      if t == "question" and float(rel[v]) >= tau)
        for name, (fn, cfg) in retrievers.items():
            forks = fn(question, graph, h_all, text_feat, manifold,
                       cfg=cfg, q_emb=q_emb)
            row = {"question_id": qid, "question": question, "retriever": name,
                   "embed_model": args.embed_model, "items_passing": n_items,
                   "n_forks": len(forks), "personas": False}
            if forks:
                f = forks[0]
                item = _item_of(f.anchor, types, parent)
                row.update(
                    top_item=describe_node(graph, item, 200) if item is not None else "",
                    top_item_rel=round(float(rel[item]), 4) if item is not None else "",
                    top_anchor_type=types.get(f.anchor, ""),
                    top_anchor=describe_node(graph, f.anchor, 160),
                    top_relevance=round(f.relevance, 4),
                    top_divergence=round(f.w, 4))
                pair = pick_personas(f, graph, 2)
                if len(pair) == 2 and all(v in dists for v in pair):
                    a, b = pair
                    cost, pos = _option_cost(graph, a, args.embed_model)
                    js = _js_bits(dists[a], dists[b])
                    emd = _emd_norm(dists[a], dists[b], cost, pos)
                    row.update(
                        personas=True,
                        scale="ordinal" if cost is None else "nominal",
                        pair_js=round(js, 4), pair_emd=round(emd, 4),
                        pair_d=round(lam * js + (1 - lam) * emd, 4),
                        persona_a=describe_node(graph, a, 0, show_q=True),
                        persona_b=describe_node(graph, b, 0, show_q=True))
            rows.append(row)
        print(f"  Q{qid}: {n_items} items >= tau")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})

    print(f"\n[{args.embed_model}]  items passing tau={tau}: median "
          f"{st.median(r['items_passing'] for r in rows):.0f}")
    for name in retrievers:
        sub = [r for r in rows if r["retriever"] == name]
        made = [r for r in sub if r["personas"]]
        line = (f"{name:6} personas {len(made)}/{len(sub)}, "
                f"no forks {sum(1 for r in sub if not r['n_forks'])}")
        if made:
            line += (f", pair D mean {st.mean(r['pair_d'] for r in made):.3f}, "
                     f"top-item rel mean "
                     f"{st.mean(r['top_item_rel'] for r in made if r['top_item_rel'] != ''):.3f}")
        print(line)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
