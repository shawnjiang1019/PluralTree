"""What each retriever hands persona drafting, per OvertonBench question. No LLM.

Run once per relevance encoder, then compare the CSVs before spending GPU time
on generation. For every question and retriever (``scout`` = persona_merge's
Wasserstein scout, ``axis`` = persona_axis's geometry-free axis_forks) it records:

  items_passing  how many survey items clear tau -- the gate was set for MiniLM
                 cosines (0.25); a larger encoder's cosines sit on another
                 scale, so this is where a kept tau shows its effect
  top_*          the top fork: its anchor text, the two persona subgroups, the
                 divergence it was ranked by (W or 0.5 JS + 0.5 EMD), relevance
  personas       whether persona_merge would get two subgroups or fall back to
                 the plain draft (scout: only when the top fork is an axis)

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
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


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
    from retrieval.scout import (axis_forks, describe_node, embed_question,
                                 load_or_compute_text_feat, scout)

    graph = load_graph(args.dataset, split_seed=args.seed, leakage_safe=True)
    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = PoincareBall(c=args.curvature)
    text_feat = load_or_compute_text_feat(graph, args.dataset, args.text_feat,
                                          model_name=args.embed_model)
    types = graph.entity_types
    retrievers = {
        "scout": (scout, replace(CONDITIONS["persona_merge"],
                                 embed_model=args.embed_model)),
        "axis": (axis_forks, replace(CONDITIONS["persona_axis"],
                                     embed_model=args.embed_model)),
    }

    rows = []
    for qid, question in load_questions():
        q_emb = embed_question(question, args.embed_model)
        q = q_emb / q_emb.norm().clamp_min(1e-9)
        rel = (text_feat / text_feat.norm(dim=-1, keepdim=True).clamp_min(1e-9)) @ q
        tau = retrievers["axis"][1].tau
        n_items = sum(1 for v, t in types.items()
                      if t == "question" and float(rel[v]) >= tau)
        for name, (fn, cfg) in retrievers.items():
            forks = fn(question, graph, h_all, text_feat, manifold,
                       cfg=cfg, q_emb=q_emb)
            row = {"question_id": qid, "question": question, "retriever": name,
                   "embed_model": args.embed_model, "items_passing": n_items,
                   "n_forks": len(forks)}
            if forks:
                f = forks[0]
                pair = pick_personas(f, graph, 2)
                row.update(
                    top_anchor_type=types.get(f.anchor, ""),
                    top_anchor=describe_node(graph, f.anchor, 160),
                    top_relevance=round(f.relevance, 4),
                    top_divergence=round(f.w, 4),
                    personas=len(pair) == 2,
                    persona_a=describe_node(graph, pair[0], 0, show_q=True)
                    if len(pair) == 2 else "",
                    persona_b=describe_node(graph, pair[1], 0, show_q=True)
                    if len(pair) == 2 else "")
            else:
                row.update(personas=False)
            rows.append(row)
        print(f"  Q{qid}: {n_items} items >= tau")

    cols = ["question_id", "question", "retriever", "embed_model",
            "items_passing", "n_forks", "top_anchor_type", "top_anchor",
            "top_relevance", "top_divergence", "personas", "persona_a",
            "persona_b"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    for name in retrievers:
        sub = [r for r in rows if r["retriever"] == name]
        made = sum(1 for r in sub if r["personas"])
        print(f"{name:6} personas on {made}/{len(sub)} questions; "
              f"no forks on {sum(1 for r in sub if not r['n_forks'])}")
    print(f"items passing tau: median "
          f"{sorted(r['items_passing'] for r in rows)[len(rows) // 2]}")
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
