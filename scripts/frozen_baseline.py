"""A3 baseline: frozen sentence-transformer + nearest-neighbor link prediction.

No Tree-GRU, no GKI, no training. Each entity is its raw frozen text embedding;
a practice->country prediction is scored purely by cosine similarity between the
practice text embedding and each candidate country's text embedding. The true
country is ranked against the same type-constrained candidate set and filtered
set used by the trained model, so the MRR/Hits are directly comparable.

This is the honesty check for the A1 results: if the frozen baseline already
gets MRR ~0.9, the Tree-GRU pipeline is not the source of the score. Run it on
the masked and unmasked text to quantify how much is literal country mention.

Usage:
    python scripts/frozen_baseline.py
    python scripts/frozen_baseline.py --keep_country_text     # leaky text (A/B)
    python scripts/frozen_baseline.py --embed_model all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from data.culturalbench import load_culturalbench, compute_text_embeddings


def evaluate_split(triples, emb, graph, known_positives):
    """Filtered ranking of the true object by cosine similarity (no relation)."""
    ranks: list[int] = []
    for s_id, r_id, o_id in triples:
        candidates = graph.type_constraints.get(r_id, [])
        if not candidates:
            continue
        s_vec = emb[s_id].unsqueeze(0)                      # (1, d)
        cand = torch.tensor(candidates, device=emb.device)
        sims = F.cosine_similarity(s_vec, emb[cand], dim=-1)  # (C,)
        sim_map = {cid: float(s) for cid, s in zip(candidates, sims.tolist())}

        correct = sim_map[o_id]
        rank = 1
        for cid in candidates:
            if cid == o_id:
                continue
            if (s_id, r_id, cid) in known_positives:        # filtered setting
                continue
            if sim_map[cid] > correct:
                rank += 1
        ranks.append(rank)

    if not ranks:
        return {}
    r = torch.tensor(ranks, dtype=torch.float)
    return {
        "n":       len(ranks),
        "mrr":     (1.0 / r).mean().item(),
        "hits@1":  (r <= 1).float().mean().item(),
        "hits@3":  (r <= 3).float().mean().item(),
        "hits@10": (r <= 10).float().mean().item(),
    }


def main():
    p = argparse.ArgumentParser(description="Frozen text + NN link-prediction baseline (A3).")
    p.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_country_text", action="store_true",
                   help="Keep country names in the text (reproduces the leak).")
    p.add_argument("--allow_leakage", action="store_true",
                   help="Reproduce the leaky graph split too.")
    args = p.parse_args()

    print("=" * 70)
    print("A3 frozen-encoder + nearest-neighbor baseline")
    print(f"  embed_model  = {args.embed_model}")
    print(f"  mask_country = {not args.keep_country_text}")
    print(f"  leakage_safe = {not args.allow_leakage}")
    print("=" * 70)

    graph = load_culturalbench(
        split_seed=args.seed,
        leakage_safe=not args.allow_leakage,
        mask_country=not args.keep_country_text,
    )
    emb = compute_text_embeddings(graph, model_name=args.embed_model)
    known_positives = set(graph.all_triples)

    for split_name, triples in [("val", graph.val_triples), ("test", graph.test_triples)]:
        m = evaluate_split(triples, emb, graph, known_positives)
        if not m:
            print(f"{split_name}: (no scorable triples)")
            continue
        print(f"{split_name:5s} (n={m['n']:4d}): "
              f"MRR={m['mrr']:.4f}  H@1={m['hits@1']:.4f}  "
              f"H@3={m['hits@3']:.4f}  H@10={m['hits@10']:.4f}")

    print("\n" + "=" * 70)
    test = evaluate_split(graph.test_triples, emb, graph, known_positives)
    if test:
        print(f"RESULT | frozen-NN | embed={args.embed_model} | "
              f"mask_country={not args.keep_country_text} | "
              f"test_mrr={test['mrr']:.4f} h@1={test['hits@1']:.4f} "
              f"h@3={test['hits@3']:.4f} h@10={test['hits@10']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
