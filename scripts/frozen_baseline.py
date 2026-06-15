"""A3 baseline: frozen sentence-transformer + nearest-neighbor link prediction.

No Tree-GRU, no GKI, no training. Each entity is its raw frozen text embedding;
a prediction is scored purely by cosine similarity between the subject text
embedding and each candidate object's text embedding (relation-agnostic). The
true object is ranked against the same type-constrained, filtered candidate set
used by the trained model, so MRR/Hits are directly comparable.

This is the honesty floor for the trained results: if the frozen baseline already
gets a high MRR, the Tree-GRU pipeline is not the source of the score. (On
CulturalBench, run it on masked vs unmasked text to quantify literal country
mention; see LABEL_LEAKAGE.md.)

Ranking reuses the vectorized ``evaluate_link_prediction`` so it scales to
WN18RR's ~40K-candidate sets.

Usage:
    python scripts/frozen_baseline.py                                  # CulturalBench
    python scripts/frozen_baseline.py --keep_country_text              # leaky text (A/B)
    python scripts/frozen_baseline.py --dataset wn18rr --embed_model all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from data.culturalbench import load_culturalbench, compute_text_embeddings
from data.negative_sampler import NegativeSampler
from evaluation.link_prediction import evaluate_link_prediction


class CosineNNPredictor:
    """Relation-agnostic frozen-NN scorer: score(s, ·, o) = cosine(h_s, h_o).

    Implements the only method ``evaluate_link_prediction`` needs
    (``score_all_candidates``), so the frozen floor uses the exact same filtered
    ranking as the trained model.
    """

    def score_all_candidates(self, h_s, r_id, h_cands):  # noqa: D401
        return F.cosine_similarity(h_s.unsqueeze(0), h_cands, dim=-1)


def load_graph(args):
    if args.dataset == "wn18rr":
        from data.wordnet import load_wn18rr
        return load_wn18rr(
            data_dir=args.data_dir,
            split_seed=args.seed,
            leakage_safe=not args.allow_leakage,
        )
    return load_culturalbench(
        split_seed=args.seed,
        leakage_safe=not args.allow_leakage,
        mask_country=not args.keep_country_text,
    )


def main():
    p = argparse.ArgumentParser(description="Frozen text + NN link-prediction baseline (A3).")
    p.add_argument("--dataset", type=str, default="culturalbench",
                   choices=["culturalbench", "wn18rr"])
    p.add_argument("--data_dir", type=str, default="data/wn18rr",
                   help="WN18RR data directory (wn18rr only).")
    p.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_country_text", action="store_true",
                   help="Keep country names in the text (reproduces the leak; "
                        "culturalbench only).")
    p.add_argument("--allow_leakage", action="store_true",
                   help="Reproduce the leaky graph split too.")
    args = p.parse_args()

    print("=" * 70)
    print("A3 frozen-encoder + nearest-neighbor baseline")
    print(f"  dataset      = {args.dataset}")
    print(f"  embed_model  = {args.embed_model}")
    if args.dataset == "culturalbench":
        print(f"  mask_country = {not args.keep_country_text}")
    print(f"  leakage_safe = {not args.allow_leakage}")
    print("=" * 70)

    graph = load_graph(args)
    emb = compute_text_embeddings(graph, model_name=args.embed_model)
    neg_sampler = NegativeSampler(graph.type_constraints, graph.all_triples)
    predictor = CosineNNPredictor()

    def run(triples):
        return evaluate_link_prediction(
            h_all=emb, triples=triples, predictor=predictor,
            graph=graph, neg_sampler=neg_sampler, device=args.device,
        )

    for split_name, triples in [("val", graph.val_triples), ("test", graph.test_triples)]:
        m = run(triples)
        print(f"{split_name:5s} (n={len(triples):5d}): "
              f"MRR={m['mrr']:.4f}  H@1={m['hits@1']:.4f}  "
              f"H@3={m['hits@3']:.4f}  H@10={m['hits@10']:.4f}")

    print("\n" + "=" * 70)
    test = run(graph.test_triples)
    print(f"RESULT | frozen-NN | dataset={args.dataset} | embed={args.embed_model} | "
          f"mask_country={not args.keep_country_text} | "
          f"test_mrr={test['mrr']:.4f} h@1={test['hits@1']:.4f} "
          f"h@3={test['hits@3']:.4f} h@10={test['hits@10']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
