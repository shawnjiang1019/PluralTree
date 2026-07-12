"""Structure-fidelity metrics: frozen-text floor vs. a trained embedding.

The structure metrics (subtree_ap, ancestor_auc, ...) are only interpretable
*relative to a floor* — the raw sentence-transformer text embeddings, with no
training and no graph. This script computes that floor for a dataset and, if you
pass a saved trained embedding, prints both side by side with the delta. That is
what tells you whether the encoder added geometry or just inherited it from text.

Usage:
    # floor only (no trained model needed):
    python scripts/eval_structure.py --dataset wn18rr --embed_model all-mpnet-base-v2

    # floor vs. a trained embedding saved by train.py --save_embeddings:
    python scripts/eval_structure.py --dataset wn18rr --embed_model all-mpnet-base-v2 \
        --embeddings embeddings_up.pt

The trained embedding is scored with hyperbolic distance (it lives on the ball);
the floor is scored with Euclidean distance (manifold=None). compute_structure_metrics
supports both, so the comparison is apples-to-apples on the *same* tree.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from data.loaders.culturalbench import load_culturalbench, compute_text_embeddings
from pluraltree.manifolds.poincare import PoincareBall
from evaluation.intrinsic.structure_metrics import compute_structure_metrics


# Direction of "better" per metric (sibling_ratio is special — see note in output).
HIGHER_IS_BETTER = {
    "depth_radius_rho": True,
    "dist_tree_rho":    True,
    "recon_map":        True,
    "subtree_ap":       True,
    "ancestor_auc":     True,
    "sibling_ratio":    False, 
}


def parse_args():
    p = argparse.ArgumentParser(description="Structure-fidelity floor vs trained embedding")
    p.add_argument("--dataset",       type=str, default="wn18rr",
                   choices=["culturalbench", "wn18rr"])
    p.add_argument("--data_dir",      type=str, default="data/wn18rr",
                   help="WN18RR data directory (wn18rr only).")
    p.add_argument("--embed_model",   type=str, default="all-mpnet-base-v2",
                   help="Sentence-transformer used for the frozen-text floor.")
    p.add_argument("--embeddings",    type=str, default=None,
                   help="Optional path to a trained (N, d) embedding tensor on the "
                        "ball, saved by train.py --save_embeddings.")
    p.add_argument("--curvature",     type=float, default=1.0,
                   help="Curvature for the trained embedding's hyperbolic distance.")
    p.add_argument("--subtree_level", type=int, default=1)
    p.add_argument("--n_anchors",     type=int, default=256)
    p.add_argument("--n_pairs",       type=int, default=4000)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--allow_leakage", action="store_true",
                   help="Build the tree from all splits (default: train only).")
    p.add_argument("--keep_country_text", action="store_true")
    return p.parse_args()


def load_graph(args):
    if args.dataset == "wn18rr":
        from data.loaders.wordnet import load_wn18rr
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
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading {args.dataset}...")
    graph = load_graph(args)
    n = len(graph.id_to_entity)
    print(f"  Entities: {n}  |  tree edges: {sum(len(c) for c in graph.children_indices)}")

    metric_kwargs = dict(
        children_indices=graph.children_indices,
        topo_order=graph.topo_order,
        subtree_level=args.subtree_level,
        n_anchors=args.n_anchors,
        n_pairs=args.n_pairs,
        seed=args.seed,
    )

    # ---- Floor: raw text embeddings, Euclidean, no training, no graph ----
    print(f"Computing frozen-text floor with {args.embed_model}...")
    text_emb = compute_text_embeddings(graph, model_name=args.embed_model)
    floor = compute_structure_metrics(text_emb, manifold=None, **metric_kwargs)

    # ---- Trained: hyperbolic embedding on the ball (optional) ----
    trained = None
    if args.embeddings:
        print(f"Loading trained embedding from {args.embeddings}...")
        h = torch.load(args.embeddings, map_location="cpu")
        if h.shape[0] != n:
            raise ValueError(
                f"Embedding has {h.shape[0]} rows but the graph has {n} entities — "
                f"mismatched dataset/run?"
            )
        manifold = PoincareBall(c=args.curvature)
        trained = compute_structure_metrics(h, manifold=manifold, **metric_kwargs)

    # ---- Report ----
    keys = list(floor.keys())
    print("\n" + "=" * 72)
    if trained is None:
        print(f"{'metric':20s} {'floor':>10s}")
        print("-" * 72)
        for k in keys:
            print(f"{k:20s} {floor[k]:>10.4f}")
        print("=" * 72)
        print("Floor only. Re-run with --embeddings to compare a trained model.")
        return

    print(f"{'metric':20s} {'floor':>10s} {'trained':>10s} {'delta':>10s}  better?")
    print("-" * 72)
    for k in keys:
        f, t = floor[k], trained[k]
        delta = t - f
        # "win" = trained beats floor in the metric's preferred direction
        if k in HIGHER_IS_BETTER:
            win = delta > 0 if HIGHER_IS_BETTER[k] else delta < 0
        else:
            win = None
        mark = "" if win is None else (" yes" if win else " no")
        print(f"{k:20s} {f:>10.4f} {t:>10.4f} {delta:>+10.4f}{mark}")
    print("=" * 72)
    print("Note: sibling_ratio lower = tighter siblings, BUT near 0 = collapse "
          "(over-smoothing). Read it together with H@1.")


if __name__ == "__main__":
    main()
