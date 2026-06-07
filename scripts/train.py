"""Entry point for training the Hyperbolic Tree-GRU + GKI model on CulturalBench.

Usage:
    python scripts/train.py
    python scripts/train.py --d_hidden 128 --n_epochs 100 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data.culturalbench import load_culturalbench, compute_text_embeddings
from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.knowledge.kg_embedding import KGEmbeddingSource
from pluraltree.combined.gki_tree_encoder import GKITreeEncoder
from pluraltree.combined.gki_tree_gru_cell import InjectionPoint
from training.scoring import HyperbolicLinkPredictor
from training.trainer import Trainer, TrainerConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train Hyperbolic Tree-GRU + GKI on CulturalBench")
    p.add_argument("--d_hidden",      type=int,   default=64)
    p.add_argument("--n_epochs",      type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--n_negative",    type=int,   default=10)
    p.add_argument("--margin",        type=float, default=1.0)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--lr_manifold",   type=float, default=1e-2)
    p.add_argument("--warmup1",       type=int,   default=500)
    p.add_argument("--warmup2",       type=int,   default=1500)
    p.add_argument("--gate_bias",     type=float, default=-2.0)
    p.add_argument("--curvature",     type=float, default=1.0)
    p.add_argument("--injection",     type=str,   default="post_agg",
                   choices=["pre_agg", "post_agg", "post_gru", "dual"])
    # --- A1 ablation flags ---
    p.add_argument("--no_gki",        action="store_true",
                   help="Disable all knowledge injection (pure Tree-GRU baseline).")
    p.add_argument("--gate_type",     type=str,   default="depth_aware",
                   choices=["depth_aware", "plain"],
                   help="depth_aware = radius-conditioned gate; plain = HyperbolicGate.")
    p.add_argument("--embed_model",   type=str,   default="all-MiniLM-L6-v2")
    p.add_argument("--allow_leakage", action="store_true",
                   help="Reproduce the original leaky setup (structural triples in "
                        "every split + tree built from all practices). Default is "
                        "leakage-safe evaluation.")
    p.add_argument("--keep_country_text", action="store_true",
                   help="Keep country names/demonyms in the question text (leaky). "
                        "Default masks them so the label is not in the input.")
    p.add_argument("--dataset",       type=str,   default="culturalbench",
                   choices=["culturalbench", "wn18rr"],
                   help="Which dataset/loader to use.")
    p.add_argument("--data_dir",      type=str,   default="data/wn18rr",
                   help="Directory of WN18RR train/valid/test.txt (wn18rr only).")
    p.add_argument("--device",        type=str,   default="cpu")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def run_tag(args) -> str:
    """A short, grep-friendly identifier for this run's configuration."""
    parts = [
        f"embed={args.embed_model}",
        f"d_hidden={args.d_hidden}",
        f"inj={args.injection}",
        f"gate={'NONE' if args.no_gki else args.gate_type}",
        f"mask_country={not args.keep_country_text}",
    ]
    return " | ".join(parts)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # 0. Run banner — makes each .out self-describing
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PluralTree training run")
    print("=" * 70)
    print(f"  RUN: {run_tag(args)}")
    print("  config:")
    for k, v in sorted(vars(args).items()):
        print(f"    {k:14s} = {v}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading {args.dataset}...")
    if args.dataset == "wn18rr":
        from data.wordnet import load_wn18rr
        graph = load_wn18rr(
            data_dir=args.data_dir,
            split_seed=args.seed,
            leakage_safe=not args.allow_leakage,
        )
        print(f"  leakage_safe = {not args.allow_leakage}")
    else:
        graph = load_culturalbench(
            split_seed=args.seed,
            leakage_safe=not args.allow_leakage,
            mask_country=not args.keep_country_text,
        )
        print(f"  leakage_safe = {not args.allow_leakage}")
        print(f"  mask_country = {not args.keep_country_text}")
    n_entities  = len(graph.id_to_entity)
    n_relations = len(graph.relation_vocab)
    print(f"  Entities: {n_entities}  |  Relations: {n_relations}")
    print(f"  Train triples: {len(graph.train_triples)}")
    print(f"  Val triples:   {len(graph.val_triples)}")
    print(f"  Test triples:  {len(graph.test_triples)}")

    # ------------------------------------------------------------------
    # 2. Compute text embeddings (node features + knowledge source)
    # ------------------------------------------------------------------
    print(f"Computing text embeddings with {args.embed_model}...")
    node_embeddings = compute_text_embeddings(graph, model_name=args.embed_model)
    d_input = node_embeddings.shape[1]
    print(f"  Embedding dim: {d_input}")

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    manifold = PoincareBall(c=args.curvature)

    # Text embeddings serve as the knowledge source (frozen)
    knowledge_source = KGEmbeddingSource(
        num_entities  = n_entities,
        embedding_dim = d_input,
        pretrained    = node_embeddings,
        freeze        = True,
    )

    injection_map = {
        "pre_agg":  InjectionPoint.PRE_AGGREGATION,
        "post_agg": InjectionPoint.POST_AGGREGATION,
        "post_gru": InjectionPoint.POST_GRU,
        "dual":     InjectionPoint.DUAL,
    }

    encoder = GKITreeEncoder(
        d_input          = d_input,
        d_hidden         = args.d_hidden,
        manifold         = manifold,
        sources          = [knowledge_source],
        injection_point  = injection_map[args.injection],
        gate_bias        = args.gate_bias,
        depth_aware      = (args.gate_type == "depth_aware"),
        inject           = (not args.no_gki),
    )
    if args.no_gki:
        print("  [ablation] GKI DISABLED — pure Tree-GRU")
    else:
        print(f"  [ablation] gate_type={args.gate_type}  injection={args.injection}")

    predictor = HyperbolicLinkPredictor(
        num_relations = n_relations,
        d_hidden      = args.d_hidden,
        manifold      = manifold,
    )

    total_params = sum(p.numel() for p in list(encoder.parameters()) + list(predictor.parameters()) if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    config = TrainerConfig(
        d_hidden       = args.d_hidden,
        n_epochs       = args.n_epochs,
        batch_size     = args.batch_size,
        n_negative     = args.n_negative,
        margin         = args.margin,
        lr             = args.lr,
        lr_manifold    = args.lr_manifold,
        warmup1        = args.warmup1,
        warmup2        = args.warmup2,
        gate_bias_init = args.gate_bias,
        device         = args.device,
    )

    trainer = Trainer(
        encoder         = encoder,
        predictor       = predictor,
        graph           = graph,
        node_embeddings = node_embeddings,
        config          = config,
    )

    trainer.train()

    # ------------------------------------------------------------------
    # 5. Final test evaluation
    # ------------------------------------------------------------------
    print("\nFinal test evaluation:")
    test_metrics = trainer.evaluate("test")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # 6. One-line summary — easy to grep/compare across runs
    #    e.g.  grep '^RESULT' logs/a1_*.out
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULT | {run_tag(args)} | best_val_mrr={trainer.best_val_mrr:.4f} | "
          f"test_mrr={test_metrics.get('mrr', float('nan')):.4f} "
          f"h@1={test_metrics.get('hits@1', float('nan')):.4f} "
          f"h@3={test_metrics.get('hits@3', float('nan')):.4f} "
          f"h@10={test_metrics.get('hits@10', float('nan')):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
