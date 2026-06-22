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
    p.add_argument("--checkpoint",    action="store_true",
                   help="Gradient-checkpoint the encoder (recompute each tree "
                        "level in backward). Big memory savings for large graphs "
                        "(WN18RR) at ~20-30%% extra compute.")
    # --- Extra message-passing flows (default off = pure bottom-up) ---
    p.add_argument("--bidirectional", action="store_true",
                   help="Add a top-down pass so children are conditioned on their "
                        "parents (symmetric parent<->child message passing, B4).")
    p.add_argument("--lateral",       action="store_true",
                   help="Add a same-depth pass so a node is conditioned on its "
                        "siblings (lateral message passing).")
    p.add_argument("--lateral_max_siblings", type=int, default=16,
                   help="Cap on siblings aggregated per node in the lateral pass "
                        "(bounds memory for high-fan-out parents).")
    # --- Structure-fidelity loss (train the geometry, not just the link) ---
    p.add_argument("--lambda_struct", type=float, default=0.0,
                   help="Weight on the hierarchy structure-fidelity loss. 0 = off "
                        "(pure link prediction). Try ~0.1 to push the embedding "
                        "geometry to respect the tree (see docs/EVALUATION.md).")
    p.add_argument("--struct_margin", type=float, default=1.0,
                   help="Margin for the structure-fidelity loss (parent closer than "
                        "a non-ancestor by this much).")
    p.add_argument("--inductive_holdout", type=float, default=0.0,
                   help="WN18RR only: hold out this fraction of leaf entities from "
                        "training and evaluate their links inductively (embedded "
                        "from text+structure alone, never trained on).")
    p.add_argument("--encode_every",  type=int,   default=1,
                   help="Re-encode the full tree every N training steps (1 = every "
                        "batch). Larger N reuses a detached embedding between "
                        "refreshes — big speedup on large graphs (WN18RR/PrimeKG).")
    p.add_argument("--save_embeddings", type=str, default=None,
                   help="Path to save the final (N, d) embedding tensor for offline "
                        "structure evaluation (scripts/eval_structure.py).")
    p.add_argument("--metrics_csv",    type=str, default=None,
                   help="Path to append per-eval validation metrics (step, epoch, "
                        "MRR, Hits) for offline plotting (scripts/plot_metrics.py).")
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
        f"flow={'+'.join(['up'] + (['down'] if args.bidirectional else []) + (['lat'] if args.lateral else []))}",
        f"lstruct={args.lambda_struct}",
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
            holdout_entities=args.inductive_holdout,
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
        gradient_checkpointing = args.checkpoint,
        bidirectional        = args.bidirectional,
        lateral              = args.lateral,
        lateral_max_siblings = args.lateral_max_siblings,
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
        lambda_struct  = args.lambda_struct,
        struct_margin  = args.struct_margin,
        encode_every   = args.encode_every,
        metrics_csv    = args.metrics_csv,
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
    # 5b. Structure-fidelity metrics — does the geometry encode the tree?
    #     (subtree/path quality, beyond link prediction; see docs/EVALUATION.md)
    # ------------------------------------------------------------------
    from evaluation.structure_metrics import compute_structure_metrics
    encoder.eval()
    with torch.no_grad():
        h_all_final = trainer.encode_tree()
    if args.save_embeddings:
        torch.save(h_all_final.detach().cpu(), args.save_embeddings)
        print(f"  Saved final embeddings to {args.save_embeddings}")
    struct = compute_structure_metrics(
        h_all_final,
        graph.children_indices,
        graph.topo_order,
        manifold = manifold,
        seed     = args.seed,
    )
    print("\nStructure-fidelity metrics (geometry quality):")
    for k, v in struct.items():
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # 5c. Inductive evaluation — link prediction on held-out entities the model
    #     never trained on (embedded from text + structure alone). Reuses the
    #     final embeddings; a lookup-table KGE would score ~0 here.
    # ------------------------------------------------------------------
    ind_metrics = None
    if getattr(graph, "inductive_test", None):
        from evaluation.link_prediction import evaluate_link_prediction
        ind_metrics = evaluate_link_prediction(
            h_all       = h_all_final,
            triples     = graph.inductive_test,
            predictor   = predictor,
            graph       = graph,
            neg_sampler = trainer.neg_sampler,
            device      = args.device,
        )
        print(f"\nInductive test (held-out entities, n={len(graph.inductive_test)}):")
        for k, v in ind_metrics.items():
            print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # 6. One-line summaries — easy to grep/compare across runs
    #    e.g.  grep '^RESULT' logs/*.out  ;  grep '^STRUCT' logs/*.out
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULT | {run_tag(args)} | best_val_mrr={trainer.best_val_mrr:.4f} | "
          f"test_mrr={test_metrics.get('mrr', float('nan')):.4f} "
          f"h@1={test_metrics.get('hits@1', float('nan')):.4f} "
          f"h@3={test_metrics.get('hits@3', float('nan')):.4f} "
          f"h@10={test_metrics.get('hits@10', float('nan')):.4f}")
    print(f"STRUCT | {run_tag(args)} | "
          + " ".join(f"{k}={v:.4f}" for k, v in struct.items()))
    if ind_metrics is not None:
        print(f"INDUCTIVE | {run_tag(args)} | n={len(graph.inductive_test)} | "
              f"mrr={ind_metrics.get('mrr', float('nan')):.4f} "
              f"h@1={ind_metrics.get('hits@1', float('nan')):.4f} "
              f"h@3={ind_metrics.get('hits@3', float('nan')):.4f} "
              f"h@10={ind_metrics.get('hits@10', float('nan')):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
