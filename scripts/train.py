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
    p.add_argument("--embed_model",   type=str,   default="all-MiniLM-L6-v2")
    p.add_argument("--device",        type=str,   default="cpu")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading CulturalBench...")
    graph = load_culturalbench(split_seed=args.seed)
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
    )

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


if __name__ == "__main__":
    main()
