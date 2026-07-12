"""Inductive link prediction on the GraIL WN18RR benchmark (comparable splits).

Trains on a version's training graph, then encodes the inductive graph (disjoint,
never-seen entities) from text + structure and predicts its held-out facts, under
BOTH protocols:
  * modern filtered MRR / Hits@{1,3,10} (rank vs all entities)        -> RESULT
  * original GraIL Hits@10 / MRR vs 50 negatives + AUC-PR             -> GRAIL

Inductive requires the text-only encoder, so GKI is always off (a GKI source's
entity table cannot transfer to new entities).

Usage:
    python scripts/train_inductive.py --version v1 --device cuda
    python scripts/train_inductive.py --version all --device cuda --bidirectional
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from data.loaders.grail_inductive import load_grail_wn18rr
from data.loaders.culturalbench import compute_text_embeddings
from data.tree_builder import build_full_tree_inputs
from data.negative_sampler import NegativeSampler
from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.knowledge.kg_embedding import KGEmbeddingSource
from pluraltree.combined.gki_tree_encoder import GKITreeEncoder
from pluraltree.combined.gki_tree_gru_cell import InjectionPoint
from training.scoring import HyperbolicLinkPredictor
from training.trainer import Trainer, TrainerConfig
from evaluation.kgc.link_prediction import evaluate_link_prediction
from evaluation.kgc.inductive_eval import grail_eval


def parse_args():
    p = argparse.ArgumentParser(description="Inductive WN18RR (GraIL splits).")
    p.add_argument("--version", type=str, default="v1",
                   help="v1 | v2 | v3 | v4 | all")
    p.add_argument("--data_dir", type=str, default="data/grail")
    p.add_argument("--embed_model", type=str, default="all-mpnet-base-v2")
    p.add_argument("--d_hidden", type=int, default=128)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--n_negative", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--lateral", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def flow_tag(args):
    return "+".join(["up"]
                    + (["down"] if args.bidirectional else [])
                    + (["lat"] if args.lateral else []))


def run_version(version: str, args) -> None:
    print("=" * 70)
    print(f"Inductive WN18RR {version} | flow={flow_tag(args)}")
    print("=" * 70)
    train_graph, ind_graph = load_grail_wn18rr(version, args.data_dir)
    n_rel = len(train_graph.relation_vocab)
    device = torch.device(args.device)

    # ---- Model (GKI off — text-only encoder transfers to new entities) ----
    train_emb = compute_text_embeddings(train_graph, model_name=args.embed_model)
    d_input = train_emb.shape[1]
    manifold = PoincareBall(c=1.0)
    ksrc = KGEmbeddingSource(num_entities=len(train_graph.id_to_entity),
                             embedding_dim=d_input, pretrained=train_emb, freeze=True)
    encoder = GKITreeEncoder(
        d_input=d_input, d_hidden=args.d_hidden, manifold=manifold, sources=[ksrc],
        injection_point=InjectionPoint.POST_AGGREGATION, inject=False,
        bidirectional=args.bidirectional, lateral=args.lateral,
    )
    predictor = HyperbolicLinkPredictor(num_relations=n_rel, d_hidden=args.d_hidden,
                                        manifold=manifold)

    config = TrainerConfig(d_hidden=args.d_hidden, n_epochs=args.n_epochs,
                           batch_size=args.batch_size, n_negative=args.n_negative,
                           lr=args.lr, device=args.device)
    trainer = Trainer(encoder=encoder, predictor=predictor, graph=train_graph,
                      node_embeddings=train_emb, config=config)
    trainer.train()

    # ---- Encode the inductive graph (new entities) and evaluate ----
    ind_emb = compute_text_embeddings(ind_graph, model_name=args.embed_model)
    ind_inputs = build_full_tree_inputs(ind_graph, ind_emb)
    encoder.eval()
    with torch.no_grad():
        h_ind = encoder(
            ind_inputs["node_features"].to(device),
            ind_inputs["node_ids"].to(device),
            ind_inputs["children_indices"],
            ind_inputs["topo_order"],
        )

    ind_neg = NegativeSampler(ind_graph.type_constraints, ind_graph.all_triples)
    filt = evaluate_link_prediction(
        h_all=h_ind, triples=ind_graph.test_triples, predictor=predictor,
        graph=ind_graph, neg_sampler=ind_neg, device=args.device,
    )
    grail = grail_eval(h_ind, ind_graph.test_triples, predictor, ind_graph,
                       n_neg=50, seed=args.seed, device=args.device)

    print(f"\nInductive {version} — filtered (rank vs all entities):")
    for k, v in filt.items():
        print(f"  {k}: {v:.4f}")
    print(f"Inductive {version} — GraIL (vs 50 negatives):")
    for k, v in grail.items():
        print(f"  {k}: {v:.4f}")

    tag = f"version={version} | flow={flow_tag(args)} | embed={args.embed_model}"
    print(f"\nRESULT | inductive {tag} | "
          f"mrr={filt.get('mrr', float('nan')):.4f} "
          f"h@1={filt.get('hits@1', float('nan')):.4f} "
          f"h@3={filt.get('hits@3', float('nan')):.4f} "
          f"h@10={filt.get('hits@10', float('nan')):.4f}")
    print(f"GRAIL  | inductive {tag} | "
          f"hits@10_50neg={grail.get('hits@10_50neg', float('nan')):.4f} "
          f"mrr_50neg={grail.get('mrr_50neg', float('nan')):.4f} "
          f"auc_pr={grail.get('auc_pr', float('nan')):.4f}")
    print("=" * 70)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    versions = ["v1", "v2", "v3", "v4"] if args.version == "all" else [args.version]
    for v in versions:
        run_version(v, args)


if __name__ == "__main__":
    main()
