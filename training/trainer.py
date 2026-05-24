"""Training loop for the Hyperbolic Tree-GRU + GKI model on CulturalBench.

Workflow per epoch:
  1. Encode the full tree once → hidden states for all entities
  2. For each batch of triples:
     a. Look up h_s, h_o for positive and negative triples
     b. Score with HyperbolicLinkPredictor
     c. Compute margin loss + gate sparsity regularization
     d. Backprop + Riemannian optimizer step
  3. Evaluate on val set every eval_every steps
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.culturalbench import CulturalGraph
from data.collate import TripleBatchSampler
from data.negative_sampler import NegativeSampler
from data.tree_builder import build_full_tree_inputs
from training.losses import gate_sparsity_loss
from training.scoring import HyperbolicLinkPredictor
from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.combined.gki_tree_encoder import GKITreeEncoder
from pluraltree.combined.knowledge_schedule import KnowledgeSchedule
from pluraltree.utils.riemannian_optim import build_optimizer
from evaluation.link_prediction import evaluate_link_prediction


@dataclass
class TrainerConfig:
    d_hidden:       int   = 64
    n_epochs:       int   = 50
    batch_size:     int   = 128
    n_negative:     int   = 10
    margin:         float = 1.0
    gate_sparsity_weight: float = 0.01
    lr:             float = 1e-3
    lr_manifold:    float = 1e-2
    eval_every:     int   = 200   # steps
    log_every:      int   = 50    # steps
    warmup1:        int   = 500
    warmup2:        int   = 1500
    gate_bias_init: float = -2.0
    device:         str   = "cpu"


class Trainer:
    def __init__(
        self,
        encoder: GKITreeEncoder,
        predictor: HyperbolicLinkPredictor,
        graph: CulturalGraph,
        node_embeddings: Tensor,
        config: TrainerConfig,
    ):
        self.encoder    = encoder
        self.predictor  = predictor
        self.graph      = graph
        self.config     = config
        self.device     = torch.device(config.device)

        # Move models and embeddings to device
        self.encoder.to(self.device)
        self.predictor.to(self.device)
        self.node_embeddings = node_embeddings.to(self.device)

        # Knowledge curriculum
        self.schedule = KnowledgeSchedule(
            warmup1=config.warmup1,
            warmup2=config.warmup2,
            initial_bias=config.gate_bias_init,
        )
        self.encoder.set_schedule(self.schedule)

        # Negative sampler
        self.neg_sampler = NegativeSampler(
            type_constraints=graph.type_constraints,
            all_triples=graph.all_triples,
        )

        # Batch sampler
        self.batch_sampler = TripleBatchSampler(
            triples=graph.train_triples,
            sampler=self.neg_sampler,
            batch_size=config.batch_size,
            n_negative=config.n_negative,
        )

        # Optimizers
        all_params = list(encoder.parameters()) + list(predictor.parameters())
        model_wrapper = nn.ModuleList([encoder, predictor])
        self.optimizer, self.max_grad_norm = build_optimizer(
            model_wrapper,
            lr=config.lr,
            lr_manifold=config.lr_manifold,
        )

        # Tree inputs (static — built once)
        self.tree_inputs = build_full_tree_inputs(graph, self.node_embeddings)

        self.global_step = 0
        self.best_val_mrr = 0.0

    def encode_tree(self) -> Tensor:
        """Run the full tree encoder and return all entity hidden states."""
        return self.encoder(
            node_features    = self.tree_inputs["node_features"],
            node_ids         = self.tree_inputs["node_ids"].to(self.device),
            children_indices = self.tree_inputs["children_indices"],
            topo_order       = self.tree_inputs["topo_order"],
            training_step    = self.global_step,
        )  # (N, d_hidden) on Poincaré ball

    def train_step(self, batch: dict, h_all: Tensor) -> dict[str, float]:
        """Single training step.

        Args:
            batch: dict from TripleBatchSampler
            h_all: (N, d_hidden) all entity hidden states

        Returns:
            dict of scalar loss values for logging
        """
        self.optimizer.zero_grad()

        pos_s = batch["pos_s"].to(self.device)
        pos_r = batch["pos_r"].to(self.device)
        pos_o = batch["pos_o"].to(self.device)
        neg_s = batch["neg_s"].to(self.device)
        neg_r = batch["neg_r"].to(self.device)
        neg_o = batch["neg_o"].to(self.device)

        # Look up hidden states for triple entities
        h_pos_s = h_all[pos_s]   # (B, d)
        h_pos_o = h_all[pos_o]   # (B, d)
        h_neg_s = h_all[neg_s]   # (B*K, d)
        h_neg_o = h_all[neg_o]   # (B*K, d)

        # Score positives and negatives using relation-aware predictor
        score_pos = self.predictor.score(h_pos_s, pos_r, h_pos_o)  # (B,)

        K = self.config.n_negative
        if neg_s.shape[0] > 0:
            score_neg = self.predictor.score(h_neg_s, neg_r, h_neg_o)  # (B*K,)
            score_pos_rep = score_pos.repeat_interleave(K)[:score_neg.shape[0]]
            loss_lp = F.relu(self.config.margin + score_neg - score_pos_rep).mean()
        else:
            loss_lp = torch.zeros(1, device=self.device, requires_grad=True)

        # Gate sparsity — penalize always-open gates
        # Approximate by measuring mean hidden state norm (proxy for gate activity)
        loss_sparse = gate_sparsity_loss(
            self.encoder.cell.gki.gate.W_g.weight.abs().mean().unsqueeze(0)
        )

        loss = loss_lp + self.config.gate_sparsity_weight * loss_sparse

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.encoder.parameters() if p.grad is not None],
            self.max_grad_norm,
        )

        self.optimizer.step()
        self.global_step += 1

        return {
            "loss":        loss.item(),
            "loss_lp":     loss_lp.item(),
            "loss_sparse": loss_sparse.item(),
        }

    def train(self) -> None:
        """Full training loop."""
        print(f"Training for {self.config.n_epochs} epochs")
        print(f"  Entities: {len(self.graph.id_to_entity)}")
        print(f"  Train triples: {len(self.graph.train_triples)}")
        print(f"  Val triples:   {len(self.graph.val_triples)}")
        print(f"  Device: {self.device}")

        for epoch in range(self.config.n_epochs):
            self.encoder.train()
            self.predictor.train()

            epoch_losses: list[float] = []
            t0 = time.time()

            for batch in self.batch_sampler:
                h_all = self.encode_tree()
                metrics = self.train_step(batch, h_all)
                epoch_losses.append(metrics["loss"])

                if self.global_step % self.config.log_every == 0:
                    gate_bias = self.schedule.get_gate_bias(self.global_step)
                    print(
                        f"  step {self.global_step:5d} | "
                        f"loss {metrics['loss']:.4f} | "
                        f"lp {metrics['loss_lp']:.4f} | "
                        f"gate_bias {gate_bias:.2f}"
                    )

                if self.global_step % self.config.eval_every == 0:
                    val_metrics = self.evaluate("val")
                    print(
                        f"  [eval] step {self.global_step} | "
                        f"MRR {val_metrics['mrr']:.4f} | "
                        f"H@1 {val_metrics['hits@1']:.4f} | "
                        f"H@10 {val_metrics['hits@10']:.4f}"
                    )
                    if val_metrics["mrr"] > self.best_val_mrr:
                        self.best_val_mrr = val_metrics["mrr"]
                        print(f"  [best] new best val MRR: {self.best_val_mrr:.4f}")

            elapsed = time.time() - t0
            mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            print(f"Epoch {epoch+1:3d}/{self.config.n_epochs} | "
                  f"mean loss {mean_loss:.4f} | {elapsed:.1f}s")

    def evaluate(self, split: str = "val") -> dict[str, float]:
        """Run link prediction evaluation on val or test split."""
        self.encoder.eval()
        self.predictor.eval()
        triples = self.graph.val_triples if split == "val" else self.graph.test_triples
        with torch.no_grad():
            h_all = self.encode_tree()
        return evaluate_link_prediction(
            h_all         = h_all,
            triples       = triples,
            predictor     = self.predictor,
            graph         = self.graph,
            neg_sampler   = self.neg_sampler,
            device        = self.device,
        )
