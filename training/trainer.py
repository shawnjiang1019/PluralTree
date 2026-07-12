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

from data.loaders.culturalbench import CulturalGraph
from data.collate import TripleBatchSampler
from data.negative_sampler import NegativeSampler
from data.tree_builder import build_full_tree_inputs
from training.losses import (
    gate_sparsity_loss,
    geodesic_separation_loss,
    boundary_penalty,
)
from training.scoring import HyperbolicLinkPredictor
from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.combined.gki_tree_encoder import GKITreeEncoder
from pluraltree.combined.knowledge_schedule import KnowledgeSchedule
from pluraltree.utils.riemannian_optim import build_optimizer
from evaluation.kgc.link_prediction import evaluate_link_prediction


@dataclass
class TrainerConfig:
    d_hidden:       int   = 64
    n_epochs:       int   = 50
    batch_size:     int   = 128
    n_negative:     int   = 10
    margin:         float = 1.0
    gate_sparsity_weight: float = 0.01
    lambda_struct:  float = 0.0   # weight on the hierarchy structure-fidelity loss
    struct_margin:  float = 1.0   # margin for that loss (parent closer than non-ancestor)
    # --- Diversity-as-objective (anti-collapse) + robust deterministic decode ---
    lambda_div:      float = 0.0  # sibling-separation floor (faithful diversity)
    div_margin:      float = 1.0  # min geodesic distance between siblings
    lambda_boundary: float = 0.0  # boundary penalty (keep mass off the rim)
    lambda_decode:   float = 0.0  # codebook-separation floor (robust NN decode)
    decode_margin:   float = 1.0  # min geodesic gap to a node's nearest other
    encode_every:   int   = 1     # re-encode the full tree every N steps (1 = every batch)
    metrics_csv:    str | None = None   # if set, append per-eval val metrics here
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

        # Structure-fidelity loss edges: (child, parent) pairs from the tree.
        # Used only when config.lambda_struct > 0 — a Poincaré-style hierarchy
        # margin loss (pull parents close, push non-ancestors away). Leakage-safe:
        # these are backbone/ontology edges, not held-out target triples.
        ch, par = [], []
        for p, kids in enumerate(graph.children_indices):
            for c in kids:
                ch.append(c)
                par.append(p)
        self._struct_child  = torch.tensor(ch,  dtype=torch.long, device=self.device)
        self._struct_parent = torch.tensor(par, dtype=torch.long, device=self.device)
        self._n_struct_edges = len(ch)

        # Sibling pairs for the diversity floor (consecutive co-children of each
        # parent — O(total children), bounds high-fan-out parents). Used only when
        # config.lambda_div > 0.
        si, sj = [], []
        for kids in graph.children_indices:
            for a, b in zip(kids[:-1], kids[1:]):
                si.append(a)
                sj.append(b)
        self._sib_i = torch.tensor(si, dtype=torch.long, device=self.device)
        self._sib_j = torch.tensor(sj, dtype=torch.long, device=self.device)
        self._n_sib_pairs = len(si)

        self.global_step = 0
        self.best_val_mrr = 0.0

        # Optional CSV log of the validation learning curve (for offline plotting).
        self._csv_writer = None
        if config.metrics_csv:
            import csv
            self._csv_file = open(config.metrics_csv, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["step", "epoch", "val_mrr", "hits@1", "hits@3", "hits@10"]
            )
            self._csv_file.flush()

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

        # Structure-fidelity loss — train the *geometry*, not just the link.
        # Hierarchy margin: a node's parent should be closer than a random
        # non-ancestor by `struct_margin`. Optimises subtree_ap / ancestor_auc /
        # depth_radius_rho (see docs/EVALUATION.md). Off when lambda_struct == 0.
        manifold = self.encoder.manifold
        loss_struct_t = torch.zeros((), device=self.device)
        if self.config.lambda_struct > 0.0 and self._n_struct_edges > 0:
            n_nodes = h_all.shape[0]
            B = min(self.config.batch_size, self._n_struct_edges)
            e = torch.randint(self._n_struct_edges, (B,), device=self.device)
            c = self._struct_child[e]
            p = self._struct_parent[e]
            neg = torch.randint(n_nodes, (B,), device=self.device)
            # Resample degenerate negatives that hit the child or its parent
            # (unconditional where — avoids a per-step .any() GPU sync).
            bad = (neg == c) | (neg == p)
            neg = torch.where(
                bad, torch.randint(n_nodes, (B,), device=self.device), neg
            )
            d_pos = manifold.distance(h_all[c], h_all[p]).squeeze(-1)
            d_neg = manifold.distance(h_all[c], h_all[neg]).squeeze(-1)
            loss_struct = F.relu(self.config.struct_margin + d_pos - d_neg).mean()
            loss = loss + self.config.lambda_struct * loss_struct
            loss_struct_t = loss_struct.detach()

        # Diversity-as-objective: sibling-separation floor (anti-collapse). Keeps
        # genuine forks measurable; link-pred/struct still pull siblings close, so
        # this only sets a floor. Validate via branch_divergence_rel_mean (up) with
        # sibling_ratio off 0 (not collapsed).
        loss_div_t = torch.zeros((), device=self.device)
        if self.config.lambda_div > 0.0 and self._n_sib_pairs > 0:
            B = min(self.config.batch_size, self._n_sib_pairs)
            e = torch.randint(self._n_sib_pairs, (B,), device=self.device)
            loss_div = geodesic_separation_loss(
                h_all[self._sib_i[e]], h_all[self._sib_j[e]],
                manifold, self.config.div_margin,
            )
            loss = loss + self.config.lambda_div * loss_div
            loss_div_t = loss_div.detach()

        # Boundary penalty — keep mass off the rim (anti-saturation).
        if self.config.lambda_boundary > 0.0:
            loss = loss + self.config.lambda_boundary * boundary_penalty(h_all, manifold)

        # Robust deterministic decode: a margin around each codebook entry so an
        # *inexact* query (bridge round-trip / perturbation) still argmins to the
        # right id. Decode itself is deterministic regardless; this widens the basin.
        loss_dec_t = torch.zeros((), device=self.device)
        if self.config.lambda_decode > 0.0:
            n_nodes = h_all.shape[0]
            B = min(self.config.batch_size, n_nodes)
            M = 32                                       # sampled negatives per anchor
            anc = torch.randint(n_nodes, (B,), device=self.device)
            neg = torch.randint(n_nodes, (B, M), device=self.device)
            d = manifold.distance(h_all[anc].unsqueeze(1), h_all[neg]).squeeze(-1)  # (B, M)
            d = d.masked_fill(neg == anc.unsqueeze(1), float("inf"))
            nearest = d.min(dim=1).values
            loss_dec = F.relu(self.config.decode_margin - nearest).mean()
            loss = loss + self.config.lambda_decode * loss_dec
            loss_dec_t = loss_dec.detach()

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.encoder.parameters() if p.grad is not None],
            self.max_grad_norm,
        )

        self.optimizer.step()
        self.global_step += 1

        # Return detached tensors (no .item() here) so the hot path has zero
        # GPU->CPU syncs; the caller materializes scalars only when logging.
        return {
            "loss":        loss.detach().reshape(()),
            "loss_lp":     loss_lp.detach().reshape(()),
            "loss_sparse": loss_sparse.detach().reshape(()),
            "loss_struct": loss_struct_t,
            "loss_div":    loss_div_t,
            "loss_dec":    loss_dec_t,
        }

    def train(self) -> None:
        """Full training loop."""
        print(f"Training for {self.config.n_epochs} epochs")
        print(f"  Entities: {len(self.graph.id_to_entity)}")
        print(f"  Train triples: {len(self.graph.train_triples)}")
        print(f"  Val triples:   {len(self.graph.val_triples)}")
        print(f"  Device: {self.device}")
        if self.config.encode_every > 1:
            print(f"  Encoding the tree every {self.config.encode_every} steps "
                  f"(reusing a detached copy in between)")

        h_all = None  # persists across batches; refreshed every encode_every steps
        for epoch in range(self.config.n_epochs):
            self.encoder.train()
            self.predictor.train()

            epoch_loss_sum = torch.zeros((), device=self.device)
            epoch_n = 0
            t0 = time.time()

            for batch in self.batch_sampler:
                # Re-encode the *static* tree only every encode_every steps; between
                # refreshes reuse a detached copy. The predictor still trains every
                # step; the encoder updates on encode steps (full grad). This
                # amortizes the dominant per-batch full-tree encode on large graphs.
                if h_all is None or self.global_step % self.config.encode_every == 0:
                    h_all = self.encode_tree()
                    h_step = h_all
                else:
                    h_step = h_all.detach()
                metrics = self.train_step(batch, h_step)
                epoch_loss_sum = epoch_loss_sum + metrics["loss"]
                epoch_n += 1

                if self.global_step % self.config.log_every == 0:
                    gate_bias = self.schedule.get_gate_bias(self.global_step)
                    print(
                        f"  step {self.global_step:5d} | "
                        f"loss {metrics['loss'].item():.4f} | "
                        f"lp {metrics['loss_lp'].item():.4f} | "
                        f"struct {metrics['loss_struct'].item():.4f} | "
                        f"div {metrics['loss_div'].item():.4f} | "
                        f"dec {metrics['loss_dec'].item():.4f} | "
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
                    if self._csv_writer is not None:
                        self._csv_writer.writerow([
                            self.global_step, epoch + 1,
                            val_metrics["mrr"], val_metrics["hits@1"],
                            val_metrics["hits@3"], val_metrics["hits@10"],
                        ])
                        self._csv_file.flush()
                    if val_metrics["mrr"] > self.best_val_mrr:
                        self.best_val_mrr = val_metrics["mrr"]
                        print(f"  [best] new best val MRR: {self.best_val_mrr:.4f}")

            elapsed = time.time() - t0
            mean_loss = (epoch_loss_sum / max(epoch_n, 1)).item()   # one sync / epoch
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
