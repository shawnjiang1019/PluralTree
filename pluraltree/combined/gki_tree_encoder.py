"""Full tree encoder with GKI — bottom-up recursion with knowledge injection.

Two execution paths produce identical results:
  - ``_forward_sequential``: the reference implementation, one node at a time.
  - ``_forward_batched``   : processes all nodes at the same tree height in a
        single batched call. Far fewer GPU kernel launches (≈ tree-depth calls
        instead of one per node), which dominates runtime on small graphs.

The batched path relies on the cell and aggregator already supporting a batch
dimension ((K, batch, d) with masks), so the per-node math is unchanged.
"""

from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.knowledge.base import KnowledgeSource
from .gki_tree_gru_cell import GKITreeGRUCell, InjectionPoint
from .knowledge_schedule import KnowledgeSchedule


class GKITreeEncoder(nn.Module):
    """Encodes a rooted tree with Hyperbolic Tree-GRU + Gated Knowledge Injection.

    Bottom-up recursion with depth-aware knowledge injection at each node.
    Supports a knowledge curriculum that gradually enables injection during training.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        manifold: PoincareBall,
        sources: list[KnowledgeSource],
        injection_point: InjectionPoint = InjectionPoint.POST_AGGREGATION,
        gate_bias: float = -2.0,
        depth_aware: bool = True,
        inject: bool = True,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.manifold = manifold
        self.input_proj = nn.Linear(d_input, d_hidden)
        self.cell = GKITreeGRUCell(
            d_hidden, d_hidden, manifold, sources, injection_point, gate_bias,
            depth_aware=depth_aware, inject=inject,
        )
        self.schedule = None
        # Cached level-batching plan (tree structure is static across forwards).
        self._plan = None
        self._plan_n = None

    def set_schedule(self, schedule: KnowledgeSchedule) -> None:
        self.schedule = schedule

    def _apply_schedule(self, training_step: int) -> None:
        """Pin the gate bias to the curriculum value for this step (in training)."""
        if self.schedule is not None and self.training:
            bias = self.schedule.get_gate_bias(training_step)
            for p in self.cell.gki.gate.parameters():
                if p.shape == self.cell.gki.gate.W_g.bias.shape:
                    with torch.no_grad():
                        p.fill_(bias)

    def forward(
        self,
        node_features: Tensor,
        node_ids: Tensor,
        children_indices: list[list[int]],
        topo_order: list[int],
        training_step: int = 0,
        mode: str = "batched",
    ) -> Tensor:
        """Encode a tree bottom-up with knowledge injection.

        Args:
            node_features: (N, d_input) features for all N nodes
            node_ids: (N,) node identifiers for knowledge lookup
            children_indices: children_indices[i] = list of child indices for node i
            topo_order: topological order (leaves first)
            training_step: current step for knowledge curriculum
            mode: "batched" (default, fast) or "sequential" (reference)

        Returns:
            (N, d_hidden) hidden states on the Poincaré ball
        """
        self._apply_schedule(training_step)
        if mode == "sequential":
            return self._forward_sequential(node_features, node_ids, children_indices, topo_order)
        return self._forward_batched(node_features, node_ids, children_indices, topo_order)

    # ------------------------------------------------------------------
    # Reference path: one node at a time
    # ------------------------------------------------------------------
    def _forward_sequential(
        self,
        node_features: Tensor,
        node_ids: Tensor,
        children_indices: list[list[int]],
        topo_order: list[int],
    ) -> Tensor:
        N = node_features.shape[0]
        device = node_features.device

        x_tan = self.input_proj(node_features)
        h = torch.zeros(N, self.d_hidden, device=device)

        for idx in topo_order:
            children = children_indices[idx]

            if len(children) == 0:
                h_agg_tan = torch.zeros(1, self.d_hidden, device=device)
                h_node = self.cell.tree_gru_cell.gru_step(
                    x_tan[idx : idx + 1], h_agg_tan
                )
                if self.cell.inject:
                    h_node = self.cell.gki(h_node, node_ids[idx : idx + 1])
            else:
                h_ch = torch.stack([h[c] for c in children], dim=0).unsqueeze(1)
                mask = torch.ones(len(children), 1, dtype=torch.bool, device=device)
                child_nids = torch.stack([node_ids[c] for c in children]).unsqueeze(1)

                h_node = self.cell(
                    x_tan[idx : idx + 1],
                    h_ch,
                    node_ids[idx : idx + 1],
                    child_nids,
                    mask,
                )

            h[idx] = h_node.squeeze(0)

        return h

    # ------------------------------------------------------------------
    # Fast path: all nodes at the same height in one batched call
    # ------------------------------------------------------------------
    def _build_plan(
        self,
        children_indices: list[list[int]],
        topo_order: list[int],
        device: torch.device,
    ) -> list[dict]:
        """Group nodes into height buckets for batched processing.

        height(leaf) = 0; height(v) = 1 + max(height(child)). Every child of v has
        strictly smaller height, so processing buckets in increasing height order
        guarantees children are computed first, and nodes within a bucket are
        mutually independent (safe to batch).
        """
        N = len(children_indices)
        height = [0] * N
        for v in topo_order:  # leaves first → children done before parents
            ch = children_indices[v]
            height[v] = 1 + max((height[c] for c in ch), default=-1)

        buckets: dict[int, list[int]] = defaultdict(list)
        for v in range(N):
            buckets[height[v]].append(v)

        plan: list[dict] = []
        for h_level in sorted(buckets):
            nodes = buckets[h_level]
            node_t = torch.tensor(nodes, dtype=torch.long, device=device)

            if h_level == 0:  # leaves: no children
                plan.append({"leaf": True, "nodes": node_t, "child_idx": None, "mask": None})
                continue

            B = len(nodes)
            max_k = max(len(children_indices[v]) for v in nodes)
            child_idx = torch.zeros(B, max_k, dtype=torch.long, device=device)
            mask = torch.zeros(max_k, B, dtype=torch.bool, device=device)
            for bi, v in enumerate(nodes):
                for ci, c in enumerate(children_indices[v]):
                    child_idx[bi, ci] = c
                    mask[ci, bi] = True
            plan.append({"leaf": False, "nodes": node_t, "child_idx": child_idx, "mask": mask})

        return plan

    def _forward_batched(
        self,
        node_features: Tensor,
        node_ids: Tensor,
        children_indices: list[list[int]],
        topo_order: list[int],
    ) -> Tensor:
        N = node_features.shape[0]
        device = node_features.device

        if self._plan is None or self._plan_n != N:
            self._plan = self._build_plan(children_indices, topo_order, device)
            self._plan_n = N

        x_tan = self.input_proj(node_features)
        h = torch.zeros(N, self.d_hidden, device=device)

        for bucket in self._plan:
            nodes = bucket["nodes"]
            x_b = x_tan[nodes]            # (B, d_input)
            nids_b = node_ids[nodes]      # (B,)

            if bucket["leaf"]:
                h_agg_tan = torch.zeros(nodes.shape[0], self.d_hidden, device=device)
                h_b = self.cell.tree_gru_cell.gru_step(x_b, h_agg_tan)
                if self.cell.inject:
                    h_b = self.cell.gki(h_b, nids_b)
            else:
                child_idx = bucket["child_idx"]                       # (B, max_k)
                mask = bucket["mask"]                                 # (max_k, B)
                h_children = h[child_idx].transpose(0, 1).contiguous()       # (max_k, B, d)
                child_nids = node_ids[child_idx].transpose(0, 1).contiguous()  # (max_k, B)
                h_b = self.cell(x_b, h_children, nids_b, child_nids, mask)

            # Out-of-place scatter keeps autograd clean across buckets.
            h = h.index_copy(0, nodes, h_b)

        return h
