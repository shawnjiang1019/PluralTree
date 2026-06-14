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
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.knowledge.base import KnowledgeSource
from pluraltree.tree_gru.aggregation import ChildAggregator
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
        gradient_checkpointing: bool = False,
        bidirectional: bool = False,
        lateral: bool = False,
        lateral_max_siblings: int = 16,
        combine_gate_bias: float = -2.0,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.manifold = manifold
        # Recompute each height bucket's activations in backward instead of
        # storing them — peak memory drops from "all buckets" to "one bucket",
        # which is what makes large graphs (e.g. WN18RR, 40K nodes) fit on a GPU.
        self.gradient_checkpointing = gradient_checkpointing
        self.input_proj = nn.Linear(d_input, d_hidden)
        self.cell = GKITreeGRUCell(
            d_hidden, d_hidden, manifold, sources, injection_point, gate_bias,
            depth_aware=depth_aware, inject=inject,
        )
        self.schedule = None
        # Cached level-batching plan (tree structure is static across forwards).
        self._plan = None
        self._plan_n = None

        # ------------------------------------------------------------------
        # Optional extra message-passing flows. Both are purely *structural*
        # (they run identically whether or not GKI is active) and are off by
        # default, so existing runs are byte-for-byte unchanged.
        #
        #   bidirectional (B4): a top-down pass so a child is conditioned on its
        #       parents — every node ends up reflecting its *ancestral context*,
        #       not just its subtree.
        #   lateral: a same-depth pass so a node is conditioned on its *siblings*
        #       (co-children of a shared parent) — peers shape each other.
        #
        # Each adds its own attention aggregator + a gated Möbius blend with a
        # negative gate bias, so the pass starts as a near no-op and the model
        # learns how much neighbour context to admit.
        # ------------------------------------------------------------------
        self.bidirectional = bidirectional
        self.lateral = lateral
        self.lateral_max_siblings = lateral_max_siblings

        if bidirectional:
            self.down_aggregator = ChildAggregator(d_hidden, manifold, attention=True)
            self.down_gate = nn.Linear(2 * d_hidden, d_hidden)
            nn.init.constant_(self.down_gate.bias, combine_gate_bias)
        if lateral:
            self.lateral_aggregator = ChildAggregator(d_hidden, manifold, attention=True)
            self.lateral_gate = nn.Linear(2 * d_hidden, d_hidden)
            nn.init.constant_(self.lateral_gate.bias, combine_gate_bias)

        # Cached plans for the extra passes (tree structure is static).
        self._down_plan = None
        self._down_plan_n = None
        self._sib_plan = None
        self._sib_plan_n = None

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
            h = self._forward_sequential(node_features, node_ids, children_indices, topo_order)
        else:
            h = self._forward_batched(node_features, node_ids, children_indices, topo_order)

        # Extra message-passing flows (no-ops unless enabled). Order: up → down →
        # lateral, so the lateral round mixes context-enriched states.
        use_ckpt = self.gradient_checkpointing and torch.is_grad_enabled()
        if self.bidirectional:
            h = self._down_pass(h, children_indices, topo_order, use_ckpt)
        if self.lateral:
            h = self._lateral_pass(h, children_indices, use_ckpt)
        return h

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

        # Only checkpoint when building a graph (training); under no_grad it
        # would just add recompute for no memory benefit.
        use_ckpt = self.gradient_checkpointing and torch.is_grad_enabled()

        for bucket in self._plan:
            nodes = bucket["nodes"]
            x_b = x_tan[nodes]            # (B, d_input)
            nids_b = node_ids[nodes]      # (B,)

            if bucket["leaf"]:
                def run_leaf(x_b, nids_b=nids_b):
                    h_agg_tan = torch.zeros(x_b.shape[0], self.d_hidden, device=x_b.device)
                    h_leaf = self.cell.tree_gru_cell.gru_step(x_b, h_agg_tan)
                    if self.cell.inject:
                        h_leaf = self.cell.gki(h_leaf, nids_b)
                    return h_leaf

                h_b = (torch_checkpoint(run_leaf, x_b, use_reentrant=False)
                       if use_ckpt else run_leaf(x_b))
            else:
                child_idx = bucket["child_idx"]                       # (B, max_k)
                mask = bucket["mask"]                                 # (max_k, B)
                h_children = h[child_idx].transpose(0, 1).contiguous()       # (max_k, B, d)
                child_nids = node_ids[child_idx].transpose(0, 1).contiguous()  # (max_k, B)

                def run_node(x_b, h_children, nids_b=nids_b, child_nids=child_nids, mask=mask):
                    return self.cell(x_b, h_children, nids_b, child_nids, mask)

                h_b = (torch_checkpoint(run_node, x_b, h_children, use_reentrant=False)
                       if use_ckpt else run_node(x_b, h_children))

            # Out-of-place scatter keeps autograd clean across buckets.
            h = h.index_copy(0, nodes, h_b)

        return h

    # ------------------------------------------------------------------
    # Extra message-passing flows (top-down B4 + lateral same-depth)
    # ------------------------------------------------------------------
    def _gated_combine(self, h_self: Tensor, h_msg: Tensor, gate: nn.Linear) -> Tensor:
        """Gated Möbius blend of a node's own state with a neighbour message.

        Both are mapped to the tangent space at the origin; a learned per-dim gate
        ``g = σ(W[log h_self ; log h_msg])`` interpolates, then we map back to the
        ball. The gate bias starts negative, so ``g ≈ 0`` (keep ``h_self``) and the
        pass is initially a near no-op the model learns to open.
        """
        a = self.manifold.log_map_zero(h_self)
        b = self.manifold.log_map_zero(h_msg)
        g = torch.sigmoid(gate(torch.cat([a, b], dim=-1)))
        return self.manifold.exp_map_zero((1.0 - g) * a + g * b)

    @staticmethod
    def _invert_children(children_indices: list[list[int]]) -> list[list[int]]:
        """parents[v] = list of v's parents (children_indices inverted)."""
        parents: list[list[int]] = [[] for _ in children_indices]
        for parent, kids in enumerate(children_indices):
            for c in kids:
                parents[c].append(parent)
        return parents

    def _build_parent_plan(
        self,
        children_indices: list[list[int]],
        topo_order: list[int],
        device: torch.device,
    ) -> list[dict]:
        """Group nodes into depth buckets for the top-down pass.

        depth(root) = 0; depth(v) = 1 + max(depth(parent)). Processing buckets in
        increasing depth guarantees every parent is updated before its children,
        and nodes within a bucket are mutually independent (safe to batch). Roots
        (no parents) are skipped — they keep their up-pass state.
        """
        parents = self._invert_children(children_indices)
        N = len(children_indices)
        depth = [0] * N
        for v in reversed(topo_order):  # parents (incl. root) before children
            depth[v] = 1 + max((depth[p] for p in parents[v]), default=-1)

        buckets: dict[int, list[int]] = defaultdict(list)
        for v in range(N):
            buckets[depth[v]].append(v)

        plan: list[dict] = []
        for d_level in sorted(buckets):
            nodes = buckets[d_level]
            node_t = torch.tensor(nodes, dtype=torch.long, device=device)

            if d_level == 0:  # roots: no parents
                plan.append({"root": True, "nodes": node_t, "par_idx": None, "mask": None})
                continue

            B = len(nodes)
            max_p = max(len(parents[v]) for v in nodes)
            par_idx = torch.zeros(B, max_p, dtype=torch.long, device=device)
            mask = torch.zeros(max_p, B, dtype=torch.bool, device=device)
            for bi, v in enumerate(nodes):
                for pi, p in enumerate(parents[v]):
                    par_idx[bi, pi] = p
                    mask[pi, bi] = True
            plan.append({"root": False, "nodes": node_t, "par_idx": par_idx, "mask": mask})

        return plan

    def _build_sibling_plan(
        self,
        children_indices: list[list[int]],
        device: torch.device,
    ) -> dict | None:
        """One synchronous round: each node aggregates its siblings (co-children
        of a shared parent), capped at ``lateral_max_siblings`` for memory.

        Truncation is deterministic (lowest ids first) so the static encoder is
        reproducible across forwards. Nodes with no siblings are excluded.
        """
        N = len(children_indices)
        sib: list[set[int]] = [set() for _ in range(N)]
        for kids in children_indices:
            if len(kids) < 2:
                continue
            kset = set(kids)
            for c in kids:
                sib[c].update(kset)
                sib[c].discard(c)

        nodes = [v for v in range(N) if sib[v]]
        if not nodes:
            return None

        cap = self.lateral_max_siblings
        sib_lists = [sorted(sib[v])[:cap] for v in nodes]
        M = len(nodes)
        max_s = max(len(s) for s in sib_lists)

        node_t = torch.tensor(nodes, dtype=torch.long, device=device)
        sib_idx = torch.zeros(M, max_s, dtype=torch.long, device=device)
        mask = torch.zeros(max_s, M, dtype=torch.bool, device=device)
        for mi, slist in enumerate(sib_lists):
            for si, s in enumerate(slist):
                sib_idx[mi, si] = s
                mask[si, mi] = True
        return {"nodes": node_t, "sib_idx": sib_idx, "mask": mask}

    def _down_pass(
        self,
        h: Tensor,
        children_indices: list[list[int]],
        topo_order: list[int],
        use_ckpt: bool,
    ) -> Tensor:
        """Top-down pass: each node blends its up-state with an aggregate of its
        parents' (already updated) down-states, so context flows root → leaves."""
        N = h.shape[0]
        if self._down_plan is None or self._down_plan_n != N:
            self._down_plan = self._build_parent_plan(children_indices, topo_order, h.device)
            self._down_plan_n = N

        for bucket in self._down_plan:
            if bucket["root"]:
                continue
            nodes = bucket["nodes"]
            mask = bucket["mask"]
            h_self = h[nodes]
            h_parents = h[bucket["par_idx"]].transpose(0, 1).contiguous()  # (max_p, B, d)

            def run_down(h_self, h_parents, mask=mask):
                m = self.down_aggregator(h_parents, mask)
                return self._gated_combine(h_self, m, self.down_gate)

            h_b = (torch_checkpoint(run_down, h_self, h_parents, use_reentrant=False)
                   if use_ckpt else run_down(h_self, h_parents))
            h = h.index_copy(0, nodes, h_b)

        return h

    def _lateral_pass(
        self,
        h: Tensor,
        children_indices: list[list[int]],
        use_ckpt: bool,
    ) -> Tensor:
        """Lateral pass: one synchronous round where each node blends its state
        with an aggregate of its siblings' (pre-update) states."""
        N = h.shape[0]
        if self._sib_plan is None or self._sib_plan_n != N:
            self._sib_plan = self._build_sibling_plan(children_indices, h.device)
            self._sib_plan_n = N

        plan = self._sib_plan
        if plan is None:  # tree with no siblings anywhere (e.g. a path)
            return h

        nodes = plan["nodes"]
        mask = plan["mask"]
        h_self = h[nodes]
        h_sibs = h[plan["sib_idx"]].transpose(0, 1).contiguous()  # (max_s, M, d)

        def run_lat(h_self, h_sibs, mask=mask):
            m = self.lateral_aggregator(h_sibs, mask)
            return self._gated_combine(h_self, m, self.lateral_gate)

        h_b = (torch_checkpoint(run_lat, h_self, h_sibs, use_reentrant=False)
               if use_ckpt else run_lat(h_self, h_sibs))
        return h.index_copy(0, nodes, h_b)
