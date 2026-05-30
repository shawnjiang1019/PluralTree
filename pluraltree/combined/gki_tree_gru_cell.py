"""Combined GKI + Tree-GRU cell with configurable injection points."""

from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.knowledge.base import KnowledgeSource
from pluraltree.gki.injector import GKIInjector
from pluraltree.tree_gru.cell import HyperbolicTreeGRUCell
from .depth_aware_gate import DepthAwareHyperbolicGate


class InjectionPoint(Enum):
    PRE_AGGREGATION = "pre_agg"
    POST_AGGREGATION = "post_agg"
    POST_GRU = "post_gru"
    DUAL = "dual"


class GKITreeGRUCell(nn.Module):
    """Hyperbolic Tree-GRU cell augmented with Gated Knowledge Injection.

    Integrates external knowledge at configurable points in the recursion:
      - PRE_AGGREGATION: inject into each child before aggregating
      - POST_AGGREGATION: inject after aggregation, before GRU gates (default)
      - POST_GRU: inject after the full GRU update
      - DUAL: both pre-aggregation and post-GRU
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        manifold: PoincareBall,
        sources: list[KnowledgeSource],
        injection_point: InjectionPoint = InjectionPoint.POST_AGGREGATION,
        gate_bias: float = -2.0,
        child_attention: bool = True,
        depth_aware: bool = True,
        inject: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.injection_point = injection_point
        # When False, all knowledge injection is bypassed (pure Tree-GRU ablation).
        self.inject = inject
        self.tree_gru_cell = HyperbolicTreeGRUCell(d_input, d_hidden, manifold, child_attention)

        # GKI injector (creates a plain HyperbolicGate by default)
        self.gki = GKIInjector(d_hidden, sources, manifold, gate_bias)

        # Optionally replace the standard gate with the depth-aware variant.
        # Ablation A1: depth_aware=False keeps the plain HyperbolicGate so we can
        # test whether conditioning the gate on radius ρ actually matters.
        if depth_aware:
            self.gki.gate = DepthAwareHyperbolicGate(d_hidden, d_hidden, manifold, gate_bias)

    def forward(
        self,
        x_v: Tensor,
        h_children: Tensor,
        node_ids: Tensor,
        child_node_ids: Tensor | None = None,
        children_mask: Tensor | None = None,
    ) -> Tensor:
        """Process one node with knowledge injection.

        Args:
            x_v: (batch, d_input) node features (tangent space)
            h_children: (K, batch, d_hidden) children states on the ball
            node_ids: (batch,) node IDs for this node's knowledge lookup
            child_node_ids: (K, batch) node IDs for children (needed for pre-agg injection)
            children_mask: (K, batch) boolean mask for valid children

        Returns:
            (batch, d_hidden) hidden state on the Poincaré ball
        """
        # Pre-aggregation injection
        if self.inject and self.injection_point in (InjectionPoint.PRE_AGGREGATION, InjectionPoint.DUAL):
            if child_node_ids is not None:
                K = h_children.shape[0]
                injected = []
                for i in range(K):
                    injected.append(self.gki(h_children[i], child_node_ids[i]))
                h_children = torch.stack(injected, dim=0)

        # Aggregate children
        h_agg = self.tree_gru_cell.aggregator(h_children, children_mask)

        # Post-aggregation injection
        if self.inject and self.injection_point == InjectionPoint.POST_AGGREGATION:
            h_agg = self.gki(h_agg, node_ids)

        # GRU step
        h_agg_tan = self.manifold.log_map_zero(h_agg)
        h_v = self.tree_gru_cell.gru_step(x_v, h_agg_tan)

        # Post-GRU injection
        if self.inject and self.injection_point in (InjectionPoint.POST_GRU, InjectionPoint.DUAL):
            h_v = self.gki(h_v, node_ids)

        return h_v
