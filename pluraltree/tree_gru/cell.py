"""Hyperbolic Tree-GRU cell operating in the Poincaré ball."""

import torch
import torch.nn as nn
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall
from .aggregation import ChildAggregator


class HyperbolicTreeGRUCell(nn.Module):
    """A single recursion step of the Hyperbolic Tree-GRU.

    Given node input features x_v and children hidden states {h_c1, ..., h_cN}:
    1. Aggregate children via Möbius weighted midpoint -> h_agg
    2. Compute GRU gates (update z, reset r) in tangent space
    3. Compute candidate state in tangent space
    4. Combine via gated update, map back to Poincaré ball
    """

    def __init__(self, d_input: int, d_hidden: int, manifold: PoincareBall, child_attention: bool = True):
        super().__init__()
        self.d_hidden = d_hidden
        self.manifold = manifold
        self.aggregator = ChildAggregator(d_hidden, manifold, attention=child_attention)

        # GRU gates operate in tangent space (Euclidean)
        self.W_z = nn.Linear(d_hidden + d_input, d_hidden)  # update gate
        self.W_r = nn.Linear(d_hidden + d_input, d_hidden)  # reset gate
        self.W_n = nn.Linear(d_hidden + d_input, d_hidden)  # candidate

    def gru_step(self, x_v_tan: Tensor, h_agg_tan: Tensor) -> Tensor:
        """GRU computation in tangent space.

        Args:
            x_v_tan: (batch, d_input) node features in tangent space
            h_agg_tan: (batch, d_hidden) aggregated children in tangent space

        Returns:
            (batch, d_hidden) new hidden state on the Poincaré ball
        """
        combined = torch.cat([h_agg_tan, x_v_tan], dim=-1)

        z = torch.sigmoid(self.W_z(combined))  # update gate
        r = torch.sigmoid(self.W_r(combined))  # reset gate

        candidate_input = torch.cat([r * h_agg_tan, x_v_tan], dim=-1)
        n = torch.tanh(self.W_n(candidate_input))  # candidate

        h_new_tan = (1.0 - z) * h_agg_tan + z * n
        return self.manifold.exp_map_zero(h_new_tan)

    def forward(
        self,
        x_v: Tensor,
        h_children: Tensor,
        children_mask: Tensor | None = None,
    ) -> Tensor:
        """Process one node in the tree.

        Args:
            x_v: (batch, d_input) node input features (in tangent space)
            h_children: (K, batch, d_hidden) children hidden states on the ball
            children_mask: (K, batch) boolean mask for valid children

        Returns:
            (batch, d_hidden) hidden state for this node, on the ball
        """
        # Aggregate children on the manifold
        h_agg = self.aggregator(h_children, children_mask)

        # Map aggregated state to tangent space for GRU computation
        h_agg_tan = self.manifold.log_map_zero(h_agg)

        return self.gru_step(x_v, h_agg_tan)
