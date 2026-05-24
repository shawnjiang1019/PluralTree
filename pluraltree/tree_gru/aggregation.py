"""Child node aggregation via Möbius weighted midpoint."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall


class ChildAggregator(nn.Module):
    """Aggregates child hidden states using the Möbius weighted midpoint.

    Supports two modes:
      - attention=True: learned attention weights over children
      - attention=False: uniform weights
    """

    def __init__(self, d_hidden: int, manifold: PoincareBall, attention: bool = True):
        super().__init__()
        self.manifold = manifold
        self.attention = attention
        if attention:
            self.W_a = nn.Linear(d_hidden, 1, bias=False)

    def forward(self, h_children: Tensor, mask: Tensor | None = None) -> Tensor:
        """Aggregate child hidden states.

        Args:
            h_children: (K, batch, d_hidden) padded tensor of child states on the ball
            mask: (K, batch) boolean mask, True for valid children

        Returns:
            (batch, d_hidden) aggregated representation on the ball
        """
        K = h_children.shape[0]

        if self.attention:
            # Compute attention in tangent space
            h_tan = self.manifold.log_map_zero(h_children)  # (K, batch, d)
            scores = self.W_a(h_tan).squeeze(-1)  # (K, batch)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=0).unsqueeze(-1)  # (K, batch, 1)
        else:
            if mask is not None:
                weights = mask.float().unsqueeze(-1)  # (K, batch, 1)
                weights = weights / torch.clamp(weights.sum(dim=0, keepdim=True), min=1e-8)
            else:
                weights = torch.full((K, h_children.shape[1], 1), 1.0 / K, device=h_children.device)

        return self.manifold.mobius_midpoint(h_children, weights)
