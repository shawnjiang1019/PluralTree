"""Gating mechanisms for knowledge injection — Euclidean and hyperbolic variants."""

import torch
import torch.nn as nn
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall


class EuclideanGate(nn.Module):
    """Standard sigmoid gate over concatenated hidden and knowledge states.

    g = σ(W_g [h ; k] + b_g)
    """

    def __init__(self, d_hidden: int, d_knowledge: int, bias_init: float = -1.0):
        super().__init__()
        self.W_g = nn.Linear(d_hidden + d_knowledge, d_hidden)
        nn.init.constant_(self.W_g.bias, bias_init)

    def forward(self, h: Tensor, k: Tensor) -> Tensor:
        """Compute gate values.

        Args:
            h: (batch, d_hidden) model hidden state
            k: (batch, d_knowledge) knowledge vector

        Returns:
            (batch, d_hidden) gate activations in (0, 1)
        """
        return torch.sigmoid(self.W_g(torch.cat([h, k], dim=-1)))


class HyperbolicGate(nn.Module):
    """Gate computed in the tangent space at the origin of the Poincaré ball.

    Projects h and k to tangent space via log_map_zero, concatenates,
    then applies a linear + sigmoid gate. This avoids needing to
    define sigmoid on the manifold.
    """

    def __init__(self, d_hidden: int, d_knowledge: int, manifold: PoincareBall, bias_init: float = -1.0):
        super().__init__()
        self.manifold = manifold
        self.W_g = nn.Linear(d_hidden + d_knowledge, d_hidden)
        nn.init.constant_(self.W_g.bias, bias_init)

    def forward(self, h: Tensor, k: Tensor) -> Tensor:
        """Compute gate values from points on the manifold.

        Args:
            h: (batch, d_hidden) point on the Poincaré ball
            k: (batch, d_knowledge) point on the Poincaré ball

        Returns:
            (batch, d_hidden) gate activations in (0, 1)
        """
        h_tan = self.manifold.log_map_zero(h)
        k_tan = self.manifold.log_map_zero(k)
        return torch.sigmoid(self.W_g(torch.cat([h_tan, k_tan], dim=-1)))
