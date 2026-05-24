"""Projection layer to map knowledge source outputs to a unified hidden dimension."""

import torch.nn as nn
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall


class ProjectionLayer(nn.Module):
    """Projects a knowledge vector from source dimension to hidden dimension.

    In Euclidean mode: Linear + LayerNorm.
    In hyperbolic mode: Linear + LayerNorm + exp_map_zero onto the manifold.
    """

    def __init__(self, d_source: int, d_target: int, manifold: PoincareBall | None = None):
        super().__init__()
        self.linear = nn.Linear(d_source, d_target)
        self.norm = nn.LayerNorm(d_target)
        self.manifold = manifold

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(self.linear(x))
        if self.manifold is not None:
            h = self.manifold.exp_map_zero(h)
        return h
