"""Depth-aware hyperbolic gate conditioned on Poincaré ball radius.

The key insight: in the Poincaré ball, ||x|| encodes hierarchical depth.
Nodes near the origin are abstract/root-like; nodes near the boundary are
leaves/specific. This gate uses the radius ρ to:
  1. Blend between broad (relational) and precise (factual) knowledge sources
  2. Condition the injection magnitude on hierarchical position
"""

import torch
import torch.nn as nn
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.manifolds.math_utils import safe_norm


class DepthAwareHyperbolicGate(nn.Module):
    """Gate that conditions on hyperbolic radius for depth-aware knowledge injection."""

    def __init__(
        self,
        d_hidden: int,
        d_knowledge: int,
        manifold: PoincareBall,
        bias_init: float = -1.0,
    ):
        super().__init__()
        self.manifold = manifold
        # +1 for the radius feature ρ
        self.W_g = nn.Linear(d_hidden + d_knowledge + 1, d_hidden)
        nn.init.constant_(self.W_g.bias, bias_init)

        # Depth-aware knowledge source blending parameters
        # Positive w_beta init: leaves (high ρ) get precise knowledge
        self.w_beta = nn.Parameter(torch.tensor(2.0))
        self.b_beta = nn.Parameter(torch.tensor(-1.0))

    def compute_rho(self, h: Tensor) -> Tensor:
        """Compute normalized radius ρ = √c · ||h|| ∈ [0, 1).

        This is the depth signal: small ρ = abstract, large ρ = specific.
        """
        c = self.manifold.c
        return c.sqrt() * safe_norm(h, dim=-1, keepdim=True)

    def blend_knowledge(self, k_broad: Tensor, k_precise: Tensor, rho: Tensor) -> Tensor:
        """Blend broad and precise knowledge based on depth.

        Abstract nodes (small ρ) get more broad relational knowledge.
        Leaf nodes (large ρ) get more precise factual knowledge.

        Args:
            k_broad: (batch, d) broad relational knowledge on the ball
            k_precise: (batch, d) precise factual knowledge on the ball
            rho: (batch, 1) normalized radius

        Returns:
            (batch, d) blended knowledge on the ball
        """
        beta = torch.sigmoid(self.w_beta * rho + self.b_beta)  # (batch, 1)
        k_broad_tan = self.manifold.log_map_zero(k_broad)
        k_precise_tan = self.manifold.log_map_zero(k_precise)
        k_blended_tan = (1.0 - beta) * k_broad_tan + beta * k_precise_tan
        return self.manifold.exp_map_zero(k_blended_tan)

    def forward(self, h: Tensor, k: Tensor) -> Tensor:
        """Compute depth-aware injection gate.

        Args:
            h: (batch, d_hidden) point on the Poincaré ball
            k: (batch, d_knowledge) knowledge point on the ball

        Returns:
            (batch, d_hidden) gate activations in (0, 1)
        """
        rho = self.compute_rho(h)
        h_tan = self.manifold.log_map_zero(h)
        k_tan = self.manifold.log_map_zero(k)
        gate_input = torch.cat([h_tan, k_tan, rho], dim=-1)
        return torch.sigmoid(self.W_g(gate_input))
