"""Poincaré ball model of hyperbolic geometry.

All operations use curvature c > 0, corresponding to a space of constant
negative curvature -c. The ball has radius 1/sqrt(c).
"""

import torch
import torch.nn as nn
from torch import Tensor

from .math_utils import (
    safe_artanh,
    safe_norm,
    project_to_ball,
    lambda_x,
    MIN_NORM,
    BOUNDARY_EPS,
)


class PoincareBall(nn.Module):
    """Poincaré ball manifold with Möbius operations."""

    def __init__(self, c: float = 1.0, learnable_c: bool = False):
        super().__init__()
        if learnable_c:
            self.c_param = nn.Parameter(torch.tensor(c))
        else:
            self.register_buffer("c_param", torch.tensor(c))

    @property
    def c(self) -> Tensor:
        return torch.abs(self.c_param) + MIN_NORM

    def project(self, x: Tensor) -> Tensor:
        """Project onto the ball interior."""
        return project_to_ball(x, self.c.item())

    def lambda_x(self, x: Tensor) -> Tensor:
        """Conformal factor at x."""
        return lambda_x(x, self.c)

    # ---- exponential / logarithmic maps at the origin ----

    def exp_map_zero(self, v: Tensor) -> Tensor:
        """Exponential map from the origin: T_0 B -> B."""
        c = self.c
        v_norm = safe_norm(v)
        coeff = torch.tanh(c.sqrt() * v_norm) / (c.sqrt() * v_norm)
        return self.project(coeff * v)

    def log_map_zero(self, x: Tensor) -> Tensor:
        """Logarithmic map to the origin: B -> T_0 B."""
        c = self.c
        x_norm = safe_norm(x)
        coeff = safe_artanh(c.sqrt() * x_norm) / (c.sqrt() * x_norm)
        return coeff * x

    # ---- general exponential / logarithmic maps ----

    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map at point x: T_x B -> B."""
        c = self.c
        v_norm = safe_norm(v)
        lx = lambda_x(x, c)
        coeff = torch.tanh(lx * c.sqrt() * v_norm / 2.0) / (c.sqrt() * v_norm)
        y = self.mobius_add(x, coeff * v)
        return self.project(y)

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map at point x: B -> T_x B."""
        c = self.c
        neg_x = self.mobius_add(-x, y)
        neg_x_norm = safe_norm(neg_x)
        lx = lambda_x(x, c)
        coeff = (2.0 / (lx * c.sqrt())) * safe_artanh(c.sqrt() * neg_x_norm)
        return coeff * (neg_x / neg_x_norm)

    # ---- Möbius operations ----

    def mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
        """Möbius addition x ⊕_c y."""
        c = self.c
        x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
        return self.project(num / torch.clamp(denom, min=MIN_NORM))

    def mobius_scalar_mul(self, r: Tensor, x: Tensor) -> Tensor:
        """Möbius scalar multiplication r ⊗_c x."""
        c = self.c
        x_norm = safe_norm(x)
        coeff = torch.tanh(r * safe_artanh(c.sqrt() * x_norm)) / (c.sqrt() * x_norm)
        return self.project(coeff * x)

    def mobius_matvec(self, M: Tensor, x: Tensor) -> Tensor:
        """Möbius matrix-vector multiplication M ⊗_c x.

        Maps x to tangent space, applies M, maps back.
        """
        v = self.log_map_zero(x)
        Mv = v @ M.transpose(-1, -2)
        return self.exp_map_zero(Mv)

    # ---- distance ----

    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance d_c(x, y)."""
        c = self.c
        diff = self.mobius_add(-x, y)
        diff_norm = safe_norm(diff)
        return (2.0 / c.sqrt()) * safe_artanh(c.sqrt() * diff_norm)

    # ---- aggregation ----

    def mobius_midpoint(self, points: Tensor, weights: Tensor) -> Tensor:
        """Weighted Möbius midpoint (Einstein midpoint in the Poincaré ball).

        Args:
            points: (K, ..., d) tensor of K points on the ball
            weights: (K, ..., 1) or (K, ..., d) tensor of non-negative weights

        Returns:
            Weighted midpoint on the ball, shape (..., d)
        """
        c = self.c
        # Conformal factors for each point
        lx = lambda_x(points, c)  # (K, ..., 1)
        # Numerator: sum of w_k * lambda_k * x_k
        numerator = torch.sum(weights * lx * points, dim=0)
        # Denominator: sum of w_k * (lambda_k - 1)
        denominator = torch.sum(weights * (lx - 1.0), dim=0)
        denominator = torch.clamp(torch.abs(denominator), min=MIN_NORM) * torch.sign(
            denominator + MIN_NORM
        )
        gamma = numerator / denominator
        # Project result back onto ball
        return self.project(gamma)

    # ---- conversions ----

    def to_hyperboloid(self, x: Tensor) -> Tensor:
        """Convert Poincaré ball point to hyperboloid (Lorentz) model.

        Returns (d+1)-dim vector where first component is the time coordinate.
        """
        c = self.c
        x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        denom = torch.clamp(1.0 - c * x_sq, min=MIN_NORM)
        time = (1.0 + c * x_sq) / denom
        space = (2.0 * c.sqrt() * x) / denom
        return torch.cat([time, space], dim=-1)

    def from_hyperboloid(self, h: Tensor) -> Tensor:
        """Convert hyperboloid point back to Poincaré ball."""
        c = self.c
        time = h[..., :1]
        space = h[..., 1:]
        x = space / (c.sqrt() * (time + 1.0))
        return self.project(x)
