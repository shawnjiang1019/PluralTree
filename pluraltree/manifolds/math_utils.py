"""Numerically stable primitives for hyperbolic operations."""

import torch
from torch import Tensor

MIN_NORM = 1e-15
EPS = 1e-7
BOUNDARY_EPS = 1e-5


def safe_artanh(x: Tensor) -> Tensor:
    """Numerically stable arctanh: clamp input to (-1+eps, 1-eps)."""
    return torch.clamp(x, -1 + EPS, 1 - EPS).arctanh()


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = True) -> Tensor:
    """Norm that avoids zero-gradient at zero by clamping."""
    return torch.clamp(x.norm(dim=dim, keepdim=keepdim), min=MIN_NORM)


def clamp_norm(x: Tensor, max_norm: float, dim: int = -1) -> Tensor:
    """Clamp the norm of x to max_norm while preserving direction."""
    norms = safe_norm(x, dim=dim, keepdim=True)
    scale = torch.clamp(max_norm / norms, max=1.0)
    return x * scale


def project_to_ball(x: Tensor, c: float, eps: float = BOUNDARY_EPS) -> Tensor:
    """Project points onto the Poincaré ball, keeping them within the boundary."""
    max_norm = (1.0 / (c ** 0.5)) - eps
    return clamp_norm(x, max_norm, dim=-1)


def lambda_x(x: Tensor, c: float) -> Tensor:
    """Conformal factor λ_c(x) = 2 / (1 - c * ||x||^2)."""
    x_sqnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), min=0.0)
    return 2.0 / torch.clamp(1.0 - c * x_sqnorm, min=MIN_NORM)
