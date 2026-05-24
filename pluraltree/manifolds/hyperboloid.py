"""Hyperboloid (Lorentz) model — fallback for numerical stability.

Used when the conformal factor in the Poincaré ball exceeds a threshold,
indicating the point is too close to the boundary for stable computation.
"""

import torch
from torch import Tensor

from .math_utils import safe_norm, MIN_NORM


def lorentz_inner(u: Tensor, v: Tensor) -> Tensor:
    """Minkowski inner product <u, v>_L = -u_0*v_0 + u_1*v_1 + ... + u_d*v_d."""
    return -u[..., :1] * v[..., :1] + torch.sum(u[..., 1:] * v[..., 1:], dim=-1, keepdim=True)


def lorentz_norm(v: Tensor) -> Tensor:
    """Lorentzian norm sqrt(|<v,v>_L|)."""
    inner = lorentz_inner(v, v)
    return torch.sqrt(torch.clamp(torch.abs(inner), min=MIN_NORM))


def exp_map_lorentz(x: Tensor, v: Tensor, c: float = 1.0) -> Tensor:
    """Exponential map on the hyperboloid at point x."""
    k = 1.0 / c
    v_norm = torch.sqrt(torch.clamp(lorentz_inner(v, v), min=MIN_NORM))
    coeff = torch.sinh(v_norm / (k ** 0.5)) / v_norm
    return torch.cosh(v_norm / (k ** 0.5)) * x + coeff * v


def log_map_lorentz(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
    """Logarithmic map on the hyperboloid at point x."""
    k = 1.0 / c
    inner = torch.clamp(lorentz_inner(x, y), max=-1.0 / k)
    theta = torch.acosh(torch.clamp(-c * inner, min=1.0 + MIN_NORM))
    v = y - (-c * inner) * x
    v_norm = torch.sqrt(torch.clamp(lorentz_inner(v, v), min=MIN_NORM))
    return (theta / v_norm) * v


def lorentz_midpoint(points: Tensor, weights: Tensor, c: float = 1.0) -> Tensor:
    """Weighted midpoint in the Lorentz model (linear in ambient coordinates).

    This is numerically more stable than the Poincaré ball midpoint
    for points near the boundary.

    Args:
        points: (K, ..., d+1) tensor of K points on the hyperboloid
        weights: (K, ..., 1) tensor of non-negative weights

    Returns:
        Midpoint on the hyperboloid, shape (..., d+1)
    """
    weighted_sum = torch.sum(weights * points, dim=0)
    # Project back onto hyperboloid: normalize by Lorentz norm
    sq_inner = lorentz_inner(weighted_sum, weighted_sum)
    scale = 1.0 / torch.sqrt(torch.clamp(-c * sq_inner, min=MIN_NORM))
    return weighted_sum * scale
