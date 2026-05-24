"""Numerical stability utilities for hyperbolic operations."""

import torch
from torch import Tensor

from pluraltree.manifolds.math_utils import MIN_NORM


CONFORMAL_THRESHOLD = 100.0


def should_use_lorentz(x: Tensor, c: float) -> Tensor:
    """Detect points where Poincaré ball operations are numerically unstable.

    Returns a boolean mask where True means the point should use the
    Lorentz (hyperboloid) model instead.
    """
    x_sqnorm = torch.sum(x * x, dim=-1)
    conformal = 2.0 / torch.clamp(1.0 - c * x_sqnorm, min=MIN_NORM)
    return conformal > CONFORMAL_THRESHOLD


def ensure_float32_manifold(fn):
    """Decorator that forces float32 precision for manifold operations.

    Prevents precision loss when using AMP (automatic mixed precision).
    """

    def wrapper(*args, **kwargs):
        # Cast all tensor args to float32
        new_args = []
        for a in args:
            if isinstance(a, Tensor) and a.dtype == torch.float16:
                new_args.append(a.float())
            else:
                new_args.append(a)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, Tensor) and v.dtype == torch.float16:
                new_kwargs[k] = v.float()
            else:
                new_kwargs[k] = v
        return fn(*new_args, **new_kwargs)

    return wrapper
