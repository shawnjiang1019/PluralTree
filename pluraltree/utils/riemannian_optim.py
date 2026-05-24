"""Riemannian optimizer wrappers using geoopt."""

import torch.nn as nn

try:
    import geoopt

    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    lr_manifold: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float = 5.0,
):
    """Build a Riemannian optimizer with separate learning rates for manifold and Euclidean params.

    Manifold parameters (geoopt.ManifoldParameter) use Riemannian Adam.
    Euclidean parameters use standard Adam.

    Args:
        model: the model
        lr: learning rate for Euclidean parameters
        lr_manifold: learning rate for manifold parameters (typically higher)
        weight_decay: L2 regularization for Euclidean parameters
        max_grad_norm: gradient clipping norm

    Returns:
        (optimizer, max_grad_norm) tuple
    """
    if not HAS_GEOOPT:
        raise ImportError("geoopt is required for Riemannian optimization. Install with: pip install geoopt")

    euclidean_params = []
    manifold_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if isinstance(param, geoopt.ManifoldParameter):
            manifold_params.append(param)
        else:
            euclidean_params.append(param)

    param_groups = []
    if euclidean_params:
        param_groups.append({"params": euclidean_params, "lr": lr, "weight_decay": weight_decay})
    if manifold_params:
        param_groups.append({"params": manifold_params, "lr": lr_manifold})

    optimizer = geoopt.optim.RiemannianAdam(param_groups)
    return optimizer, max_grad_norm
