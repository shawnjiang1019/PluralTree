"""Loss functions for knowledge graph tasks."""

import torch
import torch.nn.functional as F
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall


def link_prediction_loss(
    h_subjects: Tensor,
    h_objects_pos: Tensor,
    h_objects_neg: Tensor,
    manifold: PoincareBall,
    margin: float = 1.0,
) -> Tensor:
    """Margin-based link prediction loss using hyperbolic distance.

    L = mean(max(0, margin + d(s, o+) - d(s, o-)))
    """
    d_pos = manifold.distance(h_subjects, h_objects_pos).squeeze(-1)
    d_neg = manifold.distance(h_subjects, h_objects_neg).squeeze(-1)
    return F.relu(margin + d_pos - d_neg).mean()


def gate_sparsity_loss(gate_activations: Tensor) -> Tensor:
    """Regularizer encouraging selective (sparse) gate usage.

    L = mean(g) — penalizes gates that are always open.
    """
    return gate_activations.mean()


def curvature_regularization(manifold: PoincareBall, target_c: float = 1.0) -> Tensor:
    """Optional regularization to keep learnable curvature near a target."""
    return (manifold.c - target_c) ** 2
