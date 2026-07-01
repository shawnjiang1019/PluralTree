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


def geodesic_separation_loss(
    h_a: Tensor, h_b: Tensor, manifold: PoincareBall, margin: float = 1.0
) -> Tensor:
    """Hinge *floor* on geodesic distance between paired points.

    L = mean(relu(margin - d(h_a, h_b))). Penalizes pairs only when they are
    *closer than* ``margin`` — it never forces them apart beyond that, so it
    fixes collapse without fabricating divergence. Shared primitive: feed sibling
    pairs (anti-collapse / faithful diversity) or codebook neighbours (robust
    deterministic decode).
    """
    d = manifold.distance(h_a, h_b).squeeze(-1)
    return F.relu(margin - d).mean()


def boundary_penalty(h_all: Tensor, manifold: PoincareBall, rho_max: float = 0.9) -> Tensor:
    """Keep mass off the rim: penalize normalized radius sqrt(c)*‖h‖ above rho_max.

    Boundary saturation makes the geometry degenerate (the monoculture artifact);
    this is a real gradient replacing the manual CURV/LSTR knob-twiddling.
    """
    rho = manifold.c.sqrt() * h_all.norm(dim=-1)
    return F.relu(rho - rho_max).pow(2).mean()


def gate_sparsity_loss(gate_activations: Tensor) -> Tensor:
    """Regularizer encouraging selective (sparse) gate usage.

    L = mean(g) — penalizes gates that are always open.
    """
    return gate_activations.mean()


def curvature_regularization(manifold: PoincareBall, target_c: float = 1.0) -> Tensor:
    """Optional regularization to keep learnable curvature near a target."""
    return (manifold.c - target_c) ** 2
