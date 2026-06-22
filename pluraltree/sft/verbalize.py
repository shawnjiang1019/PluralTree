"""Verbalize a subtree into an interpretable caption for the distiller / LM.

Decodes the hyperbolic geometry into human-readable structural facts: radius (depth),
domain (level-1 ancestor), and nearby structure. This is the interpretable channel —
the same readouts a probe would reconstruct from an injected latent.
"""

from __future__ import annotations

import torch
from torch import Tensor

from evaluation.structure_metrics import _parents_from_children, _depths, _level_k_ancestors


def _name(graph, node_id: int) -> str:
    return graph.id_to_entity[node_id]


def verbalize_subtree(
    node_id: int,
    h_all: Tensor,
    graph,
    *,
    manifold=None,
    max_children: int = 5,
) -> tuple[str, str]:
    """Return ``(domain_label, caption)`` for the subtree rooted at ``node_id``.

    ``domain_label`` is the level-1 ancestor name (or the node's own name if shallow) —
    used as the path's ``source`` tag. ``caption`` is the natural-language description.
    """
    children_indices = graph.children_indices
    parents = _parents_from_children(children_indices)
    depth = _depths(children_indices, parents)
    lvl1 = _level_k_ancestors(children_indices, parents, depth, 1)

    name = _name(graph, node_id)

    # radius → abstraction level
    h = h_all[node_id]
    if manifold is not None:
        rho = float(manifold.c.sqrt() * h.norm())
    else:
        rho = float(h.norm())
    abstraction = "specific" if rho > 0.66 else "broad" if rho < 0.33 else "intermediate"

    # domain = level-1 ancestor
    dom_ids = sorted(lvl1[node_id])
    domain = _name(graph, dom_ids[0]) if dom_ids else name

    # local structure
    kids = [_name(graph, c) for c in children_indices[node_id][:max_children]]
    par = [_name(graph, p) for p in parents[node_id][:max_children]]

    parts = [f"'{name}' (domain: {domain}; {abstraction}, depth {depth[node_id]}, rho={rho:.2f})"]
    if par:
        parts.append("parents: " + ", ".join(par))
    if kids:
        parts.append("children: " + ", ".join(kids))
    caption = "; ".join(parts) + "."
    return domain, caption
