"""The Scout: retrieve structural cousins from the Poincaré ball.

Goal: given an anchor node, find K subtrees that are **structurally similar** to the
anchor's subtree (isomorphic shape) but **semantically distant** (different domains) —
"structural cousins", not "semantic neighbours". Selection is a greedy MAP over a
Determinantal Point Process (DPP): quality = structural match to the anchor, diversity
kernel = semantic similarity, so same-domain candidates are suppressed automatically.

This is the non-scalar diversity signal made operational: the DPP log-det scores a *set*,
not a single number.

Decoupled from training: it takes a frozen ``h_all`` plus the tree (``children_indices``),
exactly like ``evaluation/structure_metrics.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from evaluation.intrinsic.structure_metrics import (
    _parents_from_children,
    _depths,
    _ancestors_inclusive,
    _level_k_ancestors,
)


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------
def subtree_nodes(root: int, children_indices: list[list[int]], max_nodes: int = 64) -> list[int]:
    """Nodes in the subtree rooted at ``root`` (BFS, capped)."""
    out: list[int] = []
    queue = [root]
    while queue and len(out) < max_nodes:
        v = queue.pop(0)
        out.append(v)
        queue.extend(children_indices[v])
    return out


def structural_fingerprint(
    root: int,
    children_indices: list[list[int]],
    depth: list[int],
    max_depth: int = 6,
    max_deg: int = 6,
) -> np.ndarray:
    """Shape-only descriptor of a subtree (a WL-style fingerprint).

    Captures *shape*, not content: relative-depth histogram + degree histogram + size.
    Two subtrees from unrelated domains can share a fingerprint — that's the point.
    """
    nodes = subtree_nodes(root, children_indices)
    d0 = depth[root]
    depth_hist = np.zeros(max_depth + 1)
    deg_hist = np.zeros(max_deg + 1)
    for v in nodes:
        rd = min(max(depth[v] - d0, 0), max_depth)
        depth_hist[rd] += 1
        deg = min(len(children_indices[v]), max_deg)
        deg_hist[deg] += 1
    size = np.array([len(nodes)], dtype=float)
    fp = np.concatenate([depth_hist, deg_hist, size])
    n = np.linalg.norm(fp)
    return fp / n if n > 0 else fp


# ---------------------------------------------------------------------------
# Distances on the ball
# ---------------------------------------------------------------------------
def _pairwise_dist(h: Tensor, manifold=None) -> Tensor:
    """(M, M) geodesic (hyperbolic) or Euclidean distance matrix."""
    M = h.shape[0]
    if manifold is not None:
        out = torch.zeros(M, M)
        for i in range(M):
            d = manifold.distance(h[i].unsqueeze(0).expand(M, -1), h)
            out[i] = d.squeeze(-1) if d.dim() > 1 else d
        return out
    return torch.cdist(h, h)


# ---------------------------------------------------------------------------
# Greedy MAP for a DPP
# ---------------------------------------------------------------------------
def _greedy_dpp(L: np.ndarray, k: int) -> list[int]:
    """Greedy maximisation of log det L_S over |S| = k (Chen et al. 2018, simple form)."""
    n = L.shape[0]
    k = min(k, n)
    selected: list[int] = []
    remaining = set(range(n))
    for _ in range(k):
        best_i, best_gain = None, -np.inf
        for i in remaining:
            idx = selected + [i]
            sub = L[np.ix_(idx, idx)]
            sign, logdet = np.linalg.slogdet(sub)
            gain = logdet if sign > 0 else -np.inf
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.discard(best_i)
    return selected


# ---------------------------------------------------------------------------
# Scout
# ---------------------------------------------------------------------------
def scout(
    anchor: int,
    h_all: Tensor,
    children_indices: list[list[int]],
    k: int = 3,
    *,
    manifold=None,
    candidate_pool: int = 100,
    sigma: float = 1.0,
    same_domain_sim: float = 0.95,
    seed: int = 0,
) -> list[int]:
    """Return K structural-cousin node ids for ``anchor``.

    1. Fingerprint every candidate subtree; rank by structural match to the anchor.
    2. Keep the top ``candidate_pool`` (excluding the anchor's own ancestors/descendants).
    3. DPP-select K: quality = structural match, similarity = semantic closeness
       (so the chosen set is structurally on-target but domain-diverse).
    """
    N = len(children_indices)
    parents = _parents_from_children(children_indices)
    depth = _depths(children_indices, parents)
    lvl1 = _level_k_ancestors(children_indices, parents, depth, 1)

    anchor_fp = structural_fingerprint(anchor, children_indices, depth)
    excluded = _ancestors_inclusive(anchor, parents)
    # also exclude the anchor's descendants — cousins live elsewhere in the tree
    excluded |= set(subtree_nodes(anchor, children_indices))

    # 1-2. structural match over candidates
    cand, match = [], []
    for v in range(N):
        if v in excluded or not children_indices[v]:
            continue
        fp = structural_fingerprint(v, children_indices, depth)
        cand.append(v)
        match.append(float(np.dot(anchor_fp, fp)))
    if not cand:
        return []
    match = np.array(match)
    order = np.argsort(-match)[:candidate_pool]
    cand = [cand[i] for i in order]
    q = match[order]
    q = np.clip(q, 1e-3, None)

    # semantic similarity kernel (Gaussian on geodesic distance; same-domain pinned high)
    h = h_all[torch.tensor(cand, dtype=torch.long)]
    dmat = _pairwise_dist(h, manifold).cpu().numpy()
    S = np.exp(-(dmat ** 2) / (2 * sigma ** 2))
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            if lvl1[cand[a]] & lvl1[cand[b]]:           # shared domain → discourage co-selection
                S[a, b] = S[b, a] = max(S[a, b], same_domain_sim)
    np.fill_diagonal(S, 1.0)

    L = np.outer(q, q) * S
    chosen = _greedy_dpp(L, k)
    return [cand[i] for i in chosen]
