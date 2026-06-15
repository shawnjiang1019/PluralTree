"""Structure-fidelity metrics — does the embedding's *geometry* encode the tree?

Link prediction only checks the relational translation ``h_s ⊕ r ≈ h_o``; it does
not measure whether distances and neighbourhoods are faithful to the hierarchy,
which is what subtree- and path-similarity retrieval depend on (see
``docs/EVALUATION.md``). These metrics fill that gap. They are *intrinsic*: they
take a frozen ``h_all`` plus the tree (``children_indices`` / ``topo_order``) and
run with no training loop.

All metrics are computed by *sampling* (the graph may be ~40K–130K nodes, so
all-pairs is O(N²)); pass a fixed ``seed`` for reproducibility. Pass ``manifold``
to use hyperbolic distance/radius; pass ``manifold=None`` to use the Euclidean
versions — so the *same* function evaluates the trained model and the frozen-text
floor.

Returned keys:
    depth_radius_rho   Spearman(‖h‖, tree-depth). The depth-aware design's core
                       premise: radius should grow with depth. (higher = better)
    dist_tree_rho      Spearman(d(h_u,h_v), tree-distance). Global metric fidelity.
    recon_map          mean AP of ranking each node's tree neighbours by distance.
    subtree_ap         mean AP of retrieving same-level-k-subtree nodes. The direct
                       "subtree similarity" score. (higher = better)
    ancestor_auc       AUC separating ancestor pairs from random pairs by closeness.
                       Path/transitive structure. (higher = better)
    sibling_ratio      mean intra-sibling distance ÷ mean random-pair distance.
                       Over-smoothing guard: lower = tighter siblings, but → 0 means
                       siblings have collapsed (lateral pass smoothed them away).
"""

from __future__ import annotations

from collections import defaultdict, deque

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Small statistics helpers (no scipy / sklearn dependency)
# ---------------------------------------------------------------------------
def _ranks(x: Tensor) -> Tensor:
    """Ordinal ranks (ties broken arbitrarily) — adequate for Spearman."""
    return torch.argsort(torch.argsort(x)).float()


def _pearson(a: Tensor, b: Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


def _spearman(x: Tensor, y: Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    return _pearson(_ranks(x), _ranks(y))


def _average_precision(scores: Tensor, labels: Tensor) -> float:
    """AP for a single query: scores (higher = more relevant), binary labels."""
    if labels.sum() == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    lab = labels[order].float()
    cum_tp = torch.cumsum(lab, dim=0)
    ranks = torch.arange(1, lab.numel() + 1, device=lab.device, dtype=torch.float)
    precision_at_k = cum_tp / ranks
    return float((precision_at_k * lab).sum() / lab.sum())


def _auc(scores: Tensor, labels: Tensor) -> float:
    """Mann–Whitney AUC: P(score(pos) > score(neg))."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _ranks(scores) + 1.0
    r_pos = float(ranks[labels == 1].sum())
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Tree structure derived from children_indices
# ---------------------------------------------------------------------------
def _parents_from_children(children_indices: list[list[int]]) -> list[list[int]]:
    parents: list[list[int]] = [[] for _ in children_indices]
    for p, kids in enumerate(children_indices):
        for c in kids:
            parents[c].append(p)
    return parents


def _depths(children_indices: list[list[int]], parents: list[list[int]]) -> list[int]:
    """BFS depth from every root (node with no parents). depth(root)=0."""
    N = len(children_indices)
    depth = [-1] * N
    q: deque[int] = deque()
    for v in range(N):
        if not parents[v]:
            depth[v] = 0
            q.append(v)
    while q:
        v = q.popleft()
        for c in children_indices[v]:
            if depth[c] < depth[v] + 1:        # longest path from a root
                depth[c] = depth[v] + 1
                q.append(c)
    return depth


def _ancestors_inclusive(v: int, parents: list[list[int]]) -> set[int]:
    """All ancestors of v plus v itself (handles multi-parent DAGs)."""
    seen = {v}
    stack = [v]
    while stack:
        u = stack.pop()
        for p in parents[u]:
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def _level_k_ancestors(
    children_indices: list[list[int]],
    parents: list[list[int]],
    depth: list[int],
    k: int,
) -> list[set[int]]:
    """For each node, the set of its ancestors at depth == k (its level-k subtrees).

    Computed bottom-up over increasing depth so parents are done before children.
    Nodes shallower than k get an empty set (no level-k ancestor).
    """
    N = len(children_indices)
    lk: list[set[int]] = [set() for _ in range(N)]
    for v in sorted(range(N), key=lambda n: depth[n]):
        if depth[v] == k:
            lk[v] = {v}
        elif depth[v] > k:
            s: set[int] = set()
            for p in parents[v]:
                s |= lk[p]
            lk[v] = s
    return lk


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def compute_structure_metrics(
    h_all: Tensor,
    children_indices: list[list[int]],
    topo_order: list[int],
    *,
    manifold=None,
    subtree_level: int = 1,
    n_anchors: int = 256,
    n_pairs: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute the structure-fidelity vector for a frozen embedding ``h_all``.

    Args:
        h_all:            (N, d) entity embeddings (on the ball if hyperbolic).
        children_indices: tree adjacency (children_indices[i] = child ids of i).
        topo_order:       leaves-first order (accepted for interface symmetry).
        manifold:         PoincareBall for hyperbolic distance/radius; None = Euclidean.
        subtree_level:    depth k defining "same subtree" for subtree_ap.
        n_anchors:        number of sampled anchor nodes for AP-style metrics.
        n_pairs:          number of sampled node pairs for correlation/AUC metrics.
        seed:             RNG seed for the sampling.

    Returns:
        dict of the metric names documented in the module docstring.
    """
    device = h_all.device
    N = h_all.shape[0]
    gen = torch.Generator().manual_seed(seed)

    # Distance / radius — hyperbolic or Euclidean depending on `manifold`.
    # manifold.distance keeps a trailing singleton dim; squeeze it so distances
    # are shape (...,) and rank/AP math behaves.
    if manifold is not None:
        def dist(a: Tensor, b: Tensor) -> Tensor:
            d = manifold.distance(a, b)
            return d.squeeze(-1) if d.shape and d.shape[-1] == 1 else d
    else:
        def dist(a: Tensor, b: Tensor) -> Tensor:
            return (a - b).norm(dim=-1)

    radius = h_all.norm(dim=-1)   # monotonic in hyperbolic radius, so Spearman-safe

    parents = _parents_from_children(children_indices)
    depth = _depths(children_indices, parents)
    depth_t = torch.tensor(depth, dtype=torch.float)
    roots = {v for v in range(N) if not parents[v]}

    def rand_nodes(k: int, exclude_roots: bool = False) -> list[int]:
        pool = [v for v in range(N) if not (exclude_roots and v in roots)]
        if k >= len(pool):
            return pool
        idx = torch.randperm(len(pool), generator=gen)[:k].tolist()
        return [pool[i] for i in idx]

    def rand_ids(k: int) -> list[int]:
        """k node ids drawn uniformly *with replacement* (for pair sampling)."""
        return torch.randint(N, (k,), generator=gen).tolist()

    out: dict[str, float] = {}

    # -- 1. depth–radius Spearman (exclude roots: their radius ~0 is trivial) --
    nonroot = torch.tensor([v for v in range(N) if v not in roots], dtype=torch.long)
    out["depth_radius_rho"] = _spearman(radius[nonroot].cpu(), depth_t[nonroot])

    # -- 2. distance vs tree-distance Spearman (sampled pairs) --
    # tree-distance via LCA: d_tree(u,v) = depth_u + depth_v - 2*depth_lca.
    us = rand_ids(n_pairs)
    vs = rand_ids(n_pairs)
    emb_d, tree_d = [], []
    for u, v in zip(us, vs):
        if u == v:
            continue
        au = _ancestors_inclusive(u, parents)
        av = _ancestors_inclusive(v, parents)
        common = au & av
        if not common:
            continue
        lca_depth = max(depth[c] for c in common)
        tree_d.append(depth[u] + depth[v] - 2 * lca_depth)
        emb_d.append(float(dist(h_all[u], h_all[v])))
    if len(tree_d) >= 2:
        out["dist_tree_rho"] = _spearman(
            torch.tensor(emb_d), torch.tensor(tree_d, dtype=torch.float)
        )
    else:
        out["dist_tree_rho"] = float("nan")

    # -- 3. reconstruction mAP: rank each node's tree neighbours by distance --
    neigh_anchors = [v for v in rand_nodes(n_anchors)
                     if children_indices[v] or parents[v]]
    aps = []
    for v in neigh_anchors:
        d = dist(h_all[v].unsqueeze(0), h_all)          # (N,)
        d[v] = float("inf")                             # exclude self
        labels = torch.zeros(N, device=device)
        for nb in children_indices[v] + parents[v]:
            labels[nb] = 1.0
        aps.append(_average_precision(-d, labels))
    out["recon_map"] = float(torch.tensor(aps).nanmean()) if aps else float("nan")

    # -- 4. same-subtree retrieval AP at level k --
    lk = _level_k_ancestors(children_indices, parents, depth, subtree_level)
    group_members: dict[int, list[int]] = defaultdict(list)
    for v in range(N):
        for a in lk[v]:
            group_members[a].append(v)
    sub_anchors = [v for v in rand_nodes(n_anchors) if lk[v]]
    aps = []
    for v in sub_anchors:
        relevant = set()
        for a in lk[v]:
            relevant.update(group_members[a])
        relevant.discard(v)
        if not relevant:
            continue
        d = dist(h_all[v].unsqueeze(0), h_all)
        d[v] = float("inf")
        labels = torch.zeros(N, device=device)
        labels[torch.tensor(sorted(relevant), dtype=torch.long)] = 1.0
        aps.append(_average_precision(-d, labels))
    out["subtree_ap"] = float(torch.tensor(aps).nanmean()) if aps else float("nan")

    # -- 5. ancestor AUC: are ancestor pairs closer than random pairs? --
    pos_d, neg_d = [], []
    tries = 0
    target = n_pairs // 2
    cand = rand_nodes(min(N, max(2 * target, 1024)))
    for v in cand:
        anc = _ancestors_inclusive(v, parents) - {v}
        if anc and len(pos_d) < target:
            a = list(anc)[torch.randint(len(anc), (1,), generator=gen).item()]
            pos_d.append(float(dist(h_all[v], h_all[a])))
        if len(neg_d) < target:
            w = int(torch.randint(N, (1,), generator=gen).item())
            if w != v and w not in _ancestors_inclusive(v, parents) \
                    and v not in _ancestors_inclusive(w, parents):
                neg_d.append(float(dist(h_all[v], h_all[w])))
        tries += 1
        if len(pos_d) >= target and len(neg_d) >= target:
            break
    if pos_d and neg_d:
        scores = torch.tensor([-d for d in pos_d] + [-d for d in neg_d])  # closer = higher
        labels = torch.tensor([1] * len(pos_d) + [0] * len(neg_d))
        out["ancestor_auc"] = _auc(scores, labels)
    else:
        out["ancestor_auc"] = float("nan")

    # -- 6. sibling cohesion vs separation (over-smoothing guard) --
    multi = [p for p in range(N) if len(children_indices[p]) >= 2]
    intra = []
    if multi:
        sel = multi if len(multi) <= n_anchors else \
            [multi[i] for i in torch.randperm(len(multi), generator=gen)[:n_anchors].tolist()]
        for p in sel:
            kids = children_indices[p]
            i, j = torch.randint(len(kids), (2,), generator=gen).tolist()
            if i != j:
                intra.append(float(dist(h_all[kids[i]], h_all[kids[j]])))
    rand_d = []
    ua = rand_ids(n_pairs)
    ub = rand_ids(n_pairs)
    for a, b in zip(ua, ub):
        if a != b:
            rand_d.append(float(dist(h_all[a], h_all[b])))
    if intra and rand_d:
        inter_mean = max(sum(rand_d) / len(rand_d), 1e-9)
        out["sibling_ratio"] = (sum(intra) / len(intra)) / inter_mean
    else:
        out["sibling_ratio"] = float("nan")

    return out
