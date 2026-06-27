"""Branch-level divergence via Wasserstein distance between child subtrees.

Quantifies, at each parent node, how much its child branches diverge — the
"fork in the road" for values/culture. A parent whose children lead to
semantically distant subtrees is a **Divergence Anchor** (genuinely plural); a
parent whose children collapse to the same region is **monoculture** (many
branches, no real diversity). See pluraltree/sft/wasserstein.md.

PluralTree-native choice: the optimal-transport **ground metric is the Poincaré
geodesic** (``manifold.distance``), i.e. the learned semantic distance — not raw
DAG path length. This reuses the trained geometry and needs no shortest-path
computation. Each child subtree is the empirical distribution over its nodes'
embeddings (uniform mass by default).

Intrinsic + read-only: takes a frozen ``h_all`` + the tree, exactly like
``structure_metrics.py``. Wasserstein is computed with POT (``ot.emd2``, exact)
if installed, else a dependency-free log-domain Sinkhorn (entropic, approximate).

Usage (standalone ranking):
    python -m evaluation.branch_divergence --embeddings runs/h_all.pt \
        --dataset wn18rr --data_dir data/wn18rr --top 20
"""

from __future__ import annotations

import torch
from torch import Tensor

from evaluation.structure_metrics import _parents_from_children, _depths


# ---------------------------------------------------------------------------
# Subtree extraction
# ---------------------------------------------------------------------------
def subtree_nodes(root: int, children_indices: list[list[int]], max_nodes: int = 32) -> list[int]:
    """Nodes in the subtree rooted at ``root`` (BFS, capped for tractable OT)."""
    out: list[int] = []
    queue = [root]
    while queue and len(out) < max_nodes:
        v = queue.pop(0)
        out.append(v)
        queue.extend(children_indices[v])
    return out


# ---------------------------------------------------------------------------
# Optimal transport
# ---------------------------------------------------------------------------
def _ground_cost(P: Tensor, Q: Tensor, manifold=None) -> Tensor:
    """(n, m) ground-cost matrix: geodesic if a manifold is given, else Euclidean."""
    if manifold is None:
        return torch.cdist(P, Q)
    n, m = P.shape[0], Q.shape[0]
    out = torch.empty(n, m, device=P.device)
    for i in range(n):                                   # subtrees are capped (<=32)
        d = manifold.distance(P[i : i + 1].expand(m, -1), Q)
        out[i] = d.squeeze(-1) if d.dim() > 1 else d
    return out


def _sinkhorn_plan(a: Tensor, b: Tensor, C: Tensor, eps_frac: float = 0.05,
                   iters: int = 100) -> Tensor:
    """Entropic-OT coupling matrix via stable log-domain Sinkhorn.

    ``eps`` is set adaptively to ``eps_frac * mean(C)`` so the regulariser scales
    with the geodesic magnitude (which varies with curvature/depth).
    """
    eps = eps_frac * C.mean().clamp_min(1e-6)
    log_a, log_b = a.log(), b.log()
    logK = -C / eps
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(iters):
        f = log_a - torch.logsumexp(logK + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(logK + f.unsqueeze(1), dim=0)
    return (f.unsqueeze(1) + logK + g.unsqueeze(0)).exp()


def _sinkhorn_cost(a: Tensor, b: Tensor, C: Tensor, **kw) -> float:
    """Entropic-OT transport cost <P, C> (mass-weighted Sinkhorn plan)."""
    return float((_sinkhorn_plan(a, b, C, **kw) * C).sum())


def _transport_plan(a: Tensor, b: Tensor, C: Tensor) -> Tensor:
    """Optimal coupling gamma (n, m): exact via POT if available, else Sinkhorn."""
    try:
        import ot                                         # POT, exact EMD plan
        g = ot.emd(a.cpu().numpy(), b.cpu().numpy(), C.cpu().numpy())
        return torch.as_tensor(g, dtype=C.dtype, device=C.device)
    except ImportError:
        return _sinkhorn_plan(a, b, C)


def wasserstein(P: Tensor, Q: Tensor, manifold=None, *, weights=None) -> float:
    """W between two point sets (uniform mass unless ``weights=(a, b)`` given).

    Uses POT's exact EMD if available, else the Sinkhorn approximation.
    """
    n, m = P.shape[0], Q.shape[0]
    if n == 0 or m == 0:
        return float("nan")
    C = _ground_cost(P, Q, manifold)
    if weights is None:
        a = torch.full((n,), 1.0 / n, device=P.device)
        b = torch.full((m,), 1.0 / m, device=P.device)
    else:
        a, b = weights
    try:
        import ot                                        # POT, exact EMD
        return float(ot.emd2(a.cpu().numpy(), b.cpu().numpy(), C.cpu().numpy()))
    except ImportError:
        return _sinkhorn_cost(a, b, C)


def wasserstein_profile(P: Tensor, Q: Tensor, manifold=None, *, weights=None) -> dict[str, float]:
    """W plus the *distribution* of the transport cost between two point sets.

    The scalar ``W`` hides *where* the divergence lives. The optimal coupling
    does not, so we keep it and summarise the per-source displacement
    ``t_i = cost_i / a_i`` (whose mass-mean equals ``W``):

    - ``W``            : scalar Wasserstein (mean per-point displacement).
    - ``median``/``p90``: quantiles of ``t_i``. ``median << W`` => the distance
      is driven by a few far points, not a broad fork.
    - ``concentration``: share of total transport cost from the top-10% costliest
      source points (0..1). High => outlier-driven (the "spread" confound).
    - ``fork_dist``    : geodesic between the two subtree roots ``P[0], Q[0]``
      (``subtree_nodes`` emits the root first). Small ``fork_dist`` + large ``W``
      => branches start together and diverge only deeper.
    """
    keys = ("W", "median", "p90", "concentration", "fork_dist")
    n, m = P.shape[0], Q.shape[0]
    if n == 0 or m == 0:
        return {k: float("nan") for k in keys}
    C = _ground_cost(P, Q, manifold)
    if weights is None:
        a = torch.full((n,), 1.0 / n, device=P.device)
        b = torch.full((m,), 1.0 / m, device=P.device)
    else:
        a, b = weights
    G = _transport_plan(a, b, C)
    cost_i = (G * C).sum(1)                               # cost per source point; sums to W
    total = cost_i.sum().clamp_min(1e-12)
    t = cost_i / a.clamp_min(1e-12)                       # per-source displacement
    k = max(1, int(round(0.10 * n)))                     # top-10% costliest points
    top = torch.sort(cost_i, descending=True).values[:k].sum()
    if manifold is None:
        fork = float(torch.cdist(P[:1], Q[:1]).squeeze())
    else:
        fork = float(manifold.distance(P[:1], Q[:1]).squeeze())
    return {
        "W": float(cost_i.sum()),
        "median": float(t.median()),
        "p90": float(torch.quantile(t, 0.90)),
        "concentration": float(top / total),
        "fork_dist": fork,
    }


# ---------------------------------------------------------------------------
# Branch divergence at a parent
# ---------------------------------------------------------------------------
def branch_divergence(
    parent: int,
    h_all: Tensor,
    children_indices: list[list[int]],
    *,
    manifold=None,
    max_nodes: int = 32,
    max_children: int = 6,
    profile: bool = False,
) -> dict[str, float]:
    """Mean / max pairwise Wasserstein across ``parent``'s child subtrees.

    Returns ``{mean, max, n_children}``; mean/max are NaN if the parent has < 2
    children (no fork to measure). Children are capped at ``max_children`` to
    bound the O(children^2) pairwise cost.

    With ``profile=True`` also returns ``{fork_dist, median, concentration}`` for
    the *most divergent* child pair (the one defining ``max``) — i.e. whether that
    top distance is a broad fork or a few scattered descendants. See
    ``wasserstein_profile``.
    """
    kids = children_indices[parent][:max_children]
    prof_nan = {"fork_dist": float("nan"), "median": float("nan"),
                "concentration": float("nan")}
    if len(kids) < 2:
        out = {"mean": float("nan"), "max": float("nan"), "n_children": len(kids)}
        return {**out, **prof_nan} if profile else out
    dists = [h_all[torch.tensor(subtree_nodes(c, children_indices, max_nodes),
                                dtype=torch.long)] for c in kids]
    ws, pairs = [], []
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            ws.append(wasserstein(dists[i], dists[j], manifold))
            pairs.append((i, j))
    t = torch.tensor(ws)
    out = {"mean": float(t.nanmean()), "max": float(t.max()), "n_children": len(kids)}
    if profile:
        bi, bj = pairs[int(torch.argmax(t))]             # most divergent child pair
        prof = wasserstein_profile(dists[bi], dists[bj], manifold)
        out.update({"fork_dist": prof["fork_dist"], "median": prof["median"],
                    "concentration": prof["concentration"]})
    return out


def divergence_anchors(
    h_all: Tensor,
    children_indices: list[list[int]],
    *,
    manifold=None,
    n_anchors: int = 128,
    max_nodes: int = 32,
    max_children: int = 6,
    seed: int = 0,
) -> list[tuple[int, float]]:
    """Rank parents by mean pairwise child divergence (descending).

    Samples up to ``n_anchors`` parents with >= 2 children. Returns
    ``[(parent_id, mean_W), ...]`` sorted high-to-low — the top entries are the
    most polarizing "Divergence Anchors".
    """
    N = len(children_indices)
    candidates = [v for v in range(N) if len(children_indices[v]) >= 2]
    if not candidates:
        return []
    gen = torch.Generator().manual_seed(seed)
    if len(candidates) > n_anchors:
        idx = torch.randperm(len(candidates), generator=gen)[:n_anchors].tolist()
        candidates = [candidates[i] for i in idx]
    scored = []
    for p in candidates:
        bd = branch_divergence(p, h_all, children_indices, manifold=manifold,
                               max_nodes=max_nodes, max_children=max_children)
        if bd["mean"] == bd["mean"]:                     # not NaN
            scored.append((p, bd["mean"]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _null_divergence(
    h_all: Tensor,
    children_indices: list[list[int]],
    *,
    manifold=None,
    n_samples: int = 200,
    max_nodes: int = 32,
    seed: int = 0,
) -> tuple[float, float]:
    """Chance-level divergence: (mean, std) of W over random NON-sibling subtree pairs.

    This is the reference a fork must beat. Scattered instance-sets (e.g. 'gulf')
    have large absolute W simply because the embedding is spread out — they sit
    near this null. Genuine forks ('body') exceed it.
    """
    parents = _parents_from_children(children_indices)
    internal = [v for v in range(len(children_indices)) if children_indices[v]]
    if len(internal) < 2:
        return float("nan"), float("nan")
    gen = torch.Generator().manual_seed(seed)
    ws, tries = [], 0
    while len(ws) < n_samples and tries < n_samples * 10:
        tries += 1
        i, j = torch.randint(len(internal), (2,), generator=gen).tolist()
        u, v = internal[i], internal[j]
        if u == v or (set(parents[u]) & set(parents[v])):    # skip self / siblings
            continue
        Pu = h_all[torch.tensor(subtree_nodes(u, children_indices, max_nodes), dtype=torch.long)]
        Pv = h_all[torch.tensor(subtree_nodes(v, children_indices, max_nodes), dtype=torch.long)]
        ws.append(wasserstein(Pu, Pv, manifold))
    if not ws:
        return float("nan"), float("nan")
    t = torch.tensor(ws)
    return float(t.mean()), float(t.std())


def relative_divergence_anchors(
    h_all: Tensor,
    children_indices: list[list[int]],
    *,
    manifold=None,
    n_anchors: int = 128,
    max_nodes: int = 32,
    max_children: int = 6,
    seed: int = 0,
    n_null: int = 200,
) -> tuple[list[tuple[int, float, float]], tuple[float, float]]:
    """Anchors ranked by divergence *beyond chance* (z-score vs the null).

    Returns ``([(parent, z, raw_W), ...] sorted by z desc, (null_mean, null_std))``.
    z = (W_siblings - null_mean) / null_std, so it answers "how much more
    divergent than random subtrees" — filtering out the spread-only confound.
    """
    raw = divergence_anchors(h_all, children_indices, manifold=manifold,
                             n_anchors=n_anchors, max_nodes=max_nodes,
                             max_children=max_children, seed=seed)
    if not raw:
        return [], (float("nan"), float("nan"))
    mu, sd = _null_divergence(h_all, children_indices, manifold=manifold,
                              n_samples=n_null, max_nodes=max_nodes, seed=seed)
    out = []
    for p, w in raw:
        if mu == mu and sd and sd > 0:
            z = (w - mu) / sd
        elif mu == mu:
            z = w - mu                       # std unavailable: use raw offset
        else:
            z = w                            # null unavailable: fall back to raw
        out.append((p, z, w))
    out.sort(key=lambda x: x[1], reverse=True)
    return out, (mu, sd)


def compute_branch_divergence(
    h_all: Tensor,
    children_indices: list[list[int]],
    *,
    manifold=None,
    n_anchors: int = 128,
    max_nodes: int = 32,
    max_children: int = 6,
    seed: int = 0,
) -> dict[str, float]:
    """Aggregate branch-divergence metrics for ``compute_structure_metrics``.

    ``branch_divergence_rel_mean`` (mean sibling W minus the chance null) is the
    real monoculture indicator: <= 0 means siblings are no more divergent than
    random subtrees (redundant branches); > 0 means genuine forks exist.
    """
    ranked, (mu, sd) = relative_divergence_anchors(
        h_all, children_indices, manifold=manifold, n_anchors=n_anchors,
        max_nodes=max_nodes, max_children=max_children, seed=seed)
    if not ranked:
        return {"branch_divergence_mean": float("nan"),
                "branch_divergence_rel_mean": float("nan"),
                "branch_divergence_null": float("nan"),
                "branch_divergence_z_max": float("nan"),
                "branch_divergence_n": 0}
    raw = torch.tensor([w for _, _, w in ranked])
    z = torch.tensor([zz for _, zz, _ in ranked])
    return {
        "branch_divergence_mean": float(raw.mean()),
        "branch_divergence_rel_mean": float(raw.mean()) - mu if mu == mu else float("nan"),
        "branch_divergence_null": mu,
        "branch_divergence_z_max": float(z.max()),
        "branch_divergence_n": len(ranked),
    }


# ---------------------------------------------------------------------------
# Standalone ranking CLI
# ---------------------------------------------------------------------------
def _main():
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser(description="Rank Divergence Anchors in a frozen embedding")
    ap.add_argument("--embeddings", required=True, help=".pt of h_all on the ball")
    ap.add_argument("--dataset", choices=["wn18rr", "culturalbench"], default="wn18rr")
    ap.add_argument("--data_dir", default="data/wn18rr")
    ap.add_argument("--curvature", type=float, default=1.0)
    ap.add_argument("--euclidean", action="store_true", help="use Euclidean ground metric")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--n_anchors", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true",
                    help="show the transport-cost distribution per anchor "
                         "(fork_dist, median displacement, concentration)")
    args = ap.parse_args()

    from pluraltree.manifolds.poincare import PoincareBall
    if args.dataset == "wn18rr":
        from data.wordnet import load_wn18rr
        graph = load_wn18rr(data_dir=args.data_dir, split_seed=args.seed, leakage_safe=True)
    else:
        from data.culturalbench import load_culturalbench
        graph = load_culturalbench(split_seed=args.seed, leakage_safe=True)

    h_all = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(h_all, torch.Tensor):
        h_all = h_all["h_all"]
    manifold = None if args.euclidean else PoincareBall(c=args.curvature)

    # Prefer human-readable glosses (entity2text) over raw synset-offset ids.
    text = getattr(graph, "entity_text", {})

    def label(nid: int, width: int = 28) -> str:
        s = text.get(nid) or graph.id_to_entity[nid]
        s = s.split(",")[0].strip()                  # first sense / short form
        return s if len(s) <= width else s[: width - 1] + "…"

    ranked, (mu, sd) = relative_divergence_anchors(
        h_all, graph.children_indices, manifold=manifold,
        n_anchors=args.n_anchors, seed=args.seed)
    agg = compute_branch_divergence(h_all, graph.children_indices, manifold=manifold,
                                    n_anchors=args.n_anchors, seed=args.seed)
    print(f"null(random pairs) mean={mu:.4f} std={sd:.4f}  |  "
          f"sibling mean={agg['branch_divergence_mean']:.4f}  "
          f"rel_mean(siblings-null)={agg['branch_divergence_rel_mean']:+.4f}  "
          f"z_max={agg['branch_divergence_z_max']:.2f}")
    print(f"\nTop {args.top} Divergence Anchors (ranked by z = divergence beyond chance):")
    if args.profile:
        print("  (profile = most divergent child pair: fork=root-root geodesic, "
              "med=median displacement, conc=top-10% cost share)")
    for pid, z, w in ranked[: args.top]:
        kids = ", ".join(label(c, 20) for c in graph.children_indices[pid][:5])
        line = f"  z={z:+.2f}  W={w:.2f}  [{label(pid)}]"
        if args.profile:
            bd = branch_divergence(pid, h_all, graph.children_indices,
                                   manifold=manifold, profile=True)
            line += (f"  fork={bd['fork_dist']:.2f}  med={bd['median']:.2f}"
                     f"  conc={bd['concentration']:.2f}")
        print(line + f"\n        children: {kids}")


if __name__ == "__main__":
    _main()
