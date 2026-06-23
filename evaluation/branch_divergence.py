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


def _sinkhorn_cost(a: Tensor, b: Tensor, C: Tensor, eps_frac: float = 0.05,
                   iters: int = 100) -> float:
    """Entropic-OT transport cost <P, C> via stable log-domain Sinkhorn.

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
    logP = f.unsqueeze(1) + logK + g.unsqueeze(0)
    return float((logP.exp() * C).sum())


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
) -> dict[str, float]:
    """Mean / max pairwise Wasserstein across ``parent``'s child subtrees.

    Returns ``{mean, max, n_children}``; mean/max are NaN if the parent has < 2
    children (no fork to measure). Children are capped at ``max_children`` to
    bound the O(children^2) pairwise cost.
    """
    kids = children_indices[parent][:max_children]
    if len(kids) < 2:
        return {"mean": float("nan"), "max": float("nan"), "n_children": len(kids)}
    dists = [h_all[torch.tensor(subtree_nodes(c, children_indices, max_nodes),
                                dtype=torch.long)] for c in kids]
    ws = []
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            ws.append(wasserstein(dists[i], dists[j], manifold))
    t = torch.tensor(ws)
    return {"mean": float(t.nanmean()), "max": float(t.max()), "n_children": len(kids)}


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

    ``branch_divergence_mean`` is the monoculture indicator: low across the tree
    means branches are redundant despite the tree having many children.
    """
    scored = divergence_anchors(h_all, children_indices, manifold=manifold,
                                n_anchors=n_anchors, max_nodes=max_nodes,
                                max_children=max_children, seed=seed)
    if not scored:
        return {"branch_divergence_mean": float("nan"),
                "branch_divergence_max": float("nan"),
                "branch_divergence_n": 0}
    vals = torch.tensor([s for _, s in scored])
    return {
        "branch_divergence_mean": float(vals.mean()),
        "branch_divergence_max": float(vals.max()),
        "branch_divergence_n": len(scored),
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

    scored = divergence_anchors(h_all, graph.children_indices, manifold=manifold,
                                n_anchors=args.n_anchors, seed=args.seed)
    agg = compute_branch_divergence(h_all, graph.children_indices, manifold=manifold,
                                    n_anchors=args.n_anchors, seed=args.seed)
    print(f"branch_divergence_mean={agg['branch_divergence_mean']:.4f}  "
          f"max={agg['branch_divergence_max']:.4f}  scored={agg['branch_divergence_n']}")
    print(f"\nTop {args.top} Divergence Anchors:")
    for pid, score in scored[: args.top]:
        kids = ", ".join(label(c, 20) for c in graph.children_indices[pid][:5])
        print(f"  W={score:.3f}  [{label(pid)}]\n        children: {kids}")


if __name__ == "__main__":
    _main()
