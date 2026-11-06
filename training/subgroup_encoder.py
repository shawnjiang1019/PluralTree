"""Simple alternative to the Tree-GRU: place each subgroup directly.

A 2-layer MLP maps a subgroup's frozen features -- its survey item q, its identity
g ("attribute: group"), and, in the qgp variant, its answer barycenter p -- to a
tangent vector, soft-capped and mapped onto the ball by exp_map_zero. Nothing
touches a leaf after that map, so the radius bound holds by construction; the
Tree-GRU's collapse came from an uncapped midpoint running after it on leaves.

Internal nodes have no parameters: each is the true hyperbolic midpoint of its
children, computed bottom-up. The saved h_all has the same shape and node ids as
the Tree-GRU's, so retrieval uses it unchanged.

    python -m training.subgroup_encoder --selftest
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

LEVELS = ("axis", "question", "subtopic", "topic", "root")


class SubgroupEncoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, manifold, d_hidden: int = 256):
        super().__init__()
        self.manifold = manifold
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.manifold.exp_map_zero(self.net(x))


def uniform_midpoints(x: Tensor, parent: Tensor, n: int, c: float) -> Tensor:
    """Equal-weight gyromidpoint of the x sharing each ``parent`` id, in (n, d).

    Einstein midpoint in Klein coordinates, then back to Poincare (the 1/2-otimes
    step). Parents with no children stay at the origin.
    """
    lam = 2.0 / (1.0 - c * (x * x).sum(-1, keepdim=True)).clamp_min(1e-6)
    num = torch.zeros(n, x.shape[1], device=x.device, dtype=x.dtype).index_add(
        0, parent, lam * x)
    den = torch.zeros(n, 1, device=x.device, dtype=x.dtype).index_add(0, parent, lam - 1.0)
    k = num / den.clamp_min(1e-9)
    s = (c * (k * k).sum(-1, keepdim=True)).clamp_max(1.0 - 1e-6)
    return k / (1.0 + torch.sqrt(1.0 - s))


def tree_levels(children_indices: list[list[int]], types: dict) -> list[tuple[Tensor, Tensor]]:
    """[(child ids, parent ids)] per level, leaves upward, over attached edges."""
    out = []
    for level in LEVELS:
        ch, par = [], []
        for v, t in types.items():
            if t == level:
                for c in children_indices[v]:
                    ch.append(c)
                    par.append(v)
        if ch:
            out.append((torch.tensor(ch, dtype=torch.long),
                        torch.tensor(par, dtype=torch.long)))
    return out


def full_positions(h_leaf: Tensor, leaf_ids: Tensor, levels, n: int, c: float) -> Tensor:
    """(n, d) positions: encoded leaves, then midpoints up the tree."""
    h = torch.zeros(n, h_leaf.shape[1], device=h_leaf.device, dtype=h_leaf.dtype)
    h = h.index_copy(0, leaf_ids, h_leaf)
    for ch, par in levels:
        mid = uniform_midpoints(h[ch], par, n, c)
        ids = par.unique()
        h = h.index_copy(0, ids, mid[ids])
    return h


def _selftest() -> None:
    from pluraltree.manifolds.poincare import PoincareBall

    c = 0.5
    man = PoincareBall(c=c)
    man.tangent_cap = 1.0
    torch.manual_seed(0)
    x = man.exp_map_zero(torch.randn(6, 8))
    # one child: the midpoint is the child itself
    m = uniform_midpoints(x, torch.arange(6), 6, c)
    assert (m - x).norm(dim=-1).max() < 1e-5, "midpoint(x) != x"
    # two children: equidistant, and half the distance between them
    p = torch.tensor([0, 0])
    a, b = x[:1], x[1:2]
    m = uniform_midpoints(x[:2], p, 1, c)
    da, db, ab = (float(man.distance(a, m)), float(man.distance(m, b)),
                  float(man.distance(a, b)))
    assert abs(da - db) < 1e-4 and abs(da - ab / 2) < 1e-4, (da, db, ab)
    # root(0) -> axis(1) -> leaves 2,3; root -> axis(4) -> leaf 5
    kids = [[1, 4], [2, 3], [], [], [5], []]
    types = {0: "root", 1: "axis", 2: "opinion", 3: "opinion", 4: "axis", 5: "opinion"}
    levels = tree_levels(kids, types)
    leaf = torch.tensor([2, 3, 5])
    enc = SubgroupEncoder(4, 8, man, d_hidden=16)
    h = full_positions(enc(torch.randn(3, 4)), leaf, levels, 6, c)
    assert h.shape == (6, 8) and torch.isfinite(h).all()
    assert (h[4] - h[5]).norm() < 1e-5, "single-child axis must equal its leaf"
    assert float((c ** 0.5 * h.norm(dim=-1)).max()) < 0.62, "radius cap broken"
    h.sum().backward()                           # gradients reach the encoder
    assert all(q.grad is not None for q in enc.parameters())
    print("subgroup_encoder self-test OK: true midpoints, radius cap, gradients")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
