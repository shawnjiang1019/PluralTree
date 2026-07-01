"""Deterministic entity-identity decode: embedding -> node id.

``argmin`` over a frozen codebook (the final ``h_all`` on the ball) under the
Poincaré geodesic. A pure function — no sampling, no learned generation, fully
reproducible. Determinism is unconditional; the ``lambda_decode`` training term
only widens each entry's basin so an *inexact* query still lands on the right id.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall


class CodebookDecoder:
    """Frozen codebook + geodesic nearest-neighbour lookup."""

    def __init__(self, codebook: Tensor, manifold: PoincareBall):
        self.codebook = codebook
        self.manifold = manifold

    @torch.no_grad()
    def decode(self, query: Tensor) -> Tensor:
        """(d,) -> () or (B, d) -> (B,) long ids. argmin geodesic; ``argmin``
        returns the first minimum, so ties break to the lowest id (deterministic)."""
        single = query.dim() == 1
        if single:
            query = query.unsqueeze(0)
        N = self.codebook.shape[0]
        out = torch.empty(query.shape[0], dtype=torch.long)
        for b in range(query.shape[0]):                 # offline; row-wise bounds memory
            d = self.manifold.distance(query[b : b + 1].expand(N, -1), self.codebook)
            out[b] = int(torch.argmin(d.squeeze(-1)))
        return out[0] if single else out

    @torch.no_grad()
    def accuracy(self, queries: Tensor, ids: Tensor) -> float:
        """Decode accuracy of ``queries`` against their true ``ids`` (self-decode = 1.0)."""
        return float((self.decode(queries) == ids.cpu()).float().mean())

    def save(self, path: str) -> None:
        torch.save({"codebook": self.codebook.cpu(), "c": float(self.manifold.c)}, path)

    @classmethod
    def load(cls, path: str) -> "CodebookDecoder":
        d = torch.load(path, map_location="cpu")
        return cls(d["codebook"], PoincareBall(c=d["c"]))
