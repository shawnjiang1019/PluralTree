"""Hyperbolic link prediction scoring head.

Scores a triple (s, r, o) using TransE-style translation in hyperbolic space:

    score(s, r, o) = -d_H(h_s ⊕_c r, h_o)

where ⊕_c is Möbius addition and d_H is hyperbolic distance.
Higher score = more plausible triple.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

try:
    import geoopt
    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False

from pluraltree.manifolds.poincare import PoincareBall


class HyperbolicLinkPredictor(nn.Module):
    """Scores (subject, relation, object) triples in hyperbolic space.

    Learns one relation embedding per relation type. At scoring time,
    translates the subject embedding by the relation vector and measures
    hyperbolic distance to the object embedding.
    """

    def __init__(self, num_relations: int, d_hidden: int, manifold: PoincareBall):
        super().__init__()
        self.manifold = manifold
        self.d_hidden = d_hidden

        # Relation embeddings on the Poincaré ball
        if HAS_GEOOPT:
            ball = geoopt.PoincareBall(c=manifold.c.item())
            self.relation_embeddings = geoopt.ManifoldParameter(
                torch.zeros(num_relations, d_hidden), manifold=ball
            )
        else:
            self.relation_embeddings = nn.Parameter(torch.zeros(num_relations, d_hidden))

        nn.init.uniform_(self.relation_embeddings, -0.001, 0.001)

    def score(
        self,
        h_subjects: Tensor,
        r_ids: Tensor,
        h_objects: Tensor,
    ) -> Tensor:
        """Compute triple scores.

        Args:
            h_subjects: (B, d_hidden) subject embeddings on the ball
            r_ids:      (B,) relation ids
            h_objects:  (B, d_hidden) object embeddings on the ball

        Returns:
            (B,) scores — higher is more plausible
        """
        r = self.relation_embeddings[r_ids]                  # (B, d)
        r = self.manifold.project(r)
        translated = self.manifold.mobius_add(h_subjects, r) # (B, d) on ball
        dist = self.manifold.distance(translated, h_objects).squeeze(-1)  # (B,)
        return -dist

    def score_all_candidates(
        self,
        h_subject: Tensor,
        r_id: int,
        h_candidates: Tensor,
    ) -> Tensor:
        """Score a single subject against all candidate objects.

        Args:
            h_subject:    (d_hidden,) subject embedding
            r_id:         relation id (int)
            h_candidates: (K, d_hidden) candidate object embeddings

        Returns:
            (K,) scores
        """
        r = self.relation_embeddings[r_id].unsqueeze(0)         # (1, d)
        r = self.manifold.project(r)
        h_s = h_subject.unsqueeze(0)                            # (1, d)
        translated = self.manifold.mobius_add(h_s, r)           # (1, d)
        translated = translated.expand(h_candidates.shape[0], -1)
        dist = self.manifold.distance(translated, h_candidates).squeeze(-1)
        return -dist
