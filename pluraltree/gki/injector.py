"""GKI Injector — orchestrates knowledge retrieval, selection, gating, and injection."""

import torch
import torch.nn as nn
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.knowledge.base import KnowledgeSource
from pluraltree.knowledge.projection import ProjectionLayer
from pluraltree.knowledge.selector import KnowledgeSelector
from .gate import EuclideanGate, HyperbolicGate


class GKIInjector(nn.Module):
    """Core GKI module: retrieves external knowledge and selectively injects it.

    Supports both Euclidean and hyperbolic modes. When a manifold is provided,
    all operations respect the Poincaré ball geometry.

    Workflow:
        1. Retrieve knowledge vectors from each source
        2. Project each to the hidden dimension (and onto manifold if hyperbolic)
        3. Select/combine sources via attention
        4. Compute injection gate
        5. Mix hidden state with knowledge via gated combination
    """

    def __init__(
        self,
        d_hidden: int,
        sources: list[KnowledgeSource],
        manifold: PoincareBall | None = None,
        gate_bias: float = -1.0,
    ):
        super().__init__()
        self.sources = nn.ModuleList(sources)
        self.manifold = manifold

        self.projections = nn.ModuleList(
            [ProjectionLayer(s.output_dim, d_hidden, manifold) for s in sources]
        )

        if len(sources) > 1:
            self.selector = KnowledgeSelector(d_hidden, len(sources))
        else:
            self.selector = None

        if manifold is None:
            self.gate = EuclideanGate(d_hidden, d_hidden, gate_bias)
        else:
            self.gate = HyperbolicGate(d_hidden, d_hidden, manifold, gate_bias)

    def forward(self, h: Tensor, node_ids: Tensor) -> Tensor:
        """Inject knowledge into hidden state.

        Args:
            h: (batch, d_hidden) hidden state
            node_ids: (batch,) node identifiers for knowledge lookup

        Returns:
            (batch, d_hidden) updated hidden state with injected knowledge
        """
        # 1. Retrieve and project
        k_list = [proj(src.lookup(node_ids)) for src, proj in zip(self.sources, self.projections)]

        # 2. Select/combine
        if self.selector is not None:
            if self.manifold is not None:
                # Compute selection in tangent space
                h_tan = self.manifold.log_map_zero(h)
                k_tan_list = [self.manifold.log_map_zero(k) for k in k_list]
                k_tan = self.selector(h_tan, k_tan_list)
                k = self.manifold.exp_map_zero(k_tan)
            else:
                k = self.selector(h, k_list)
        else:
            k = k_list[0]

        # 3. Compute gate
        g = self.gate(h, k)

        # 4. Inject
        if self.manifold is None:
            # Euclidean: h' = g * k + (1 - g) * h
            return g * k + (1.0 - g) * h
        else:
            # Hyperbolic: Möbius weighted midpoint
            points = torch.stack([h, k], dim=0)  # (2, batch, d)
            weights = torch.stack([1.0 - g, g], dim=0)  # (2, batch, d)
            return self.manifold.mobius_midpoint(points, weights)
