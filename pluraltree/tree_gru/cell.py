"""Hyperbolic Tree-GRU cell operating in the Poincaré ball."""

import torch
import torch.nn as nn
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall
from .aggregation import ChildAggregator


class HyperbolicTreeGRUCell(nn.Module):
    """A single recursion step of the Hyperbolic Tree-GRU.

    Given node input features x_v and children hidden states {h_c1, ..., h_cN}:
    1. Aggregate children via Möbius weighted midpoint -> h_agg
    2. Compute GRU gates (update z, reset r) in tangent space
    3. Compute candidate state in tangent space
    4. Combine via gated update, map back to Poincaré ball
    """

    def __init__(self, d_input: int, d_hidden: int, manifold: PoincareBall,
                 child_attention: bool = True, tangent_scale: float = 0.0,
                 tangent_clip: float = 0.0):
        super().__init__()
        self.d_hidden = d_hidden
        self.manifold = manifold
        # SATURATION. exp_map_zero gives ||x|| = tanh(sqrt(c)||v||)/sqrt(c), so the
        # normalized radius is rho = tanh(sqrt(c)||v||): rho > 0.999 once
        # sqrt(c)||v|| > 3.8. h_new_tan is a convex mix of tanh-bounded terms, so
        # its norm grows like sqrt(d_hidden) -- at d=64 that is ~7, and at c=0.5
        # EVERY node lands on the projection clamp (measured: rho = 0.9999 for
        # 100% of nodes, on both the ATP and ISSP graphs). The radius channel then
        # carries no information and gradients through the map are attenuated by
        # sech^2(5) ~ 2e-4, which is also why a penalty on rho cannot fix it.
        # tangent_scale > 0 multiplies the tangent vector first; 1/sqrt(d_hidden)
        # puts ||v|| ~ 1 and rho ~ 0.6, mid-ball. 0.0 keeps the original behaviour
        # so existing embeddings stay reproducible.
        self.tangent_scale = tangent_scale
        # A SCALE IS NOT ENOUGH, measured: with tangent_scale=1/sqrt(64) the ATP
        # run still had p50 rho = 0.9998. h_new_tan mixes in h_agg_tan, which is
        # log_map_zero of the aggregated children -- and the log map of a point
        # near the rim has UNBOUNDED norm. Saturated children therefore produce a
        # huge tangent vector, which saturates the parent, on up the tree. A
        # constant factor only moves where that loop catches; a cap breaks it,
        # because rho <= tanh(sqrt(c) * clip) no matter what the children did
        # (clip=1.0 at c=0.5 gives rho <= 0.61). 0.0 disables.
        self.tangent_clip = tangent_clip
        self.aggregator = ChildAggregator(d_hidden, manifold, attention=child_attention)

        # GRU gates operate in tangent space (Euclidean)
        self.W_z = nn.Linear(d_hidden + d_input, d_hidden)  # update gate
        self.W_r = nn.Linear(d_hidden + d_input, d_hidden)  # reset gate
        self.W_n = nn.Linear(d_hidden + d_input, d_hidden)  # candidate

    def gru_step(self, x_v_tan: Tensor, h_agg_tan: Tensor) -> Tensor:
        """GRU computation in tangent space.

        Args:
            x_v_tan: (batch, d_input) node features in tangent space
            h_agg_tan: (batch, d_hidden) aggregated children in tangent space

        Returns:
            (batch, d_hidden) new hidden state on the Poincaré ball
        """
        combined = torch.cat([h_agg_tan, x_v_tan], dim=-1)

        z = torch.sigmoid(self.W_z(combined))  # update gate
        r = torch.sigmoid(self.W_r(combined))  # reset gate

        candidate_input = torch.cat([r * h_agg_tan, x_v_tan], dim=-1)
        n = torch.tanh(self.W_n(candidate_input))  # candidate

        h_new_tan = (1.0 - z) * h_agg_tan + z * n
        if self.tangent_scale > 0.0:
            h_new_tan = h_new_tan * self.tangent_scale
        if self.tangent_clip > 0.0:
            nrm = h_new_tan.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            h_new_tan = h_new_tan * torch.clamp(self.tangent_clip / nrm, max=1.0)
        return self.manifold.exp_map_zero(h_new_tan)

    def forward(
        self,
        x_v: Tensor,
        h_children: Tensor,
        children_mask: Tensor | None = None,
    ) -> Tensor:
        """Process one node in the tree.

        Args:
            x_v: (batch, d_input) node input features (in tangent space)
            h_children: (K, batch, d_hidden) children hidden states on the ball
            children_mask: (K, batch) boolean mask for valid children

        Returns:
            (batch, d_hidden) hidden state for this node, on the ball
        """
        # Aggregate children on the manifold
        h_agg = self.aggregator(h_children, children_mask)

        # Map aggregated state to tangent space for GRU computation
        h_agg_tan = self.manifold.log_map_zero(h_agg)

        return self.gru_step(x_v, h_agg_tan)
