"""Full tree encoder — bottom-up recursion using the Hyperbolic Tree-GRU cell."""

import torch
import torch.nn as nn
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall
from .cell import HyperbolicTreeGRUCell


class TreeEncoder(nn.Module):
    """Encodes a rooted tree bottom-up via Hyperbolic Tree-GRU.

    Expects trees in topological order (leaves first, root last).
    """

    def __init__(self, d_input: int, d_hidden: int, manifold: PoincareBall, child_attention: bool = True):
        super().__init__()
        self.d_hidden = d_hidden
        self.manifold = manifold
        self.input_proj = nn.Linear(d_input, d_hidden)
        self.cell = HyperbolicTreeGRUCell(d_hidden, d_hidden, manifold, child_attention)

    def forward(
        self,
        node_features: Tensor,
        children_indices: list[list[int]],
        topo_order: list[int],
    ) -> Tensor:
        """Encode a tree bottom-up.

        Args:
            node_features: (N, d_input) features for all N nodes
            children_indices: children_indices[i] is a list of child node indices for node i
            topo_order: list of node indices in topological order (leaves first)

        Returns:
            (N, d_hidden) hidden states for all nodes, on the Poincaré ball
        """
        N = node_features.shape[0]
        device = node_features.device

        # Project input features to hidden dim (tangent space)
        x_tan = self.input_proj(node_features)  # (N, d_hidden)

        # Initialize hidden states on the manifold (at origin = zero vector)
        h = torch.zeros(N, self.d_hidden, device=device)

        for idx in topo_order:
            children = children_indices[idx]

            if len(children) == 0:
                # Leaf node: no children to aggregate, use zero as h_agg
                h_agg_tan = torch.zeros(1, self.d_hidden, device=device)
                h_node = self.cell.gru_step(x_tan[idx : idx + 1], h_agg_tan)
            else:
                # Gather children hidden states
                h_ch = torch.stack([h[c] for c in children], dim=0)  # (K, d_hidden)
                h_ch = h_ch.unsqueeze(1)  # (K, 1, d_hidden)
                mask = torch.ones(len(children), 1, dtype=torch.bool, device=device)

                h_node = self.cell(x_tan[idx : idx + 1], h_ch, mask)

            h[idx] = h_node.squeeze(0)

        return h
