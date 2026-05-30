"""Full tree encoder with GKI — bottom-up recursion with knowledge injection."""

import torch
import torch.nn as nn
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall
from pluraltree.knowledge.base import KnowledgeSource
from .gki_tree_gru_cell import GKITreeGRUCell, InjectionPoint
from .knowledge_schedule import KnowledgeSchedule


class GKITreeEncoder(nn.Module):
    """Encodes a rooted tree with Hyperbolic Tree-GRU + Gated Knowledge Injection.

    Bottom-up recursion with depth-aware knowledge injection at each node.
    Supports a knowledge curriculum that gradually enables injection during training.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        manifold: PoincareBall,
        sources: list[KnowledgeSource],
        injection_point: InjectionPoint = InjectionPoint.POST_AGGREGATION,
        gate_bias: float = -2.0,
        depth_aware: bool = True,
        inject: bool = True,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.manifold = manifold
        self.input_proj = nn.Linear(d_input, d_hidden)
        self.cell = GKITreeGRUCell(
            d_hidden, d_hidden, manifold, sources, injection_point, gate_bias,
            depth_aware=depth_aware, inject=inject,
        )
        self.schedule = None

    def set_schedule(self, schedule: KnowledgeSchedule) -> None:
        self.schedule = schedule

    def forward(
        self,
        node_features: Tensor,
        node_ids: Tensor,
        children_indices: list[list[int]],
        topo_order: list[int],
        training_step: int = 0,
    ) -> Tensor:
        """Encode a tree bottom-up with knowledge injection.

        Args:
            node_features: (N, d_input) features for all N nodes
            node_ids: (N,) node identifiers for knowledge lookup
            children_indices: children_indices[i] = list of child indices for node i
            topo_order: topological order (leaves first)
            training_step: current step for knowledge curriculum

        Returns:
            (N, d_hidden) hidden states on the Poincaré ball
        """
        # Apply knowledge schedule if set
        if self.schedule is not None and self.training:
            bias = self.schedule.get_gate_bias(training_step)
            for p in self.cell.gki.gate.parameters():
                if p.shape == self.cell.gki.gate.W_g.bias.shape:
                    with torch.no_grad():
                        p.fill_(bias)

        N = node_features.shape[0]
        device = node_features.device

        x_tan = self.input_proj(node_features)
        h = torch.zeros(N, self.d_hidden, device=device)

        for idx in topo_order:
            children = children_indices[idx]

            if len(children) == 0:
                h_agg_tan = torch.zeros(1, self.d_hidden, device=device)
                h_node = self.cell.tree_gru_cell.gru_step(
                    x_tan[idx : idx + 1], h_agg_tan
                )
                # Apply post-agg or post-GRU injection even for leaves
                if self.cell.inject:
                    h_node = self.cell.gki(h_node, node_ids[idx : idx + 1])
            else:
                h_ch = torch.stack([h[c] for c in children], dim=0).unsqueeze(1)
                mask = torch.ones(len(children), 1, dtype=torch.bool, device=device)
                child_nids = torch.stack([node_ids[c] for c in children]).unsqueeze(1)

                h_node = self.cell(
                    x_tan[idx : idx + 1],
                    h_ch,
                    node_ids[idx : idx + 1],
                    child_nids,
                    mask,
                )

            h[idx] = h_node.squeeze(0)

        return h
