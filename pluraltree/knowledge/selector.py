"""Multi-source knowledge selector using attention-based routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KnowledgeSelector(nn.Module):
    """Selects and combines knowledge from multiple sources via attention.

    Given a hidden state h and a list of projected knowledge vectors k_1..k_S,
    computes attention weights alpha_1..alpha_S and returns a weighted combination.
    """

    def __init__(self, d_hidden: int, n_sources: int):
        super().__init__()
        self.W_query = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_keys = nn.ModuleList([nn.Linear(d_hidden, d_hidden, bias=False) for _ in range(n_sources)])
        self.scale = d_hidden ** 0.5

    def forward(self, h: Tensor, k_list: list[Tensor]) -> Tensor:
        """Select and combine knowledge sources.

        Args:
            h: (batch, d_hidden) hidden state (in tangent space if hyperbolic)
            k_list: list of S tensors, each (batch, d_hidden)

        Returns:
            (batch, d_hidden) combined knowledge vector
        """
        query = self.W_query(h)
        scores = torch.stack(
            [torch.sum(query * W_k(k_s), dim=-1) / self.scale for W_k, k_s in zip(self.W_keys, k_list)],
            dim=-1,
        )  # (batch, n_sources)
        alpha = F.softmax(scores, dim=-1)  # (batch, n_sources)
        combined = sum(a.unsqueeze(-1) * k for a, k in zip(alpha.unbind(-1), k_list))
        return combined
