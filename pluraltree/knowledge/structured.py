"""Knowledge source from structured node attributes (type, degree, schema features)."""

import torch
import torch.nn as nn
from torch import Tensor

from .base import KnowledgeSource


class StructuredSource(KnowledgeSource, nn.Module):
    """Encodes structured attributes into dense vectors.

    Attributes:
        type_vocab_size: Number of distinct entity types.
        type_dim: Embedding dimension for entity type.
        extra_features: Number of additional scalar features (degree, etc.).
        output_dim: Total output dimension after projection.
    """

    def __init__(self, type_vocab_size: int, type_dim: int = 32, extra_features: int = 0, out_dim: int = 64):
        nn.Module.__init__(self)
        self._out_dim = out_dim
        self.type_embed = nn.Embedding(type_vocab_size, type_dim)
        self.proj = nn.Linear(type_dim + extra_features, out_dim)

    def lookup(self, node_ids: Tensor, type_ids: Tensor | None = None, features: Tensor | None = None) -> Tensor:
        """Lookup structured attributes.

        For the base KnowledgeSource interface, only node_ids is required.
        type_ids and features can be passed when available.
        """
        if type_ids is None:
            type_ids = torch.zeros_like(node_ids)
        parts = [self.type_embed(type_ids)]
        if features is not None:
            parts.append(features)
        return self.proj(torch.cat(parts, dim=-1))

    @property
    def output_dim(self) -> int:
        return self._out_dim
