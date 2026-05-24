"""Knowledge source backed by pretrained KG embeddings (TransE, RotatE, etc.)."""

import torch
import torch.nn as nn
from torch import Tensor

from .base import KnowledgeSource


class KGEmbeddingSource(KnowledgeSource, nn.Module):
    """Looks up pretrained entity embeddings from a knowledge graph model."""

    def __init__(self, num_entities: int, embedding_dim: int, pretrained: Tensor | None = None, freeze: bool = True):
        nn.Module.__init__(self)
        self._dim = embedding_dim
        if pretrained is not None:
            self.embeddings = nn.Embedding.from_pretrained(pretrained, freeze=freeze)
        else:
            self.embeddings = nn.Embedding(num_entities, embedding_dim)
            if freeze:
                self.embeddings.weight.requires_grad_(False)

    def lookup(self, node_ids: Tensor) -> Tensor:
        return self.embeddings(node_ids)

    @property
    def output_dim(self) -> int:
        return self._dim
