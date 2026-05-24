"""Abstract base class for knowledge sources."""

from abc import ABC, abstractmethod
from torch import Tensor


class KnowledgeSource(ABC):
    """Interface for external knowledge providers.

    Each source maps node IDs to dense vectors representing
    external knowledge about those nodes.
    """

    @abstractmethod
    def lookup(self, node_ids: Tensor) -> Tensor:
        """Return knowledge vectors for given node IDs.

        Args:
            node_ids: (batch,) integer tensor of node identifiers

        Returns:
            (batch, output_dim) tensor of knowledge vectors
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the knowledge vectors produced by this source."""
