"""Knowledge curriculum schedule for staged training.

Phase 1: Gates nearly closed, base Tree-GRU representation learning
Phase 2: Gradually open gates, add knowledge sources
Phase 3: All sources active, unbiased gates
"""


class KnowledgeSchedule:
    """Controls injection intensity and source activation over training."""

    def __init__(
        self,
        warmup1: int = 5000,
        warmup2: int = 15000,
        initial_bias: float = -2.0,
        final_bias: float = 0.0,
        sources_order: list[str] | None = None,
    ):
        self.warmup1 = warmup1
        self.warmup2 = warmup2
        self.initial_bias = initial_bias
        self.final_bias = final_bias
        self.sources_order = sources_order or ["kg", "structured", "retrieval"]

    def get_gate_bias(self, step: int) -> float:
        """Get the gate bias for the current training step.

        Phase 1 (0 to warmup1): initial_bias (gates nearly closed)
        Phase 2 (warmup1 to warmup2): linear interpolation
        Phase 3 (warmup2+): final_bias (unbiased)
        """
        if step < self.warmup1:
            return self.initial_bias
        elif step < self.warmup2:
            progress = (step - self.warmup1) / (self.warmup2 - self.warmup1)
            return self.initial_bias + (self.final_bias - self.initial_bias) * progress
        else:
            return self.final_bias

    def get_active_sources(self, step: int) -> list[str]:
        """Get the list of active knowledge sources at the current step.

        Phase 1: first source only (typically KG embeddings)
        Phase 2: first two sources
        Phase 3: all sources
        """
        if step < self.warmup1:
            return self.sources_order[:1]
        elif step < self.warmup2:
            return self.sources_order[:2]
        else:
            return list(self.sources_order)
