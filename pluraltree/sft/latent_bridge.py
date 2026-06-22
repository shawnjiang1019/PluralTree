"""Latent bridge: Poincaré point -> tangent space -> k soft tokens for the LM.

The Map Reader / Reasoner receives the hyperbolic embedding as a small block of
soft tokens prepended to the input sequence. The mapping is:

    h (on the ball)  --log_map_zero-->  v (tangent at origin)  --MLP-->  k x d_model

Only this module and the LM's LoRA adapter are trained; the LM and the frozen
``h_all`` are not. log_map_zero is the standard tangent-space linearisation used
everywhere else in the codebase, so the bridge is differentiable end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from pluraltree.manifolds.poincare import PoincareBall


class LatentBridge(nn.Module):
    """Project ``n`` Poincaré points to ``n * n_tokens`` LM soft-token embeddings.

    forward(h): h is ``(B, d_latent)`` or ``(B, M, d_latent)`` (M latents per
    example, e.g. M structural cousins). Returns ``(B, M * n_tokens, d_model)``.
    """

    def __init__(
        self,
        manifold: PoincareBall,
        d_latent: int,
        d_model: int,
        n_tokens: int = 4,
        d_hidden: int | None = None,
    ):
        super().__init__()
        self.manifold = manifold
        self.n_tokens = n_tokens
        self.d_model = d_model
        d_hidden = d_hidden or 2 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_tokens * d_model),
        )
        # Keep soft tokens at a sane scale relative to the LM's token embeddings.
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, h: Tensor) -> Tensor:
        squeeze = h.dim() == 2
        if squeeze:
            h = h.unsqueeze(1)                       # (B, 1, d_latent)
        B, M, d = h.shape
        v = self.manifold.log_map_zero(h.reshape(B * M, d))   # tangent at origin
        tok = self.mlp(v).reshape(B * M * self.n_tokens, self.d_model)
        tok = self.out_norm(tok).reshape(B, M * self.n_tokens, self.d_model)
        return tok

    @property
    def tokens_per_latent(self) -> int:
        return self.n_tokens
