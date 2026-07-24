"""Group-relative advantage for GRPO.

GRPO drops the value network: for a GROUP of G rollouts sampled from the same
prompt, the baseline is the group's own mean reward, and the advantage is the
z-scored reward within the group,

    A_i = (r_i - mean(r)) / (std(r) + eps).

Every token of rollout i is trained with the same scalar A_i (times the clipped
importance ratio, handled by the trainer). This module is the standalone,
trainer-agnostic normalization so it can be unit-tested and reused if we ever
drop TRL. See docs/grpo_alignment.txt.
"""

from __future__ import annotations

import statistics as st
from typing import Sequence


def group_relative_advantage(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    """Z-score rewards within one group. Zero-mean by construction; a group whose
    rewards are all equal yields all-zero advantages (no signal, as intended)."""
    n = len(rewards)
    if n == 0:
        return []
    mean = st.mean(rewards)
    if n == 1:
        return [0.0]
    sd = st.pstdev(rewards)
    if sd < eps:
        return [0.0] * n
    return [(r - mean) / (sd + eps) for r in rewards]


def grouped_advantages(rewards: Sequence[float], group_size: int,
                       eps: float = 1e-6) -> list[float]:
    """Advantages for a flat batch laid out as consecutive groups of ``group_size``.

    len(rewards) must be a multiple of group_size (one prompt per group).
    """
    if group_size <= 0 or len(rewards) % group_size != 0:
        raise ValueError(f"len(rewards)={len(rewards)} not a multiple of "
                         f"group_size={group_size}")
    out: list[float] = []
    for i in range(0, len(rewards), group_size):
        out.extend(group_relative_advantage(rewards[i:i + group_size], eps))
    return out
