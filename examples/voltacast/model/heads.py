"""Shared output head for quantile models."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def monotone_quantiles(out: torch.Tensor) -> torch.Tensor:
    """Turn free logits into non-crossing quantiles along the last axis.

    The first quantile is unconstrained; each subsequent one adds a
    non-negative softplus increment, so P10 ≤ P50 ≤ P90 holds for any input.
    This is the output contract both models must share — hence one function.
    """
    base = out[..., :1]
    increments = F.softplus(out[..., 1:])
    return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
