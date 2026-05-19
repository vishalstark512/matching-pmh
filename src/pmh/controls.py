"""Falsification controls: wrong-W and signal-W projectors."""

from __future__ import annotations

import torch


def wrong_W_projector(
    d: int,
    rank: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random orthonormal columns U in R^d, rank r (Lemma C wrong-W control).

    Use Sigma_wrong = U U^T in pmh_penalty.
    """
    if rank <= 0 or rank > d:
        raise ValueError(f"rank must be in 1..{d}, got {rank}")
    g = generator or torch.Generator(device=device)
    A = torch.randn(d, rank, device=device, dtype=dtype, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q


def signal_W_projector(
    signal_direction: torch.Tensor,
) -> torch.Tensor:
    """Rank-1 projector onto a known signal direction (Corollaries E / E*)."""
    s = signal_direction.float().reshape(-1)
    norm = s.norm().clamp(min=1e-12)
    s = s / norm
    return torch.outer(s, s)
