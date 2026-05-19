"""Lemma D2: isotropic nuisance Sigma_task = sigma^2 I."""

from __future__ import annotations

import torch

from pmh._tensor import ensure_psd


def estimate_d2(
    *,
    dim: int,
    noise_level: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if dim < 1:
        raise ValueError("dim must be positive")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")
    sigma2 = float(noise_level) ** 2
    return ensure_psd(sigma2 * torch.eye(dim, device=device, dtype=dtype), shrinkage=0.0)
