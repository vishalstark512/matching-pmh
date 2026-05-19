"""Lemma D5: compositional nuisance on a coordinate block."""

from __future__ import annotations

import torch

from pmh._tensor import ensure_psd


def estimate_d5(
    features: torch.Tensor,
    nuisance_indices: torch.Tensor | list[int],
    *,
    shrinkage: float = 1e-6,
) -> torch.Tensor:
    """Covariance of features restricted to nuisance coordinates."""
    x = features.float()
    if x.dim() != 2:
        raise ValueError("features must be [N, d]")
    idx = torch.as_tensor(nuisance_indices, dtype=torch.long, device=x.device)
    block = x[:, idx]
    block = block - block.mean(0, keepdim=True)
    n = block.shape[0]
    d_full = x.shape[1]
    cov_block = (block.T @ block) / max(n, 1)
    sigma = torch.zeros(d_full, d_full, device=x.device, dtype=torch.float32)
    sigma[idx[:, None], idx] = cov_block
    return ensure_psd(sigma, shrinkage=shrinkage)
