"""Shared tensor utilities."""

from __future__ import annotations

import torch


def as_tensor(
    x: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.to(dtype=dtype)
    if device is not None:
        x = x.to(device)
    return x


def ensure_psd(sigma: torch.Tensor, shrinkage: float = 1e-6) -> torch.Tensor:
    """Symmetrise and add shrinkage so Cholesky is stable."""
    d = sigma.shape[-1]
    eye = torch.eye(d, device=sigma.device, dtype=sigma.dtype)
    sym = 0.5 * (sigma + sigma.T)
    return sym + shrinkage * eye


def flatten_features(feats: torch.Tensor) -> torch.Tensor:
    """[B, C] or [B, C, H, W] -> [N, C] centred-ready rows."""
    if feats.dim() == 4:
        b, c, h, w = feats.shape
        return feats.float().permute(0, 2, 3, 1).reshape(b * h * w, c)
    if feats.dim() == 2:
        return feats.float()
    raise ValueError(f"Expected feature tensor with 2 or 4 dims, got shape {tuple(feats.shape)}")
