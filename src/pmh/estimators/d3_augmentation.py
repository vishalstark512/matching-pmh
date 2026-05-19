"""Lemma D3: finite augmentation modes beta_k."""

from __future__ import annotations

import torch

from pmh._tensor import ensure_psd


def estimate_d3(
    aug_deltas: torch.Tensor,
    *,
    shrinkage: float = 1e-6,
) -> torch.Tensor:
    """Sigma = (1/K) sum_k E[beta_k beta_k^T].

    Parameters
    ----------
    aug_deltas : Tensor [K, N, d] or [K, d]
        Per-mode mean delta vectors (e.g. Jacobian of aug in feature space).
    """
    if aug_deltas.dim() == 2:
        # [K, d] — one mean direction per mode
        modes = aug_deltas.float()
        k = modes.shape[0]
        sigma = (modes.T @ modes) / max(k, 1)
    elif aug_deltas.dim() == 3:
        k, n, d = aug_deltas.shape
        acc = torch.zeros(d, d, device=aug_deltas.device, dtype=torch.float32)
        for i in range(k):
            b = aug_deltas[i].float()
            b = b - b.mean(0, keepdim=True)
            acc = acc + (b.T @ b) / max(n, 1)
        sigma = acc / max(k, 1)
    else:
        raise ValueError("aug_deltas must be [K, d] or [K, N, d]")
    return ensure_psd(sigma, shrinkage=shrinkage)
