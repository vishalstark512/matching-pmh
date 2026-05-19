"""Lemma D6: temporal / sensor residual scatter."""

from __future__ import annotations

import torch

from pmh._tensor import ensure_psd


def estimate_d6(
    sequences: torch.Tensor,
    *,
    shrinkage: float = 1e-6,
) -> torch.Tensor:
    """Sigma from content residuals Delta h along sequences.

    Parameters
    ----------
    sequences : Tensor [N, T, d] or [N, d]
        If [N, T, d], uses consecutive differences h_{t+1} - h_t per sequence.
        If [N, d], treats rows as precomputed residuals.
    """
    if sequences.dim() == 2:
        r = sequences.float()
        r = r - r.mean(0, keepdim=True)
        n = r.shape[0]
        sigma = (r.T @ r) / max(n, 1)
    elif sequences.dim() == 3:
        diffs = sequences[:, 1:] - sequences[:, :-1]
        r = diffs.reshape(-1, diffs.shape[-1]).float()
        r = r - r.mean(0, keepdim=True)
        n = r.shape[0]
        sigma = (r.T @ r) / max(n, 1)
    else:
        raise ValueError("sequences must be [N, d] or [N, T, d]")
    return ensure_psd(sigma, shrinkage=shrinkage)
