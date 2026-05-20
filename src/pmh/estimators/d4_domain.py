"""Lemma D4: cross-domain Gram from source/target features."""

from __future__ import annotations

import torch

from pmh._tensor import ensure_psd, flatten_features


def gram_from_diff(feats_s: torch.Tensor, feats_t: torch.Tensor) -> torch.Tensor:
    """Sigma = (1/N) D^T D for D = centre(phi_s) - centre(phi_t)."""
    s = flatten_features(feats_s)
    t = flatten_features(feats_t)
    s = s - s.mean(0, keepdim=True)
    t = t - t.mean(0, keepdim=True)
    n = min(s.shape[0], t.shape[0])
    diff = s[:n] - t[:n]
    return (diff.T @ diff) / max(n, 1)


def gram_from_paired_diffs(diff: torch.Tensor) -> torch.Tensor:
    """Sigma from already paired rows ``diff[i] = phi_s[i] - phi_t[i]`` (class-aligned D4)."""
    d = flatten_features(diff)
    d = d - d.mean(0, keepdim=True)
    n = d.shape[0]
    return (d.T @ d) / max(n, 1)


def estimate_d4(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    rank: int | None = None,
    shrinkage: float = 1e-6,
) -> torch.Tensor:
    sigma = gram_from_diff(source, target)
    if rank is not None:
        sigma = _truncate_rank(sigma, rank)
    return ensure_psd(sigma, shrinkage=shrinkage)


def estimate_d4_from_paired_diffs(
    diff: torch.Tensor,
    *,
    rank: int | None = None,
    shrinkage: float = 1e-6,
) -> torch.Tensor:
    """D4 Gram from paired difference rows (class-aligned batches)."""
    sigma = gram_from_paired_diffs(diff)
    if rank is not None:
        sigma = _truncate_rank(sigma, rank)
    return ensure_psd(sigma, shrinkage=shrinkage)


def _truncate_rank(cov: torch.Tensor, rank: int) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(cov.float())
    evals = evals.clamp(min=0.0)
    r = min(rank, cov.shape[0])
    top = evecs[:, -r:]
    lam = evals[-r:]
    return top @ torch.diag(lam) @ top.T
