"""Lemma D1: low-rank subspace W W^T from cross-domain differences."""

from __future__ import annotations

import torch

from pmh._tensor import ensure_psd, flatten_features
from pmh.estimators.d4_domain import gram_from_diff


def estimate_d1(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    rank: int,
    shrinkage: float = 1e-6,
) -> torch.Tensor:
    """Top-r eigenspace of cross-domain covariance (subspace nuisance)."""
    if rank < 1:
        raise ValueError("rank must be >= 1")
    cov = gram_from_diff(source, target)
    evals, evecs = torch.linalg.eigh(cov.float())
    r = min(rank, cov.shape[0])
    top_e = evecs[:, -r:]
    top_l = evals[-r:].clamp(min=0.0)
    sigma = top_e @ torch.diag(top_l) @ top_e.T
    return ensure_psd(sigma, shrinkage=shrinkage)


def cross_domain_svd_W(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    """Return W in R^{d x r} with orthonormal columns (for explicit subspace PMH)."""
    s = flatten_features(source)
    t = flatten_features(target)
    diff = s - t
    diff = diff - diff.mean(0, keepdim=True)
    # thin SVD on [N, d]
    _, _, vh = torch.linalg.svd(diff, full_matrices=False)
    r = min(rank, vh.shape[0])
    return vh[:r].T.contiguous()
