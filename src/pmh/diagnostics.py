"""Diagnostics: eigengap pre-flight (Lemma D1)."""

from __future__ import annotations

import torch


def eigengap_ratio(cov: torch.Tensor, rank: int) -> float:
    """gamma_r = lambda_r / lambda_{r+1} on symmetrised sample covariance."""
    if rank < 1:
        raise ValueError("rank must be >= 1")
    c = 0.5 * (cov + cov.T)
    evals = torch.linalg.eigvalsh(c.float())
    evals = torch.sort(evals, descending=True).values
    if rank >= len(evals):
        return float("inf")
    lam_r = evals[rank - 1].item()
    lam_rp1 = evals[rank].item()
    if lam_rp1 <= 1e-12:
        return float("inf")
    return lam_r / lam_rp1
