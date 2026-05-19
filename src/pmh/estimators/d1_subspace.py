"""Lemma D1: low-rank subspace W W^T from class-aligned cross-domain differences."""

from __future__ import annotations

import numpy as np
import torch

from pmh._tensor import ensure_psd, flatten_features
from pmh.estimators.d4_domain import gram_from_diff


def estimate_d1_gram_unlabeled(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    rank: int,
    shrinkage: float = 1e-6,
) -> torch.Tensor:
    """Unlabeled top-r eigenspace of cross-domain Gram (D4-like; not paper D1).

    Use :func:`estimate_d1` with class labels for Lemma D1 / T1 protocol.
    """
    if rank < 1:
        raise ValueError("rank must be >= 1")
    cov = gram_from_diff(source, target)
    evals, evecs = torch.linalg.eigh(cov.float())
    r = min(rank, cov.shape[0])
    top_e = evecs[:, -r:]
    top_l = evals[-r:].clamp(min=0.0)
    sigma = top_e @ torch.diag(top_l) @ top_e.T
    return ensure_psd(sigma, shrinkage=shrinkage)


# Back-compat alias (deprecated for labeled D1 workflows)
estimate_d1_from_gram = estimate_d1_gram_unlabeled


def estimate_d1(
    x_src: torch.Tensor,
    y_src: torch.Tensor,
    x_tgt: torch.Tensor,
    y_tgt: torch.Tensor,
    *,
    rank: int,
    shrinkage: float = 1e-6,
    seed: int = 0,
    n_pairs_per_class: int = 100,
    include_mean_shift: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper D1: class-aligned cross-domain SVD → :math:`\\Sigma = W W^T`.

    Returns
    -------
    sigma
        PSD matrix [d, d].
    w
        Subspace basis [d, r] (columns span nuisance directions).
    """
    from pmh.numpy_api import estimate_cross_domain_subspace_numpy

    if rank < 1:
        raise ValueError("rank must be >= 1")
    xs = np.asarray(x_src.detach().cpu().numpy(), dtype=np.float32)
    ys = np.asarray(y_src.detach().cpu().numpy()).reshape(-1)
    xt = np.asarray(x_tgt.detach().cpu().numpy(), dtype=np.float32)
    yt = np.asarray(y_tgt.detach().cpu().numpy()).reshape(-1)
    w = estimate_cross_domain_subspace_numpy(
        xs,
        ys,
        xt,
        yt,
        rank=rank,
        seed=seed,
        n_pairs_per_class=n_pairs_per_class,
        include_mean_shift=include_mean_shift,
    )
    sigma_np = (w @ w.T).astype(np.float32)
    sigma = ensure_psd(torch.from_numpy(sigma_np), shrinkage=shrinkage)
    w_t = torch.from_numpy(w.astype(np.float32))
    return sigma, w_t


def cross_domain_svd_W(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    """Return W in R^{d x r} from unlabeled pooled diff SVD (legacy helper)."""
    s = flatten_features(source)
    t = flatten_features(target)
    diff = s - t
    diff = diff - diff.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(diff, full_matrices=False)
    r = min(rank, vh.shape[0])
    return vh[:r].T.contiguous()
