"""Scikit-learn helpers: project onto matched nuisance complement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pmh.artifact import SigmaTaskEstimate
from pmh.numpy_api import (
    estimate_cross_domain_subspace_numpy,
    estimate_sigma_task_numpy,
    gram_from_diff_numpy,
)
from pmh.config import SigmaTaskConfig


def wrong_w_subspace_numpy(
    w_matched: np.ndarray,
    rank: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Random rank-`rank` subspace orthogonalized against matched ``W`` (T1 / Lemma C)."""
    d = int(w_matched.shape[0])
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((d, rank)).astype(np.float32)
    residual = m - w_matched @ (w_matched.T @ m)
    q, _ = np.linalg.qr(residual)
    r = min(rank, q.shape[1])
    return q[:, :r].astype(np.float32)


def domain_d4_subspace_numpy(
    x_src: np.ndarray,
    x_tgt: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Top-`rank` directions of D4 domain Gram (unmatched nuisance control)."""
    sigma = gram_from_diff_numpy(x_src, x_tgt)
    evals, evecs = np.linalg.eigh(sigma)
    r = min(rank, sigma.shape[0])
    return evecs[:, -r:].astype(np.float32)


def project_onto_complement(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """P_{W^perp} x = x - (x W) W^T for W in R^{d x r}."""
    return x - (x @ w) @ w.T


def project_from_sigma(x: np.ndarray, sigma: np.ndarray, rank: int) -> np.ndarray:
    """Use top-`rank` eigenvectors of Sigma as nuisance subspace, then project out."""
    evals, evecs = np.linalg.eigh(sigma)
    r = min(rank, sigma.shape[0])
    w = evecs[:, -r:]
    return project_onto_complement(x, w)


@dataclass
class MatchedSubspaceProjector:
    """Fit class-aligned D1 subspace; transform features for classical linear models.

    Example
    -------
    >>> proj = MatchedSubspaceProjector(rank=16)
    >>> proj.fit(x_amazon, y_a, x_dslr, y_d)
    >>> x_train_m = proj.transform(x_amazon)
    >>> clf.fit(x_train_m, y_a)
    """

    rank: int = 16
    n_pairs_per_class: int = 100
    seed: int = 0
    w_: np.ndarray | None = None
    artifact_: SigmaTaskEstimate | None = None

    def fit(
        self,
        x_src: np.ndarray,
        y_src: np.ndarray,
        x_tgt: np.ndarray,
        y_tgt: np.ndarray,
    ) -> MatchedSubspaceProjector:
        self.w_ = estimate_cross_domain_subspace_numpy(
            x_src,
            y_src,
            x_tgt,
            y_tgt,
            rank=self.rank,
            seed=self.seed,
            n_pairs_per_class=self.n_pairs_per_class,
            include_mean_shift=True,
        )
        self.artifact_ = estimate_sigma_task_numpy(
            x_src, y_src, x_tgt, y_tgt,
            config=SigmaTaskConfig.for_subspace(rank=self.rank),
        )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("call fit() first")
        return project_onto_complement(np.asarray(x, dtype=np.float32), self.w_)

    def fit_transform(
        self,
        x_src: np.ndarray,
        y_src: np.ndarray,
        x_tgt: np.ndarray,
        y_tgt: np.ndarray,
    ) -> np.ndarray:
        self.fit(x_src, y_src, x_tgt, y_tgt)
        return self.transform(x_src)
