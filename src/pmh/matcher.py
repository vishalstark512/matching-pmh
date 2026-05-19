"""Sklearn-style PMHMatcher: fit Sigma_task, transform features (numpy path)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pmh.artifact import SigmaTaskEstimate
from pmh.config import SigmaTaskConfig
from pmh.numpy_api import estimate_cross_domain_subspace_numpy, estimate_sigma_task_numpy
from pmh.nuisance import config_from_nuisance, default_rank, resolve_method
from pmh.suggest import resolve_nuisance_arg
from pmh.sklearn_match import project_from_sigma, project_onto_complement

try:
    from sklearn.base import BaseEstimator, TransformerMixin

    _SklearnBase = BaseEstimator
    _SklearnMixin = TransformerMixin
except ImportError:

    class _SklearnBase:  # type: ignore[no-redef]
        pass

    class _SklearnMixin:  # type: ignore[no-redef]
        pass


class PMHMatcher(_SklearnBase, _SklearnMixin):
    """Estimate deployment nuisance geometry and optionally project features.

    Parameters
    ----------
    nuisance : str
        Human-readable name (e.g. ``"domain_shift"`` → D4) or ``"D1"``–``"D7"``.
    rank : int, optional
        Subspace rank for D1/D4/D7. Default: ``min(32, d//4)`` from data.
    shrinkage : float
        PSD regularization on ``Sigma_task``.
    dim, noise_level : float
        For D2 isotropic noise.
    nuisance_indices : list[int]
        For D5 compositional blocks.
    seed : int
        D1 class-pair sampling seed.

    Attributes
    ----------
    artifact_ : SigmaTaskEstimate
        Fitted nuisance estimate (use with :class:`PMHLoss` in PyTorch).
    w_ : ndarray or None
        D1 subspace basis ``[d, r]`` when applicable.

    Examples
    --------
    >>> import numpy as np
    >>> from pmh import PMHMatcher
    >>> rng = np.random.default_rng(0)
    >>> xs = rng.standard_normal((100, 20), dtype=np.float32)
    >>> xt = xs + 0.3
    >>> m = PMHMatcher(nuisance="domain_shift", rank=8).fit(xs, xt)
    >>> m.transform(xs).shape
    (100, 20)
    """

    def __init__(
        self,
        nuisance: str = "domain_shift",
        *,
        rank: int | None = None,
        shrinkage: float = 1e-6,
        dim: int | None = None,
        noise_level: float = 0.1,
        nuisance_indices: list[int] | None = None,
        seed: int = 0,
        n_pairs_per_class: int = 100,
        has_source_labels: bool = True,
        has_target_labels: bool = False,
        has_target_domain: bool = True,
        has_augmentation_modes: bool = False,
        has_style_pairs: bool = False,
    ) -> None:
        self.nuisance = resolve_nuisance_arg(
            nuisance,
            has_source_labels=has_source_labels,
            has_target_labels=has_target_labels,
            has_target_domain=has_target_domain,
            has_augmentation_modes=has_augmentation_modes,
            has_style_pairs=has_style_pairs,
        )
        self.rank = rank
        self.shrinkage = shrinkage
        self.dim = dim
        self.noise_level = noise_level
        self.nuisance_indices = nuisance_indices
        self.seed = seed
        self.n_pairs_per_class = n_pairs_per_class
        self.artifact_: SigmaTaskEstimate | None = None
        self.w_: np.ndarray | None = None
        self._transform_rank: int | None = None

    def _resolved_method(self) -> str:
        return resolve_method(self.nuisance)

    def _build_config(self, *, dim: int | None = None, n_samples: int = 0) -> SigmaTaskConfig:
        method = self._resolved_method()
        rank = self.rank
        if method in ("D1", "D4", "D7") and dim is not None:
            rank = default_rank(dim=dim, n_samples=n_samples, requested=self.rank)
        return config_from_nuisance(
            self.nuisance,
            rank=rank,
            shrinkage=self.shrinkage,
            dim=dim if dim is not None else self.dim,
            noise_level=self.noise_level,
            nuisance_indices=self.nuisance_indices,
        )

    def fit(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray | None = None,
        X_target: np.ndarray | None = None,
        y_target: np.ndarray | None = None,
        *,
        aug_deltas: np.ndarray | None = None,
    ) -> PMHMatcher:
        """Estimate ``Sigma_task`` from source/target feature matrices.

        D4 (domain shift): ``fit(X_source, X_target=...)`` or ``fit(X_source, None, X_target)``.
        D1 (subspace): labels required on both domains.
        D2 (isotropic): ``fit(X_source)`` infers ``dim`` from columns; or set ``dim=`` in ``__init__``.
        D5: pass compositional features as ``X_source`` with ``nuisance_indices``.
        D3: ``aug_deltas=`` with shape ``[K, d]`` or ``[K, N, d]``.
        D6: ``X_source`` with shape ``[N, T, d]``.
        """
        method = self._resolved_method()
        x_src = np.asarray(X_source, dtype=np.float32)
        if method != "D6" and x_src.ndim != 2:
            raise ValueError("X_source must be 2D [n_samples, n_features] (D6: [N, T, d])")

        # fit(X_source, X_target) shorthand for D4
        if (
            X_target is None
            and y_source is not None
            and method == "D4"
            and np.asarray(y_source).ndim == 2
            and np.asarray(y_source).shape[1] == x_src.shape[1]
        ):
            X_target = y_source
            y_source = None

        if method == "D2":
            dim = self.dim or x_src.shape[1]
            cfg = self._build_config(dim=dim)
            self.artifact_ = estimate_sigma_task_numpy(config=cfg)
            self._transform_rank = min(self.rank or 8, dim)
            return self

        if method == "D4":
            if X_target is None:
                raise ValueError(
                    "domain_shift (D4) requires X_target. "
                    "Call fit(X_source, X_target=xt) or fit(X_source, None, X_target)."
                )
            x_tgt = np.asarray(X_target, dtype=np.float32)
            cfg = self._build_config(dim=x_src.shape[1], n_samples=len(x_src) + len(x_tgt))
            self.artifact_ = estimate_sigma_task_numpy(x_src, x_tgt, config=cfg)
            self._transform_rank = cfg.rank
            return self

        if method == "D1":
            if y_source is None or X_target is None or y_target is None:
                raise ValueError(
                    "subspace (D1) requires fit(X_source, y_source, X_target, y_target)"
                )
            y_s = np.asarray(y_source)
            y_t = np.asarray(y_target)
            x_tgt = np.asarray(X_target, dtype=np.float32)
            cfg = self._build_config(dim=x_src.shape[1], n_samples=len(x_src))
            r = cfg.rank
            assert r is not None
            self.w_ = estimate_cross_domain_subspace_numpy(
                x_src,
                y_s,
                x_tgt,
                y_t,
                rank=r,
                seed=self.seed,
                n_pairs_per_class=self.n_pairs_per_class,
            )
            self.artifact_ = estimate_sigma_task_numpy(
                x_src, y_s, x_tgt, y_t, config=cfg
            )
            self._transform_rank = r
            return self

        if method == "D5":
            if self.nuisance_indices is None:
                raise ValueError("compositional (D5) requires nuisance_indices= in PMHMatcher")
            cfg = self._build_config(dim=x_src.shape[1])
            self.artifact_ = estimate_sigma_task_numpy(x_src, config=cfg)
            self._transform_rank = len(self.nuisance_indices)
            return self

        if method == "D3":
            if aug_deltas is None:
                raise ValueError(
                    "D3: pass aug_deltas= (shape [K, d] or [K, N, d]) to fit(), "
                    "or use PMHTrainer with augmentations=."
                )
            import torch
            from pmh.estimate import estimate_from_config

            deltas = torch.from_numpy(np.asarray(aug_deltas, dtype=np.float32))
            cfg = SigmaTaskConfig.for_augmentation(shrinkage=self.shrinkage)
            self.artifact_ = estimate_from_config(cfg, aug_deltas=deltas)
            self._transform_rank = deltas.shape[0]
            return self

        if method == "D6":
            if x_src.ndim != 3:
                raise ValueError("temporal (D6): X_source must be [N, T, d] residuals or sequences")
            import torch
            from pmh.estimate import estimate_from_config

            cfg = self._build_config(dim=x_src.shape[2], n_samples=x_src.shape[0])
            self.artifact_ = estimate_from_config(cfg, torch.from_numpy(x_src))
            self._transform_rank = cfg.rank
            return self

        raise ValueError(
            f"PMHMatcher numpy fit supports D1–D6; got {method}. "
            "For D7 use HFPMHTrainer or estimate_style_sigma."
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project onto complement of estimated nuisance subspace (optional preprocessing)."""
        if self.artifact_ is None:
            raise RuntimeError("Call fit() before transform().")
        x = np.asarray(X, dtype=np.float32)
        if self.w_ is not None:
            return project_onto_complement(x, self.w_)
        sigma = self.artifact_.sigma.detach().cpu().numpy()
        r = self._transform_rank or self.rank or min(32, sigma.shape[0] // 4)
        return project_from_sigma(x, sigma, rank=int(r))

    def fit_transform(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray | None = None,
        X_target: np.ndarray | None = None,
        y_target: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        self.fit(X_source, y_source, X_target, y_target, **kwargs)
        return self.transform(X_source)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "nuisance": self.nuisance,
            "rank": self.rank,
            "shrinkage": self.shrinkage,
            "dim": self.dim,
            "noise_level": self.noise_level,
            "nuisance_indices": self.nuisance_indices,
            "seed": self.seed,
            "n_pairs_per_class": self.n_pairs_per_class,
        }

    def set_params(self, **params: Any) -> PMHMatcher:
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
        return self
