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
    from sklearn.utils.validation import check_array, check_is_fitted, validate_data

    _SklearnBase = BaseEstimator
    _SklearnMixin = TransformerMixin
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

    class _SklearnBase:  # type: ignore[no-redef]
        pass

    class _SklearnMixin:  # type: ignore[no-redef]
        pass

    def check_array(X, *args, **kwargs):  # type: ignore[misc]
        return np.asarray(X)

    def check_is_fitted(estimator, attributes=None):  # type: ignore[misc]
        if not hasattr(estimator, "artifact_"):
            raise RuntimeError("Call fit() before transform().")

    def validate_data(  # type: ignore[misc]
        self,
        X,
        y=None,
        *,
        reset=True,
        validate_separately=False,
        skip_check_array=False,
        **check_params,
    ):
        X = np.asarray(X, dtype=np.float32)
        if reset:
            self.n_features_in_ = X.shape[1]
        return X, y


class PMHMatcher(_SklearnMixin, _SklearnBase):
    """Estimate **deployment shift** geometry and optionally project features.

    The ``nuisance=`` argument is the **shift type** (e.g. ``domain_shift`` = site A vs B,
    same labels). See :func:`format_shift_types` or docs/WHAT_IS_DEPLOYMENT_SHIFT.md — not
    “nuisance” in the everyday sense.

    sklearn contract
    ----------------
    * ``fit(X, y=None)`` — standard entry point; for D4 pass ``X_target`` via
      ``__init__``, ``fit(..., X_target=...)``, or metadata routing
      ``pipe.fit(X, y, pmh__X_target=xt)`` after ``set_fit_request(X_target=True)``.
    * ``transform(X)`` — project onto complement of estimated nuisance subspace.
    * ``get_params`` / ``set_params`` / ``clone`` — compatible with
      :class:`~sklearn.pipeline.Pipeline` and :class:`~sklearn.model_selection.GridSearchCV`.

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
    X_target, y_target : array, optional
        Target-domain data stored at construction (enables ``Pipeline.fit(X, y)``
        without metadata routing).
    has_source_labels, has_target_labels, has_target_domain, ...
        Flags for ``nuisance="auto"``.

    Attributes
    ----------
    artifact_ : SigmaTaskEstimate
        Fitted nuisance estimate (use with :class:`PMHLoss` in PyTorch).
    w_ : ndarray
        D1 subspace basis ``[d, r]`` when applicable.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from pmh import PMHMatcher
    >>> rng = np.random.default_rng(0)
    >>> xs = rng.standard_normal((100, 20), dtype=np.float32)
    >>> xt = xs + 0.3
    >>> m = PMHMatcher(nuisance="domain_shift", rank=8).fit(xs, X_target=xt)
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
        X_target: np.ndarray | None = None,
        y_target: np.ndarray | None = None,
        has_source_labels: bool = True,
        has_target_labels: bool = False,
        has_target_domain: bool = True,
        has_augmentation_modes: bool = False,
        has_style_pairs: bool = False,
    ) -> None:
        if isinstance(nuisance, str):
            self.nuisance = resolve_nuisance_arg(
                nuisance,
                has_source_labels=has_source_labels,
                has_target_labels=has_target_labels,
                has_target_domain=has_target_domain,
                has_augmentation_modes=has_augmentation_modes,
                has_style_pairs=has_style_pairs,
            )
        else:
            self.nuisance = nuisance
        self.rank = rank
        self.shrinkage = shrinkage
        self.dim = dim
        self.noise_level = noise_level
        self.nuisance_indices = nuisance_indices
        self.seed = seed
        self.n_pairs_per_class = n_pairs_per_class
        self.X_target = X_target
        self.y_target = y_target
        self.has_source_labels = has_source_labels
        self.has_target_labels = has_target_labels
        self.has_target_domain = has_target_domain
        self.has_augmentation_modes = has_augmentation_modes
        self.has_style_pairs = has_style_pairs

    def _resolved_method(self) -> str:
        if not isinstance(self.nuisance, str):
            raise ValueError(f"nuisance must be a string, got {type(self.nuisance).__name__}")
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

    @staticmethod
    def _coerce_target_features(y: Any, n_features: int) -> np.ndarray | None:
        """``fit(xs, xt)`` shorthand: second argument is unlabeled target features."""
        if y is None:
            return None
        arr = np.asarray(y)
        if arr.ndim == 2 and arr.shape[1] == n_features:
            return arr.astype(np.float32, copy=False)
        return None

    def _validate_X(self, X: np.ndarray, *, allow_3d: bool = False) -> np.ndarray:
        if _HAS_SKLEARN and not allow_3d:
            return check_array(
                X,
                dtype=[np.float64, np.float32],
                ensure_2d=True,
                accept_sparse=False,
            )
        x = np.asarray(X, dtype=np.float32)
        if not allow_3d and x.ndim != 2:
            raise ValueError("X must be 2D [n_samples, n_features] (D6: [N, T, d])")
        return x

    def _resolve_fit_domains(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        *,
        X_target: np.ndarray | None,
        y_target: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        method = self._resolved_method()
        allow_3d = method == "D6"
        x_src = self._validate_X(X, allow_3d=allow_3d)

        xt = X_target if X_target is not None else self.X_target
        yt = y_target if y_target is not None else self.y_target
        ys = y

        # fit(xs, xt) positional shorthand for D4
        if xt is None:
            coerced = self._coerce_target_features(y, x_src.shape[1])
            if coerced is not None and method == "D4":
                xt = coerced
                ys = None

        if xt is not None:
            xt = np.asarray(xt, dtype=np.float32)
        if ys is not None:
            ys = np.asarray(ys)
        if yt is not None:
            yt = np.asarray(yt)

        return x_src, ys, xt, yt

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *args: np.ndarray,
        X_target: np.ndarray | None = None,
        y_target: np.ndarray | None = None,
        aug_deltas: np.ndarray | None = None,
    ) -> PMHMatcher:
        """Estimate ``Sigma_task`` from source (and optional target) feature matrices.

        Standard sklearn: ``fit(X, y=None)`` with ``X_target`` / ``y_target`` in
        ``__init__``, as keyword arguments, or via metadata routing.

        Legacy positional: ``fit(X, y, X_target)`` or ``fit(X, y, X_target, y_target)``;
        ``fit(X, X_target)`` when the second argument is a 2D feature matrix (D4).
        """
        if len(args) == 1 and X_target is None:
            X_target = args[0]
        elif len(args) == 2:
            if X_target is None:
                X_target = args[0]
            if y_target is None:
                y_target = args[1]
        elif len(args) > 2:
            raise TypeError(
                "PMHMatcher.fit accepts at most two extra positional arguments "
                "(X_target, y_target). Use keyword arguments."
            )

        x_src, y_source, x_tgt, y_tgt = self._resolve_fit_domains(
            X, y, X_target=X_target, y_target=y_target
        )
        method = self._resolved_method()
        if method != "D6" and _HAS_SKLEARN:
            validated = validate_data(
                self,
                x_src,
                y=y_source,
                reset=True,
                accept_sparse=False,
                dtype=[np.float64, np.float32],
            )
            if y_source is not None:
                x_src, y_source = validated
            else:
                x_src = validated
        elif method != "D6":
            self.n_features_in_ = int(x_src.shape[1])

        if method == "D2":
            # Always match feature dimension of X (sklearn check_estimator varies d).
            dim = int(x_src.shape[1]) if x_src.ndim == 2 else int(self.dim or 10)
            cfg = self._build_config(dim=dim)
            self.artifact_ = estimate_sigma_task_numpy(config=cfg)
            self._transform_rank = min(self.rank or 8, dim)
            return self

        if method == "D4":
            if x_tgt is None:
                raise ValueError(
                    "domain_shift (D4) requires target-domain features. Pass "
                    "X_target= in fit(), set PMHMatcher(X_target=...) in __init__, "
                    "or use metadata routing: "
                    "matcher.set_fit_request(X_target=True); "
                    "pipe.fit(X, y, pmh__X_target=xt)."
                )
            cfg = self._build_config(dim=x_src.shape[1], n_samples=len(x_src) + len(x_tgt))
            self.artifact_ = estimate_sigma_task_numpy(x_src, x_tgt, config=cfg)
            self._transform_rank = cfg.rank
            return self

        if method == "D1":
            if y_source is None or x_tgt is None or y_tgt is None:
                raise ValueError(
                    "subspace (D1) requires fit(X, y, X_target=xt, y_target=yt) "
                    "or y_target= in __init__ with labels on both domains."
                )
            cfg = self._build_config(dim=x_src.shape[1], n_samples=len(x_src))
            r = cfg.rank
            assert r is not None
            self.w_ = estimate_cross_domain_subspace_numpy(
                x_src,
                y_source,
                x_tgt,
                y_tgt,
                rank=r,
                seed=self.seed,
                n_pairs_per_class=self.n_pairs_per_class,
            )
            self.artifact_ = estimate_sigma_task_numpy(
                x_src, y_source, x_tgt, y_tgt, config=cfg
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
                raise ValueError("temporal (D6): X must be [N, T, d] residuals or sequences")
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
        check_is_fitted(self, "artifact_")
        method = self._resolved_method()
        allow_3d = method == "D6"
        if allow_3d:
            x = self._validate_X(X, allow_3d=True)
        elif _HAS_SKLEARN:
            x = validate_data(
                self,
                X,
                reset=False,
                accept_sparse=False,
                dtype=[np.float64, np.float32],
            )
        else:
            x = self._validate_X(X, allow_3d=False)
        if getattr(self, "w_", None) is not None:
            return project_onto_complement(x, self.w_)
        sigma = self.artifact_.sigma.detach().cpu().numpy()
        r = self._transform_rank or self.rank or min(32, sigma.shape[0] // 4)
        return project_from_sigma(x, sigma, rank=int(r))

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *args: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.fit(X, y, *args, **kwargs).transform(X)

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
            "X_target": self.X_target,
            "y_target": self.y_target,
            "has_source_labels": self.has_source_labels,
            "has_target_labels": self.has_target_labels,
            "has_target_domain": self.has_target_domain,
            "has_augmentation_modes": self.has_augmentation_modes,
            "has_style_pairs": self.has_style_pairs,
        }

    def set_params(self, **params: Any) -> PMHMatcher:
        known = set(self.get_params())
        for key, value in params.items():
            if key not in known:
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
        return self
