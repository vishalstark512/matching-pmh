"""sklearn Pipeline + GridSearchCV helpers for :class:`PMHMatcher`."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from pmh.matcher import PMHMatcher
from pmh.tune import TuneResult

try:
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _require_sklearn() -> None:
    if not _HAS_SKLEARN:
        raise ImportError(
            'sklearn integration requires scikit-learn. '
            'Install with: pip install "matching-pmh[sklearn]"'
        )


def make_pmh_pipeline(
    x_target: np.ndarray,
    y_target: np.ndarray | None = None,
    *,
    nuisance: str = "domain_shift",
    rank: int | None = None,
    classifier: Any | None = None,
    pmh_kwargs: Mapping[str, Any] | None = None,
    clf_kwargs: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Build ``Pipeline([PMHMatcher, classifier])`` with target domain fixed at construction.

    Target features (and optional labels for D1) live on the matcher so
    :func:`~sklearn.model_selection.GridSearchCV` and :class:`~sklearn.pipeline.Pipeline`
    can call ``fit(X_source, y_source)`` without extra routing.

    Parameters
    ----------
    x_target
        Unlabeled target-domain features ``[n_tgt, d]`` (D4) or paired with ``y_target`` (D1).
    y_target
        Target labels when using ``nuisance="subspace"`` (D1).
    nuisance, rank
        Passed to :class:`PMHMatcher`.
    classifier
        Final estimator (default ``LogisticRegression(max_iter=500)``).
    pmh_kwargs, clf_kwargs
        Extra keyword arguments for the matcher / classifier constructors.

    Examples
    --------
    >>> from pmh.sklearn_pipeline import make_pmh_pipeline, default_pmh_param_grid
    >>> from sklearn.model_selection import GridSearchCV
    >>> pipe = make_pmh_pipeline(x_target, nuisance="domain_shift", rank=8)
    >>> search = GridSearchCV(pipe, default_pmh_param_grid(rank_grid=(4, 8, 16)), cv=3)
    >>> search.fit(x_source, y_source)
    """
    _require_sklearn()

    extra_pmh = dict(pmh_kwargs or {})
    extra_clf = dict(clf_kwargs or {})
    matcher = PMHMatcher(
        nuisance=nuisance,
        rank=rank,
        X_target=np.asarray(x_target, dtype=np.float32),
        y_target=None if y_target is None else np.asarray(y_target),
        **extra_pmh,
    )
    if classifier is None:
        clf: Any = LogisticRegression(max_iter=500, **extra_clf)
    else:
        clf = clone(classifier) if hasattr(classifier, "get_params") else classifier

    return Pipeline([("pmh", matcher), ("clf", clf)])


def default_pmh_param_grid(
    *,
    rank_grid: Iterable[int] = (4, 8, 16, 32),
    shrinkage_grid: Iterable[float] | None = None,
    clf_C_grid: Iterable[float] | None = None,
) -> dict[str, list[Any]]:
    """Default ``param_grid`` keys for :func:`make_pmh_pipeline`."""
    grid: dict[str, list[Any]] = {"pmh__rank": list(rank_grid)}
    if shrinkage_grid is not None:
        grid["pmh__shrinkage"] = list(shrinkage_grid)
    if clf_C_grid is not None:
        grid["clf__C"] = list(clf_C_grid)
    return grid


def tune_result_from_grid_search(search: GridSearchCV) -> TuneResult:
    """Convert a fitted :class:`~sklearn.model_selection.GridSearchCV` to :class:`TuneResult`."""
    rows: list[dict[str, Any]] = []
    for params, score in zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
    ):
        rows.append({"params": dict(params), "score": float(score)})
    return TuneResult(
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        all_results=rows,
    )


def grid_search_pmh_pipeline(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray | None = None,
    *,
    nuisance: str = "domain_shift",
    param_grid: Mapping[str, Iterable[Any]] | None = None,
    pipeline: Pipeline | None = None,
    cv: int | str = 5,
    scoring: str | None = None,
    n_jobs: int | None = None,
    refit: bool = True,
    return_search: bool = False,
    **gridsearch_kwargs: Any,
) -> TuneResult | GridSearchCV:
    """Cross-validated grid search over ``pmh__rank`` (and optional classifier params).

    Fits :func:`make_pmh_pipeline` on source folds; ``X_target`` stays fixed on the
    matcher (standard domain-adaptation protocol).

    Parameters
    ----------
    return_search
        If ``True``, return the fitted :class:`~sklearn.model_selection.GridSearchCV`
        instead of :class:`TuneResult`.
    gridsearch_kwargs
        Forwarded to :class:`~sklearn.model_selection.GridSearchCV` (e.g. ``verbose``).

    Returns
    -------
    TuneResult or GridSearchCV
        Best hyperparameters and CV scores.
    """
    _require_sklearn()

    pipe = pipeline or make_pmh_pipeline(
        x_target,
        y_target=y_target,
        nuisance=nuisance,
    )
    grid = dict(param_grid or default_pmh_param_grid())
    search = GridSearchCV(
        pipe,
        grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        **gridsearch_kwargs,
    )
    search.fit(np.asarray(x_source), np.asarray(y_source))
    if return_search:
        return search
    return tune_result_from_grid_search(search)
