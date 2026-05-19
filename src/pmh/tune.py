"""Lightweight hyperparameter search for PMH."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from pmh.config import PMHConfig
from pmh.matcher import PMHMatcher


@dataclass
class TuneResult:
    """Best settings from a small grid search."""

    best_params: dict[str, Any]
    best_score: float
    all_results: list[dict[str, Any]]


def tune_sklearn_matcher(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    *,
    scorer: Callable[[np.ndarray, np.ndarray], float],
    nuisance: str = "subspace",
    rank_grid: Iterable[int] = (4, 8, 16, 32),
    n_folds: int = 3,
    seed: int = 0,
) -> TuneResult:
    """Grid search ``rank`` for :class:`PMHMatcher` + downstream ``scorer(x_proj, y)``.

    ``scorer`` should return a metric to **maximize** (e.g. validation accuracy).
    Uses simple holdout splits on source for speed.
    """
    from sklearn.model_selection import KFold

    ranks = list(rank_grid)
    results: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_params: dict[str, Any] = {"rank": ranks[0]}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for rank in ranks:
        fold_scores: list[float] = []
        for tr_idx, va_idx in kf.split(x_source):
            m = PMHMatcher(nuisance=nuisance, rank=rank, seed=seed)
            m.fit(
                x_source[tr_idx],
                y_source[tr_idx],
                x_target,
                y_target,
            )
            x_va = m.transform(x_source[va_idx])
            fold_scores.append(float(scorer(x_va, y_source[va_idx])))
        mean_score = float(np.mean(fold_scores))
        results.append({"rank": rank, "score": mean_score})
        if mean_score > best_score:
            best_score = mean_score
            best_params = {"rank": rank, "nuisance": nuisance}

    return TuneResult(best_params=best_params, best_score=best_score, all_results=results)


def tune_pmh_config(
    task_loss_fn: Callable[[PMHConfig], float],
    *,
    weight_grid: Iterable[float] = (0.1, 0.3, 0.5),
    cap_ratio_grid: Iterable[float] = (0.2, 0.3, 0.5),
    warmup_grid: Iterable[int] = (0, 2),
) -> TuneResult:
    """Grid search :class:`PMHConfig` scalars via user ``task_loss_fn(config) -> loss``.

    ``task_loss_fn`` should run a short training snippet and return a scalar to **minimize**
    (e.g. validation loss). Sigma is assumed fixed.
    """
    results: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params: dict[str, Any] = {}

    for w in weight_grid:
        for cap in cap_ratio_grid:
            for warm in warmup_grid:
                cfg = PMHConfig(weight=w, cap_ratio=cap, warmup_epochs=warm)
                score = float(task_loss_fn(cfg))
                row = {"weight": w, "cap_ratio": cap, "warmup_epochs": warm, "score": score}
                results.append(row)
                if score < best_score:
                    best_score = score
                    best_params = dict(row)

    return TuneResult(best_params=best_params, best_score=best_score, all_results=results)
