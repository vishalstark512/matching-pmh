"""tune and compare helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pmh import PMHConfig, compare_arms_sklearn
from pmh.suggest import suggest_nuisance
from pmh.tune import tune_sklearn_matcher


def test_pmh_config_presets():
    assert PMHConfig.conservative().weight < PMHConfig.aggressive().weight


def test_compare_arms_sklearn():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n, d = 100, 20
    xs = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 3, n)
    xt = xs + 0.5
    yt = y.copy()
    res = compare_arms_sklearn(xs, y, xt, yt, rank=5, include_coral=False)
    assert "b0" in res.arms and "matched" in res.arms


def test_tune_sklearn_matcher():
    pytest = __import__("pytest")
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(1)
    n, d = 80, 12
    xs = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 2, n)
    xt = xs + 0.3
    yt = y.copy()

    def scorer(x_proj, y_va):
        clf = LogisticRegression(max_iter=200)
        clf.fit(x_proj, y_va)
        return accuracy_score(y_va, clf.predict(x_proj))

    out = tune_sklearn_matcher(
        xs, y, xt, yt,
        scorer=scorer,
        nuisance="domain_shift",
        rank_grid=(4, 8),
        n_folds=2,
    )
    assert "rank" in out.best_params
    assert out.best_score >= 0
