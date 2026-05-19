"""GridSearchCV + Pipeline helpers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pmh import (
    default_pmh_param_grid,
    grid_search_pmh_pipeline,
    make_pmh_pipeline,
    tune_result_from_grid_search,
)
from pmh.matcher import PMHMatcher


def test_make_pmh_pipeline_clone_preserves_target():
    from sklearn.base import clone

    rng = np.random.default_rng(0)
    xs = rng.standard_normal((40, 6)).astype(np.float32)
    xt = xs + 0.1
    pipe = make_pmh_pipeline(xt, nuisance="domain_shift", rank=4)
    pipe2 = clone(pipe)
    np.testing.assert_array_equal(
        pipe2.named_steps["pmh"].X_target,
        xt,
    )


def test_default_param_grid_keys():
    grid = default_pmh_param_grid(rank_grid=(4, 8), clf_C_grid=(0.1, 1.0))
    assert grid["pmh__rank"] == [4, 8]
    assert grid["clf__C"] == [0.1, 1.0]


def test_grid_search_pmh_pipeline():
    rng = np.random.default_rng(1)
    n, d = 120, 10
    xs = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 2, n)
    xt = xs + 0.25

    result = grid_search_pmh_pipeline(
        xs,
        y,
        xt,
        nuisance="domain_shift",
        param_grid={"pmh__rank": [4, 8]},
        cv=3,
    )
    assert "pmh__rank" in result.best_params
    assert result.best_score >= 0.0
    assert len(result.all_results) == 2


def test_grid_search_return_search_object():
    rng = np.random.default_rng(2)
    xs = rng.standard_normal((80, 8)).astype(np.float32)
    y = rng.integers(0, 2, 80)
    xt = xs + 0.1

    search = grid_search_pmh_pipeline(
        xs,
        y,
        xt,
        param_grid={"pmh__rank": [3, 5]},
        cv=2,
        return_search=True,
    )
    assert hasattr(search, "best_estimator_")
    pred = search.predict(xs[:4])
    assert pred.shape == (4,)


def test_tune_sklearn_matcher_gridsearchcv_flag():
    from pmh.tune import tune_sklearn_matcher

    rng = np.random.default_rng(3)
    xs = rng.standard_normal((90, 8)).astype(np.float32)
    y = rng.integers(0, 2, 90)
    xt = xs + 0.2
    yt = y.copy()

    out = tune_sklearn_matcher(
        xs,
        y,
        xt,
        yt,
        scorer=None,
        nuisance="domain_shift",
        rank_grid=(4, 8),
        n_folds=2,
        use_gridsearchcv=True,
    )
    assert "pmh__rank" in out.best_params


def test_subspace_pipeline_with_labels():
    rng = np.random.default_rng(4)
    n, d = 100, 12
    xs = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 3, n)
    xt = xs + 0.15
    yt = y.copy()

    pipe = make_pmh_pipeline(xt, yt, nuisance="subspace", rank=4)
    pipe.fit(xs, y)
    assert isinstance(pipe.named_steps["pmh"], PMHMatcher)
    assert pipe.named_steps["pmh"].artifact_.method == "D1"
