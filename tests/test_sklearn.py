"""sklearn: PMHMatcher API, pipelines, grid search, tuning."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pmh import (
    PMHConfig,
    PMHMatcher,
    compare_arms_sklearn,
    default_pmh_param_grid,
    grid_search_pmh_pipeline,
    make_pmh_pipeline,
    tune_result_from_grid_search,
)
from pmh.suggest import suggest_nuisance
from pmh.tune import tune_sklearn_matcher


def test_get_params_keys():
    est = PMHMatcher(nuisance="domain_shift", rank=8)
    params = est.get_params()
    assert params["nuisance"] == "domain_shift"
    assert params["rank"] == 8
    assert "shrinkage" in params
    assert "X_target" in params


def test_set_params_roundtrip():
    est = PMHMatcher()
    est.set_params(nuisance="subspace", rank=4)
    assert est.get_params()["nuisance"] == "subspace"
    assert est.rank == 4


def test_sklearn_clone():
    from sklearn.base import clone

    est = PMHMatcher(nuisance="domain_shift", rank=8, shrinkage=1e-5)
    est2 = clone(est)
    assert est2.get_params() == est.get_params()
    assert est2 is not est
    assert not hasattr(est2, "artifact_")


def test_transform_after_fit_d4():
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((40, 12)).astype(np.float32)
    xt = xs + 0.2
    m = PMHMatcher(nuisance="domain_shift", rank=4)
    m.fit(xs, X_target=xt)
    z = m.transform(xs)
    assert z.shape == xs.shape
    assert m.n_features_in_ == 12


def test_fit_sklearn_signature_d4_init_target():
    rng = np.random.default_rng(2)
    xs = rng.standard_normal((50, 10)).astype(np.float32)
    xt = xs + 0.15
    m = PMHMatcher(nuisance="domain_shift", rank=4, X_target=xt)
    m.fit(xs)
    assert m.artifact_.method == "D4"


def test_pipeline_fit_with_init_target():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(1)
    xs = rng.standard_normal((60, 8)).astype(np.float32)
    y = rng.integers(0, 2, 60)
    xt = xs + 0.1

    pipe = Pipeline(
        [
            ("pmh", PMHMatcher(nuisance="domain_shift", rank=3, X_target=xt)),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipe.fit(xs, y)
    pred = pipe.predict(xs[:5])
    assert pred.shape == (5,)


def test_fit_request_metadata_direct():
    import sklearn

    sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if sklearn_version < (1, 4):
        pytest.skip("metadata routing requires sklearn>=1.4")

    sklearn.set_config(enable_metadata_routing=True)
    rng = np.random.default_rng(3)
    xs = rng.standard_normal((60, 8)).astype(np.float32)
    y = rng.integers(0, 2, 60)
    xt = xs + 0.1

    pmh = PMHMatcher(nuisance="domain_shift", rank=3)
    pmh.set_fit_request(X_target=True)
    pmh.fit(xs, y, X_target=xt)
    assert pmh.artifact_.method == "D4"
    sklearn.set_config(enable_metadata_routing=False)


def test_check_estimator_isotropic():
    from sklearn.utils.estimator_checks import check_estimator

    est = PMHMatcher(nuisance="isotropic", rank=2, noise_level=0.1)
    check_estimator(est)


def test_make_pmh_pipeline_clone_preserves_target():
    from sklearn.base import clone

    rng = np.random.default_rng(0)
    xs = rng.standard_normal((40, 6)).astype(np.float32)
    xt = xs + 0.1
    pipe = make_pmh_pipeline(xt, nuisance="domain_shift", rank=4)
    pipe2 = clone(pipe)
    np.testing.assert_array_equal(pipe2.named_steps["pmh"].X_target, xt)


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


def test_pmh_config_presets():
    assert PMHConfig.conservative().weight < PMHConfig.aggressive().weight


def test_compare_arms_sklearn():
    rng = np.random.default_rng(0)
    n, d = 100, 20
    xs = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 3, n)
    xt = xs + 0.5
    yt = y.copy()
    res = compare_arms_sklearn(xs, y, xt, yt, rank=5, include_coral=False)
    assert "b0" in res.arms and "matched" in res.arms


def test_tune_sklearn_matcher():
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
        xs,
        y,
        xt,
        yt,
        scorer=scorer,
        nuisance="domain_shift",
        rank_grid=(4, 8),
        n_folds=2,
    )
    assert "rank" in out.best_params
    assert out.best_score >= 0


def test_suggest_nuisance_frozen_features():
    sug = suggest_nuisance(has_source_labels=True, has_target_labels=True, has_target_domain=True)
    assert sug.nuisance in ("subspace", "domain_shift")
