"""sklearn API compatibility (optional extra)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pmh import PMHMatcher


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
    """Metadata routing on the estimator (sklearn >= 1.4, routing enabled)."""
    import sklearn

    sklearn_version = tuple(
        int(x) for x in sklearn.__version__.split(".")[:2]
    )
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
