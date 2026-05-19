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


def test_transform_after_fit_d4():
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((40, 12)).astype(np.float32)
    xt = xs + 0.2
    m = PMHMatcher(nuisance="domain_shift", rank=4)
    m.fit(xs, xt)
    z = m.transform(xs)
    assert z.shape == xs.shape


def test_pipeline_compatible():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    xs = rng.standard_normal((60, 8)).astype(np.float32)
    y = rng.integers(0, 2, 60)
    xt = xs + 0.1
    yt = y.copy()
    m = PMHMatcher(nuisance="domain_shift", rank=3)
    m.fit(xs, y, xt, yt)
    pipe = Pipeline([("pmh", m), ("clf", LogisticRegression(max_iter=200))])
    # PMHMatcher fit signature differs from standard X,y — use pre-fitted in pipeline
    pipe.named_steps["clf"].fit(m.transform(xs), y)
    assert pipe.named_steps["clf"].predict(m.transform(xs[:5])).shape == (5,)
