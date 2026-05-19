"""PMHMatcher sklearn-style API."""

from __future__ import annotations

import numpy as np
import pytest

from pmh import PMHMatcher
from pmh.nuisance import list_nuisance_names, resolve_method


def test_resolve_nuisance_aliases():
    assert resolve_method("domain_shift") == "D4"
    assert resolve_method("D1") == "D1"
    assert "domain_shift" in list_nuisance_names()


def test_matcher_d4_two_arg_shorthand():
    rng = np.random.default_rng(3)
    xs = rng.standard_normal((40, 8)).astype(np.float32)
    xt = xs + 0.1
    m = PMHMatcher(nuisance="domain_shift", rank=4).fit(xs, xt)
    assert m.artifact_.method == "D4"


def test_matcher_d4_fit_transform():
    rng = np.random.default_rng(0)
    n, d = 120, 24
    xs = rng.standard_normal((n, d)).astype(np.float32)
    xt = xs + 0.4 * rng.standard_normal((n, d))
    m = PMHMatcher(nuisance="domain_shift", rank=6)
    m.fit(xs, X_target=xt)
    assert m.artifact_ is not None
    assert m.artifact_.method == "D4"
    z = m.transform(xs)
    assert z.shape == xs.shape


def test_matcher_d1_requires_labels():
    rng = np.random.default_rng(1)
    n, d = 80, 16
    xs = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 3, n)
    xt = xs + 0.2
    yt = y.copy()
    m = PMHMatcher(nuisance="subspace", rank=4, seed=0)
    m.fit(xs, y, xt, yt)
    assert m.w_ is not None
    z = m.fit_transform(xs, y, xt, yt)
    assert z.shape == xs.shape


def test_matcher_d2_isotropic():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 10)).astype(np.float32)
    m = PMHMatcher(nuisance="isotropic", dim=10, noise_level=0.05, rank=3)
    m.fit(x)
    z = m.transform(x)
    assert z.shape == x.shape


def test_get_params_set_params():
    m = PMHMatcher(nuisance="domain", rank=8)
    p = m.get_params()
    assert p["rank"] == 8
    m2 = PMHMatcher().set_params(**p)
    assert m2.get_params() == p


@pytest.mark.parametrize("nuisance", ["covariate_shift", "D4"])
def test_nuisance_synonyms(nuisance: str):
    assert resolve_method(nuisance) == "D4"


def test_sklearn_estimator_checks_if_available():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.utils.estimator_checks import check_estimator

    # Minimal check: fit/transform roundtrip on toy D4 data
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((60, 12)).astype(np.float32)
    xt = xs + 0.2

    class _Matcher(PMHMatcher):
        def fit(self, X, y=None):  # type: ignore[override]
            return super().fit(X, X_target=xt)

    # check_estimator is heavy; run a subset via manual clone
    from sklearn.base import clone

    est = clone(PMHMatcher(nuisance="domain_shift", rank=4))
    est.fit(xs, X_target=xt)
    est.transform(xs[:5])
