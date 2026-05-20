"""Paper presets and calibrators."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pmh.benchmark.presets import get_preset, list_presets
from pmh.calibrate import (
    content_residual_subspace,
    gradient_subspace_numpy,
    subspace_artifact_from_deltas,
)
from pmh.compare import compare_arms_sklearn


def test_list_presets():
    names = list_presets()
    assert "t1_office31_sklearn" in names
    assert "t4_domain_d4" in names


def test_office31_preset_sklearn_kwargs():
    p = get_preset("t1_office31_sklearn")
    assert p.sklearn_benchmark["paper_protocol"] is True
    assert p.default_rank == 32


def test_subspace_from_deltas():
    rng = np.random.default_rng(0)
    d, r, n = 20, 4, 50
    w_true = rng.standard_normal((d, r)).astype(np.float32)
    g = rng.standard_normal((n, r)) @ w_true.T
    art = subspace_artifact_from_deltas(g, rank=r)
    assert art.sigma.shape == (d, d)
    assert "w" in art.metadata


def test_content_residual_temporal():
    x = np.random.randn(10, 5, 8).astype(np.float32)
    w, art = content_residual_subspace(x, rank=3, source="temporal")
    assert w.shape[0] == 8


def test_gradient_subspace():
  g = np.random.randn(30, 12).astype(np.float32)
  w, art = gradient_subspace_numpy(g, rank=4)
  assert w.shape == (12, 4)


def test_compare_sklearn_with_preset():
    pytest.importorskip("sklearn")
    from pmh.benchmark.sklearn_protocol import synthetic_office31_features

    xs, y, xt, yt = synthetic_office31_features(n=100, d=24, seed=0)
    res = compare_arms_sklearn(xs, y, xt, yt, preset="t1_synthetic_sklearn")
    assert "matched" in res.arms
    assert any("Preset:" in n for n in res.notes)
