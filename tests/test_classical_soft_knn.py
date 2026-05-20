"""T1 soft k-NN helpers."""

from __future__ import annotations

import numpy as np

from pmh.classical import compare_knn_hard_vs_soft, softlift
from pmh.sklearn_match import MatchedSubspaceProjector


def test_softlift_alpha_zero_is_hard_project():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 8)).astype(np.float32)
    y = rng.integers(0, 3, 40)
    proj = MatchedSubspaceProjector(rank=3, seed=0).fit(x, y, x + 0.5, y)
    hard = proj.transform(x)
    soft0 = softlift(x, proj.w_, alpha=0.0)
    np.testing.assert_allclose(hard, soft0, atol=1e-5)


def test_compare_knn_hard_vs_soft_keys():
    from pmh import load_g2_demo_arrays

    xs, ys, xt, yt = load_g2_demo_arrays(n=200, seed=1)
    proj = MatchedSubspaceProjector(rank=8, seed=0).fit(xs, ys, xt, yt)
    out = compare_knn_hard_vs_soft(xs, ys, xt, yt, proj.w_, seed=0)
    assert "b0" in out and "matched_hard" in out and "matched_soft" in out
