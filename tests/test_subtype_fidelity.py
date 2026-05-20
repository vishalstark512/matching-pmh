"""Synthetic goldens: subtype estimators vs closed form / Paper2 T1 reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pmh import SigmaTaskConfig, estimate_from_config
from pmh.calibrate import (
    content_residual_subspace,
    gradient_subspace_numpy,
    style_gram_from_deltas,
    subspace_artifact_from_deltas,
)
from pmh.estimators.d3_augmentation import estimate_d3
from pmh.estimators.d4_domain import gram_from_diff
from pmh.numpy_api import (
    estimate_cross_domain_subspace_numpy,
    estimate_sigma_task_numpy,
    gram_from_diff_numpy,
)

_PAPER2_T1 = (
    Path(__file__).resolve().parents[1].parent / "Paper2" / "T1" / "classical_pmh"
)


def _paper_estimate_cross_domain(*args, **kwargs):
    if not (_PAPER2_T1 / "common.py").is_file():
        pytest.skip("Paper2 T1 common.py not found (sibling repo)")
    if str(_PAPER2_T1) not in sys.path:
        sys.path.insert(0, str(_PAPER2_T1))
    from common import estimate_cross_domain_subspace  # noqa: PLC0415

    return estimate_cross_domain_subspace(*args, **kwargs)


def test_d4_gram_numpy_closed_form():
    rng = np.random.default_rng(7)
    n, d = 40, 8
    s = rng.standard_normal((n, d)).astype(np.float32)
    t = s + 0.3
    s_c = s - s.mean(0, keepdims=True)
    t_c = t - t.mean(0, keepdims=True)
    diff = s_c[:n] - t_c[:n]
    expected = (diff.T @ diff) / n
    got = gram_from_diff_numpy(s, t)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_d4_numpy_matches_torch():
    rng = np.random.default_rng(3)
    s = torch.from_numpy(rng.standard_normal((35, 6)).astype(np.float32))
    t = s + 0.2
    g_np = gram_from_diff_numpy(s.numpy(), t.numpy())
    g_th = gram_from_diff(s, t).numpy()
    np.testing.assert_allclose(g_np, g_th, rtol=1e-5, atol=1e-5)


def test_d4_estimate_config_matches_gram():
    rng = np.random.default_rng(11)
    x_s = rng.standard_normal((50, 10)).astype(np.float32)
    x_t = x_s + rng.standard_normal((50, 10)).astype(np.float32) * 0.1
    shrink = 1e-6
    art = estimate_sigma_task_numpy(
        x_s, x_t, config=SigmaTaskConfig.for_domain(shrinkage=shrink)
    )
    g = gram_from_diff_numpy(x_s, x_t)
    expected = 0.5 * (g + g.T) + shrink * np.eye(10, dtype=np.float32)
    np.testing.assert_allclose(art.sigma.numpy(), expected, rtol=1e-4, atol=1e-4)


def test_d1_w_matches_numpy_api_and_paper2():
    rng = np.random.default_rng(42)
    n, d, n_classes = 80, 16, 3
    x_src = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, n_classes, n)
    q, _ = np.linalg.qr(rng.standard_normal((d, 5)).astype(np.float32))
    shift = (x_src @ q) @ q.T
    x_tgt = (x_src + shift + 0.05 * rng.standard_normal((n, d))).astype(np.float32)

    w_lib = estimate_cross_domain_subspace_numpy(
        x_src, y, x_tgt, y, rank=4, seed=0, n_pairs_per_class=50
    )
    sigma_lib = w_lib @ w_lib.T
    art = estimate_sigma_task_numpy(
        x_src, y, x_tgt, y, config=SigmaTaskConfig.for_subspace(rank=4, shrinkage=0.0)
    )
    np.testing.assert_allclose(art.sigma.numpy(), sigma_lib, rtol=1e-5, atol=1e-5)

    w_paper = _paper_estimate_cross_domain(
        x_src, y, x_tgt, y, rank=4, seed=0, n_pairs_per_class=50
    )
    # Subspaces should align: ||P_lib - P_paper||_F small
    p_lib = w_lib @ w_lib.T
    p_paper = w_paper @ w_paper.T
    err = np.linalg.norm(p_lib - p_paper, ord="fro") / max(np.linalg.norm(p_paper, ord="fro"), 1e-8)
    assert err < 0.05


def test_d3_aug_modes_rank_bounded():
    rng = torch.Generator().manual_seed(1)
    k, n, d = 5, 30, 12
    modes = torch.randn(k, n, d, generator=rng)
    sigma = estimate_d3(modes, shrinkage=0.0)
    evals = torch.linalg.eigvalsh(sigma)
    rank_eff = int((evals > 1e-5).sum().item())
    assert rank_eff <= k * d  # trivial
    assert rank_eff <= d
    # With K distinct random mode means [K,d], Gram rank <= K
    modes2 = torch.randn(k, d, generator=rng)
    sigma2 = estimate_d3(modes2, shrinkage=0.0)
    evals2 = torch.linalg.eigvalsh(sigma2)
    assert int((evals2 > 1e-5).sum().item()) <= k


def test_calibrator_gradient_subspace_psd():
    rng = np.random.default_rng(0)
    g = rng.standard_normal((40, 10)).astype(np.float32)
    _w, art = gradient_subspace_numpy(g, rank=4)
    assert art.method == "D3"
    sig = art.sigma.numpy() if hasattr(art.sigma, "numpy") else np.asarray(art.sigma)
    assert sig.shape == (10, 10)
    evals = np.linalg.eigvalsh(sig)
    assert np.all(evals >= -1e-5)
    w = art.metadata["w"]
    assert tuple(w.shape) == (10, 4)


def test_calibrator_content_residual_shape():
    rng = np.random.default_rng(2)
    seq = rng.standard_normal((12, 8, 6)).astype(np.float32)
    w, art = content_residual_subspace(seq, rank=3, source="content")
    assert w.shape == (6, 3)
    assert art.method == "D6"
    evals = np.linalg.eigvalsh(art.sigma.numpy())
    assert np.all(evals >= -1e-5)


def test_calibrator_style_gram_psd():
    deltas = torch.randn(25, 8)
    art = style_gram_from_deltas(deltas, rank=4, shrinkage=0.01)
    assert art.method == "D7"
    evals = torch.linalg.eigvalsh(art.sigma.float())
    assert bool((evals >= -1e-5).all())
    assert art.metadata["w"].shape == (8, 4)


def test_calibrator_pgd_deltas_matches_d7_path():
    rng = np.random.default_rng(9)
    deltas = rng.standard_normal((30, 7)).astype(np.float32)
    art = subspace_artifact_from_deltas(deltas, rank=3, method="D7")
    assert art.sigma.shape == (7, 7)
    w = art.metadata["w"].numpy() if hasattr(art.metadata["w"], "numpy") else art.metadata["w"]
    assert w.shape == (7, 3)


def test_d1_torch_estimate_from_config_w_norm():
    rng = torch.Generator().manual_seed(5)
    x = torch.randn(50, 9, generator=rng)
    y = torch.randint(0, 2, (50,))
    x_t = x + 0.4 * torch.randn(50, 9, generator=rng)
    art = estimate_from_config(
        SigmaTaskConfig.for_subspace(rank=3),
        x,
        y,
        x_t,
        y,
        seed=0,
    )
    w = art.metadata["w"]
    sigma = art.sigma
    # Sigma = W W^T (up to shrinkage)
    ww = w @ w.T
    assert torch.allclose(sigma, ww, atol=1e-3) or torch.norm(sigma - ww) / torch.norm(ww) < 0.1
