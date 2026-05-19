"""D1 labeled estimator and Lemma-C wrong-W training."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pmh import PMHLoss, SigmaTaskConfig, estimate_from_config, estimate_sigma_task
from pmh.estimators.d1_subspace import estimate_d1_gram_unlabeled
from pmh.sklearn_match import wrong_w_subspace_numpy


def test_d1_requires_labels():
    src, tgt = torch.randn(20, 6), torch.randn(20, 6)
    with pytest.raises(ValueError, match="D1 requires"):
        estimate_sigma_task(src, tgt, method="D1", rank=2)


def test_d1_labeled_returns_w_in_artifact():
    rng = torch.Generator().manual_seed(0)
    x_src = torch.randn(40, 8, generator=rng)
    x_tgt = x_src + 0.5 * torch.randn(40, 8, generator=rng)
    y = torch.randint(0, 3, (40,))
    art = estimate_from_config(
        SigmaTaskConfig.for_subspace(rank=3),
        x_src,
        y,
        x_tgt,
        y,
        seed=0,
    )
    assert art.method == "D1"
    assert "w" in art.metadata
    assert art.metadata["w"].shape == (8, 3)


def test_d1_labeled_differs_from_gram_unlabeled():
    rng = np.random.default_rng(0)
    n, d = 60, 12
    x_src = torch.from_numpy(rng.standard_normal((n, d)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, 4, n))
    q, _ = np.linalg.qr(rng.standard_normal((d, 4)).astype(np.float32))
    shift = (x_src.numpy() @ q) @ q.T
    x_tgt = torch.from_numpy((x_src.numpy() + shift).astype(np.float32))
    s_labeled = estimate_from_config(
        SigmaTaskConfig.for_subspace(rank=4),
        x_src,
        y,
        x_tgt,
        y,
        seed=0,
    ).sigma
    s_gram = estimate_d1_gram_unlabeled(x_src, x_tgt, rank=4)
    assert not torch.allclose(s_labeled, s_gram, atol=1e-4)


def test_wrong_w_orthogonal_to_matched():
    d, rank = 16, 4
    rng = np.random.default_rng(1)
    w = rng.standard_normal((d, rank)).astype(np.float32)
    q, _ = np.linalg.qr(w)
    w_m = q[:, :rank]
    q_wrong = wrong_w_subspace_numpy(w_m, rank, seed=2)
    inner = w_m.T @ q_wrong
    assert np.max(np.abs(inner)) < 1e-4


def test_pmh_loss_wrong_w_uses_orthogonal_sigma():
    d, rank = 10, 3
    art = estimate_from_config(
        SigmaTaskConfig.for_subspace(rank=rank),
        torch.randn(30, d),
        torch.randint(0, 3, (30,)),
        torch.randn(30, d),
        torch.randint(0, 3, (30,)),
        seed=0,
    )
    loss_m = PMHLoss(art, mode="matched", wrong_rank=rank, wrong_seed=0)
    loss_w = PMHLoss(art, mode="wrong_w", wrong_rank=rank, wrong_seed=0)
    h = torch.randn(8, d, requires_grad=True)
    lin = torch.nn.Linear(d, d, bias=False)
    h2 = lin(h)
    p_m = loss_m(h2)
    p_w = loss_w(h2)
    assert p_m.item() >= 0 and p_w.item() >= 0
    w = loss_w._matched_basis(d)
    gen = torch.Generator()
    gen.manual_seed(0)
    m = torch.randn(d, rank, generator=gen)
    residual = m - w @ (w.T @ m)
    q, _ = torch.linalg.qr(residual)
    assert torch.max(torch.abs(w.T @ q[:, :rank])).item() < 1e-4
