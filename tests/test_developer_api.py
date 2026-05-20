"""Developer-facing high-level API."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import (
    check_applicability,
    evaluate_baseline_vs_pmh,
    evaluate_robust_fit,
    load_g2_demo_arrays,
    robust_fit,
    suggest_hook,
)
from pmh.developer import DomainPair


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(8, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.enc(x)))


def test_check_applicability_no_go():
    r = check_applicability(new_classes_at_deploy=True)
    assert r.verdict == "no_go"
    assert not r.can_proceed


def test_check_applicability_suggests_subtype():
    r = check_applicability(stack="pytorch", has_target_labels=False)
    assert r.suggested_nuisance == "domain_shift"
    assert any("D4" in x for x in r.reasons)


def test_domain_pair_validate():
    xs = np.random.randn(20, 8).astype(np.float32)
    xt = np.random.randn(15, 8).astype(np.float32)
    p = DomainPair.from_arrays(xs, xt)
    assert p.feature_dim == 8


def test_suggest_hook():
    m = Tiny()
    x = torch.randn(2, 8)
    s = suggest_hook(m, probe_input=x, alias="enc")
    assert s.repr_dim == 4


def test_robust_fit_quick():
    torch.manual_seed(0)
    m = Tiny()
    tr = DataLoader(TensorDataset(torch.randn(32, 8), torch.randint(0, 2, (32,))), batch_size=8)
    src = DataLoader(TensorDataset(torch.randn(24, 8), torch.randint(0, 2, (24,))), batch_size=8)
    tgt = DataLoader(TensorDataset(torch.randn(24, 8) + 0.5, torch.randint(0, 2, (24,))), batch_size=8)
    out = robust_fit(m, tr, source_batches=src, target_batches=tgt, hook=m.enc, head=m.head, epochs=1)
    assert out.stats
    assert out.preflight is not None


def test_evaluate_robust_fit():
    torch.manual_seed(0)
    m = Tiny()
    tr = DataLoader(TensorDataset(torch.randn(40, 8), torch.randint(0, 2, (40,))), batch_size=8)
    val = DataLoader(TensorDataset(torch.randn(20, 8) + 0.3, torch.randint(0, 2, (20,))), batch_size=10)
    src = DataLoader(TensorDataset(torch.randn(24, 8), torch.randint(0, 2, (24,))), batch_size=8)
    tgt = DataLoader(TensorDataset(torch.randn(24, 8) + 0.5, torch.randint(0, 2, (24,))), batch_size=8)
    rep = evaluate_robust_fit(
        m,
        tr,
        val,
        source_batches=src,
        target_batches=tgt,
        hook=m.enc,
        head=m.head,
        epochs=1,
        include_falsification=True,
    )
    assert 0 <= rep.baseline_metric <= 1
    assert 0 <= rep.pmh_metric <= 1
    assert rep.falsification_arms
    assert "matched" in rep.falsification_arms
    assert rep.summary()


def test_load_g2_demo_arrays():
    x, y, xt, yt = load_g2_demo_arrays(n=80, seed=1)
    assert x.shape[0] == y.shape[0] == 80
    assert xt.shape == x.shape


def test_evaluate_baseline_vs_pmh():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n, d = 120, 16
    xs = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 3, n)
    q, _ = np.linalg.qr(rng.standard_normal((d, 4)))
    xt = xs + (xs @ q) @ q.T
    rep = evaluate_baseline_vs_pmh(
        x_source=xs, y_source=y, x_target=xt, y_target=y,
        compare_to=(),
    )
    assert 0 <= rep.baseline_metric <= 1
    assert 0 <= rep.pmh_metric <= 1
    assert rep.falsification_arms
    assert {"matched", "wrong_w", "isotropic"} <= set(rep.falsification_arms)
    assert rep.summary()
