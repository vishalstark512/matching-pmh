"""Benchmark protocol and reports."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, SigmaTaskConfig, estimate_from_config
from pmh.benchmark import (
    benchmark_to_markdown,
    run_benchmark_protocol,
    run_sklearn_benchmark,
    write_benchmark_report,
)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(8, 4)
        self.head = nn.Linear(4, 2)

    def encode(self, x):
        return torch.relu(self.enc(x))


def test_benchmark_protocol_runs_four_arms():
    torch.manual_seed(0)
    x = torch.randn(80, 8)
    h0 = torch.relu(nn.Linear(8, 4)(x))
    h1 = h0 + 0.4 * torch.randn(80, 4)
    art = estimate_from_config(SigmaTaskConfig.for_domain(rank=3), h0, h1)

    xtr, ytr = torch.randn(64, 8), torch.randint(0, 2, (64,))
    xva, yva = torch.randn(32, 8), torch.randint(0, 2, (32,))
    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=16)
    val_loader = DataLoader(TensorDataset(xva, yva), batch_size=32)

    def factory():
        return Tiny()

    def setup(m):
        return m.encode, m.head, torch.optim.Adam(m.parameters(), lr=1e-2)

    res = run_benchmark_protocol(
        art,
        factory,
        setup,
        train_loader,
        val_loader,
        epochs=2,
        pmh_config=PMHConfig(weight=0.5, cap_ratio=0.3, warmup_epochs=0),
        max_steps_per_epoch=5,
    )
    assert set(res.arms.keys()) == {"b0", "matched", "wrong_w", "isotropic"}
    for arm in res.arms.values():
        assert arm.val_metric is not None
        assert len(arm.epochs) == 2

    md = benchmark_to_markdown(res)
    assert "matched" in md and "wrong_w" in md


def test_sklearn_benchmark_high_d():
    """Isotropic arm uses Vt (feature directions); must work when n < d."""
    pytest.importorskip("sklearn")
    from pmh.benchmark.sklearn_protocol import run_sklearn_benchmark, synthetic_office31_features

    xs, y, xt, yt = synthetic_office31_features(n=80, d=256, seed=1)
    res = run_sklearn_benchmark(xs, y, xt, yt, rank=8, include_coral=False, seed=1)
    assert "isotropic" in res.arms


def test_sklearn_benchmark_synthetic():
    pytest.importorskip("sklearn")
    import numpy as np

    rng = np.random.default_rng(0)
    n, d = 120, 20
    x_a = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 3, n)
    q, _ = np.linalg.qr(rng.standard_normal((d, 5)))
    x_d = x_a + 1.2 * (x_a @ q) @ q.T
    res = run_sklearn_benchmark(x_a, y, x_d, y, rank=5, include_coral=True)
    assert "b0" in res.arms and "matched" in res.arms
    assert res.arms["matched"].val_metric is not None


def test_write_benchmark_report(tmp_path):
    torch.manual_seed(1)
    art = estimate_from_config(SigmaTaskConfig.for_isotropic(6, 0.1))
    from pmh.benchmark.protocol import BenchmarkResult, ArmRunResult

    res = BenchmarkResult("D2", "pass", None)
    res.arms["b0"] = ArmRunResult("b0", val_metric=0.5)
    paths = write_benchmark_report(res, tmp_path)
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "b0" in data["arms"]
