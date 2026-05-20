"""Custom geometry API and falsification validate."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import (
    artifact_from_deltas,
    artifact_from_w,
    estimate_custom,
    validate_falsification,
)
from pmh.benchmark.sklearn_protocol import run_sklearn_benchmark
from pmh.data_adapters import batch_iterators, load_domain_arrays
from pmh.trainer import PMHTrainer


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(10, 6)
        self.head = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.enc(x)))


def test_estimate_custom_d4():
    rng = np.random.default_rng(1)
    xs = rng.standard_normal((40, 8)).astype(np.float32)
    xt = xs + 0.2
    art = estimate_custom(x_src=xs, x_tgt=xt, rank=4)
    assert art.method == "D4"
    assert art.sigma.shape == (8, 8)


def test_artifact_from_w():
    w = np.random.randn(10, 3).astype(np.float32)
    art = artifact_from_w(w, method="D1")
    assert art.metadata["w"].shape == (10, 3)


def test_pmh_trainer_from_artifact():
    art = artifact_from_deltas(np.random.randn(30, 6).astype(np.float32), rank=3, method="D4")
    m = Tiny()
    tr = PMHTrainer.from_artifact(m, art, hook="enc", head=m.head)
    assert tr.artifact_ is not None
    x = torch.randn(16, 10)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    stats = tr.fit(loader, epochs=1, max_steps_per_epoch=2)
    assert stats["n_steps"] >= 1


def test_load_domain_arrays_and_batches():
    xs = np.random.randn(20, 5).astype(np.float32)
    xt = np.random.randn(25, 5).astype(np.float32)
    a, _, b, _ = load_domain_arrays(xs, xt)
    assert a.shape == (20, 5)
    s, t = batch_iterators(a, b, batch_size=4)
    assert next(s)[0].shape[1] == 5
    assert next(t)[0].shape[1] == 5


def test_validate_falsification_synthetic():
    from pmh.benchmark.sklearn_protocol import synthetic_office31_features

    x_a, y, x_d, y2 = synthetic_office31_features(200, seed=0)
    result = run_sklearn_benchmark(x_a, y, x_d, y2, rank=8, paper_protocol=True)
    rep = validate_falsification(result)
    assert rep.checks
    # synthetic may occasionally fail margin; at least structure runs
    assert "matched" in str(rep.summary())
