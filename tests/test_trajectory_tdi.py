"""Trajectory TDI metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHTrainer, trajectory_tdi_encoder, trajectory_tdi_layerwise


def test_trajectory_tdi_layerwise_lower_when_stable():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((50, 8)).astype(np.float32)
    pert = clean + 0.01 * rng.standard_normal((50, 8)).astype(np.float32)
    loose = clean + 0.5 * rng.standard_normal((50, 8)).astype(np.float32)
    tdi_stable, _ = trajectory_tdi_layerwise([clean], [pert])
    tdi_loose, _ = trajectory_tdi_layerwise([clean], [loose])
    assert tdi_stable < tdi_loose


def test_trajectory_tdi_encoder():
    class Enc(nn.Module):
        def forward(self, x):
            return self.lin(x)

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(6, 4)

    enc = Enc()
    x = torch.randn(40, 6)
    loader = DataLoader(TensorDataset(x), batch_size=8)
    out = trajectory_tdi_encoder(enc, enc, loader, sigma=0.05, max_batches=5, seed=0)
    assert "trajectory_tdi" in out
    assert out["n_samples"] > 0


def test_pmh_trainer_measure_trajectory_tdi():
    class B(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(5, 3)

        def forward(self, x):
            return self.net(x)

    m = B()
    x = torch.randn(30, 5)
    loader = DataLoader(TensorDataset(x), batch_size=10)
    tr = PMHTrainer(m, hook=m, nuisance="domain_shift", rank=2)
    metrics = tr.measure_trajectory_tdi(loader, sigma=0.02, max_batches=3)
    assert metrics["trajectory_tdi"] >= 0.0
