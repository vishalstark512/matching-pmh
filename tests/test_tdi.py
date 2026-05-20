"""TDI, geometry, and trajectory metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHTrainer, trajectory_tdi_encoder, trajectory_tdi_layerwise
from pmh.tdi import (
    directional_drift_numpy,
    geometry_report,
    tdi_cls,
    tdi_feature_isotropic,
)


def test_tdi_cls_lower_when_tighter_clusters():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), 40)
    loose = rng.standard_normal((120, 8)).astype(np.float32)
    tight = loose.copy()
    for c in range(3):
        mask = labels == c
        tight[mask] = tight[mask].mean(0) + 0.05 * rng.standard_normal((mask.sum(), 8))
    assert tdi_cls(tight, labels) < tdi_cls(loose, labels)


def test_directional_drift_ratio():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 10)).astype(np.float32)
    w = rng.standard_normal((10, 3)).astype(np.float32)
    d_n, d_s, ratio = directional_drift_numpy(x, w, sigma=0.2, n_noise=20, seed=0)
    assert d_n >= 0 and d_s >= 0 and ratio >= 0


def test_geometry_report_keys():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((60, 6)).astype(np.float32)
    y = rng.integers(0, 2, 60)
    w = rng.standard_normal((6, 2)).astype(np.float32)
    rep = geometry_report(x, y, w=w)
    d = rep.to_dict()
    assert d["tdi_cls"] is not None
    assert d["D_N_over_D_S"] is not None
    assert tdi_feature_isotropic(x) >= 0


def test_sklearn_benchmark_includes_geometry():
    pytest.importorskip("sklearn")
    from pmh.benchmark.sklearn_protocol import run_sklearn_benchmark, synthetic_office31_features

    xs, y, xt, yt = synthetic_office31_features(n=120, d=24, seed=0)
    res = run_sklearn_benchmark(xs, y, xt, yt, rank=6, include_coral=False, seed=0)
    geom = res.arms["matched"].geometry
    assert "tdi_cls" in geom
    assert res.arms["matched"].val_metric is not None


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
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(6, 4)

        def forward(self, x):
            return self.lin(x)

    enc = Enc()
    x = torch.randn(40, 6)
    loader = DataLoader(TensorDataset(x), batch_size=8)
    out = trajectory_tdi_encoder(enc, enc, loader, sigma=0.05, max_batches=5, seed=0)
    assert "trajectory_tdi" in out
    assert out["n_samples"] > 0


def test_pmh_trainer_measure_trajectory_tdi():
    class B(nn.Module):
        def __init__(self) -> None:
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
