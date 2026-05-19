import tempfile
from pathlib import Path

import torch

from pmh import (
    PMHConfig,
    SigmaTaskConfig,
    SigmaTaskEstimate,
    estimate_from_config,
    preflight_eigengap,
)
from pmh.preflight import PreflightStatus


def test_config_factories():
    c = SigmaTaskConfig.for_domain(rank=8)
    assert c.method == "D4" and c.rank == 8
    c2 = SigmaTaskConfig.for_isotropic(16, noise_level=0.2)
    assert c2.dim == 16 and c2.noise_level == 0.2


def test_artifact_roundtrip():
    src = torch.randn(40, 6)
    tgt = src + 0.2 * torch.randn(40, 6)
    cfg = SigmaTaskConfig.for_domain(rank=3)
    est = estimate_from_config(cfg, src, tgt)
    assert isinstance(est, SigmaTaskEstimate)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sigma_task"
        pt_path = est.save(path)
        loaded = SigmaTaskEstimate.load(pt_path)
        assert loaded.method == "D4"
        assert torch.allclose(loaded.sigma, est.sigma, atol=1e-5)


def test_preflight_status():
    cov = torch.diag(torch.tensor([5.0, 4.0, 1.0, 0.1]))
    st, g = preflight_eigengap(cov, rank=2)
    assert st == PreflightStatus.PASS
    assert g == 4.0


def test_pmh_config_warmup():
    cfg = PMHConfig(warmup_epochs=2, warmup_ramp_epochs=4, weight=0.5)
    assert cfg.pmh_weight_for_epoch(1) == 0.0
    assert cfg.pmh_weight_for_epoch(3) == 0.25  # ramp multiplier at epoch 3
    assert cfg.pmh_weight_for_epoch(10) == 1.0
