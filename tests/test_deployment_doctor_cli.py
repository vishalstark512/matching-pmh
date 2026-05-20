"""Deployment export, doctor, estimate dirs, validate pytorch."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pmh import PMHConfig, export_deployment, load_deployment_bundle, run_doctor
from pmh.config import SigmaTaskConfig
from pmh.data_adapters import load_domain_dirs, resolve_feature_npy
from pmh.numpy_api import estimate_sigma_task_numpy


def test_export_and_load_deployment():
    xs = np.random.randn(30, 6).astype(np.float32)
    xt = xs + 0.1
    art = estimate_sigma_task_numpy(xs, xt, config=SigmaTaskConfig.for_domain(rank=3))
    with tempfile.TemporaryDirectory() as td:
        bundle = export_deployment(
            art,
            Path(td) / "bundle",
            pmh_config=PMHConfig.balanced(),
            hook="backbone",
            nuisance="domain_shift",
        )
        assert bundle.sigma_pt.is_file()
        assert bundle.manifest.is_file()
        loaded, manifest, pcfg = load_deployment_bundle(bundle.root)
        assert loaded.method == art.method
        assert manifest["hook"] == "backbone"
        assert pcfg is not None


def test_run_doctor_pytorch():
    rep = run_doctor(stack="pytorch")
    assert rep.ok
    assert "torch" in " ".join(rep.checks).lower()


def test_load_domain_dirs(tmp_path: Path):
    sd = tmp_path / "a"
    td = tmp_path / "b"
    sd.mkdir()
    td.mkdir()
    np.save(sd / "features.npy", np.random.randn(20, 5).astype(np.float32))
    np.save(td / "features.npy", np.random.randn(25, 5).astype(np.float32))
    xs, _, xt, _ = load_domain_dirs(sd, td)
    assert xs.shape == (20, 5)
    assert xt.shape == (25, 5)
    assert resolve_feature_npy(sd).name == "features.npy"


def test_cli_doctor(capsys):
    from pmh.cli.main import main

    assert main(["doctor", "--stack", "pytorch"]) == 0
    assert "doctor" in capsys.readouterr().out.lower()


def test_cli_estimate_npy_flags(tmp_path: Path, capsys):
    from pmh.cli.main import main

    s = tmp_path / "s.npy"
    t = tmp_path / "t.npy"
    np.save(s, np.random.randn(15, 4).astype(np.float32))
    np.save(t, np.random.randn(18, 4).astype(np.float32))
    out = tmp_path / "art"
    assert main(["estimate", "--source-npy", str(s), "--target-npy", str(t), "-o", str(out)]) == 0
    assert (out.with_suffix(".pt")).is_file() or Path(str(out) + ".pt").is_file()


@pytest.mark.slow
def test_pytorch_validate_smoke():
    from pmh.benchmark.pytorch_smoke import run_pytorch_benchmark_smoke
    from pmh.benchmark.validate import validate_falsification

    result = run_pytorch_benchmark_smoke(rank=4, epochs=2)
    rep = validate_falsification(result, min_margin=-1.0)
    assert result.arms["matched"].val_metric is not None
    assert rep.checks or rep.failures
