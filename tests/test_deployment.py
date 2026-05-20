"""Deployment bundles, doctor, data dirs, Office-31 layout."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pmh import PMHConfig, export_deployment, load_deployment_bundle, run_doctor
from pmh.config import SigmaTaskConfig
from pmh.data_adapters import load_domain_dirs, resolve_feature_npy
from pmh.datasets.office31 import (
    DOMAIN_NAMES,
    download_office31,
    verify_office31_layout,
)
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
    assert "Step 5" in rep.summary()


def test_run_doctor_sklearn_demo():
    rep = run_doctor(stack="sklearn")
    assert rep.ok
    assert any("G2 demo" in c for c in rep.checks)


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


@pytest.mark.slow
def test_pytorch_validate_smoke():
    from pmh.benchmark.pytorch_smoke import run_pytorch_benchmark_smoke
    from pmh.benchmark.validate import validate_falsification

    result = run_pytorch_benchmark_smoke(rank=4, epochs=2)
    rep = validate_falsification(result, min_margin=-1.0)
    assert result.arms["matched"].val_metric is not None
    assert rep.checks or rep.failures


def _fake_domain_tree(root: Path, domain: str) -> None:
    d = root / domain / "class0"
    d.mkdir(parents=True)
    (d / "img.jpg").write_bytes(b"\xff\xd8\xff")


def test_verify_office31_layout_ok(tmp_path: Path) -> None:
    for dom in DOMAIN_NAMES:
        _fake_domain_tree(tmp_path, dom)
    verify_office31_layout(tmp_path)


def test_verify_office31_layout_missing(tmp_path: Path) -> None:
    _fake_domain_tree(tmp_path, "amazon")
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_office31_layout(tmp_path)


def test_download_skips_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for dom in DOMAIN_NAMES:
        _fake_domain_tree(tmp_path, dom)

    called = {"n": 0}

    def _no_download(*_a, **_k):
        called["n"] += 1

    monkeypatch.setattr("urllib.request.urlretrieve", _no_download)
    download_office31(tmp_path, force=False)
    assert called["n"] == 0
