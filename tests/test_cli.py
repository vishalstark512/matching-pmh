"""pmh-train CLI smoke tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_list_methods():
    from pmh.cli.main import main

    assert main(["list-methods"]) == 0


def test_estimate_d2(tmp_path: Path):
    from pmh.cli.main import main

    job = {
        "estimator": {"method": "D2", "dim": 8, "noise_level": 0.2},
        "data": {"dim": 8},
        "output": str(tmp_path / "d2"),
    }
    cfg = tmp_path / "job.json"
    cfg.write_text(json.dumps(job), encoding="utf-8")
    assert main(["estimate", "--config", str(cfg)]) == 0
    assert (tmp_path / "d2.pt").exists()


def test_estimate_d4_npy(tmp_path: Path):
    from pmh.cli.main import main

    src = tmp_path / "src.npy"
    tgt = tmp_path / "tgt.npy"
    rng = np.random.default_rng(0)
    np.save(src, rng.standard_normal((40, 16)).astype(np.float32))
    np.save(tgt, rng.standard_normal((40, 16)).astype(np.float32) + 0.5)
    job = {
        "estimator": {"method": "D4", "rank": 4},
        "data": {"source_npy": str(src), "target_npy": str(tgt)},
        "output": str(tmp_path / "d4"),
    }
    cfg = tmp_path / "d4.json"
    cfg.write_text(json.dumps(job), encoding="utf-8")
    assert main(["estimate", "--config", str(cfg)]) == 0


def test_run_dry_run(tmp_path: Path):
    from pmh import SigmaTaskConfig, estimate_from_config
    from pmh.cli.main import main

    a = torch_rand_artifact(tmp_path)
    job = {
        "artifact": str(a),
        "pmh": {"weight": 0.3},
        "training": {"backend": "hf_trainer"},
    }
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps(job), encoding="utf-8")
    assert main(["run", "--config", str(cfg)]) == 0


def torch_rand_artifact(tmp_path: Path) -> Path:
    import torch

    from pmh import SigmaTaskConfig, estimate_from_config

    h0 = torch.randn(32, 8)
    h1 = h0 + 0.2
    art = estimate_from_config(SigmaTaskConfig.for_domain(rank=2), h0, h1)
    return art.save(tmp_path / "sigma")


def test_catalog_validate():
    from pmh.catalog import validate_job_data

    assert "rank" in validate_job_data("D1", {"source_npy": "a.npy"})
    assert validate_job_data("D2", {"dim": 4}) == []


@pytest.mark.skipif(sys.platform.startswith("win"), reason="console script may be uninstalled in editable-only env")
def test_console_script_list():
    subprocess.run([sys.executable, "-m", "pmh.cli.main", "list-methods"], check=True)
