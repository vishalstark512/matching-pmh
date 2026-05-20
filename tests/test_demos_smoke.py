"""Smoke-run scripts/demos (CI only — adoption path is notebooks)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "scripts" / "demos"

SMOKE_SCRIPTS = [
    "first_run_domain_shift.py",
    "office31_sklearn.py",
    "benchmark_sklearn_table.py",
]


def _run_demo(script: str, timeout: int = 120) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PMH_QUICK"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    path = DEMOS / script
    assert path.is_file(), f"missing {path}"
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


@pytest.mark.parametrize("script", SMOKE_SCRIPTS)
def test_demo_smoke(script: str) -> None:
    proc = _run_demo(script, timeout=180)
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
