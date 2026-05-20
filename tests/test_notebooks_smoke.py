"""Execute onboarding notebooks headlessly (CPU, PMH_QUICK=1)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

# Fast synthetic notebooks (no downloads)
NOTEBOOK_SMOKE = [
    "domain_shift_first_run.ipynb",
    "sklearn_frozen_features_first_run.ipynb",
]

NOTEBOOK_OPTIONAL = [
    "hf_two_corpora_first_run.ipynb",
]


def _execute_notebook(name: str, timeout: int = 300) -> subprocess.CompletedProcess:
    nb = NOTEBOOKS / name
    assert nb.is_file(), nb
    out = ROOT / "tmp_notebook_out" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PMH_QUICK"] = "1"
    env["USE_TF"] = "0"
    env["USE_FLAX"] = "0"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--output",
            str(out),
            str(nb),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


@pytest.fixture(scope="module")
def _nbconvert():
    pytest.importorskip("nbconvert")
    pytest.importorskip("nbformat")


@pytest.mark.parametrize("name", NOTEBOOK_SMOKE)
def test_notebook_smoke(name: str, _nbconvert) -> None:
    proc = _execute_notebook(name, timeout=360)
    assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.parametrize("name", NOTEBOOK_OPTIONAL)
def test_notebook_optional_hf(name: str, _nbconvert) -> None:
    pytest.importorskip("transformers")
    proc = _execute_notebook(name, timeout=420)
    assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"
