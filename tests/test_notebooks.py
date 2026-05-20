"""Starter notebooks: headless execute via nbconvert (slow — not in default CI)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

NOTEBOOK_SMOKE = [
    "tasks/t04a-vision-domain.ipynb",
    "tasks/t01-classical.ipynb",
]

NOTEBOOK_OPTIONAL = [
    "tasks/t07a-llm-style.ipynb",
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
def test_starter_notebook_smoke(name: str, _nbconvert) -> None:
    proc = _execute_notebook(name, timeout=360)
    assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.parametrize("name", NOTEBOOK_OPTIONAL)
def test_starter_notebook_optional_hf(name: str, _nbconvert) -> None:
    pytest.importorskip("transformers")
    proc = _execute_notebook(name, timeout=420)
    assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"
