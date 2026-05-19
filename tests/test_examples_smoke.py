"""Smoke-run example scripts (integration templates stay executable)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# Scripts that must finish quickly on CPU without downloads
SMOKE_SCRIPTS = [
    "01_domain_shift_d4.py",
    "02_save_load_artifact.py",
    "03_compositional_d5.py",
    "04_falsification_controls.py",
    "05_yaml_config.py",
    "minimal_loop.py",
    "07_vision_multilayer.py",
    "12_resnet_hook_d4.py",
    "13_compositional_train_d5.py",
    "14_vit_cls_d4.py",
    "15_speech_encoder_d4.py",
    "16_qm9_molecule_d5.py",
    "17_code_tokens_d5.py",
    "18_augmentation_d3.py",
]

# Optional: need extras or longer runtime
OPTIONAL_SCRIPTS = [
    ("06_office31_sklearn.py", []),
    ("08_hf_style_d7.py", []),
    ("09_lightning_module.py", ["lightning"]),
    ("10_hf_trainer.py", ["hf"]),
]


def _run_example(script: str, timeout: int = 120) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PMH_QUICK"] = "1"
    path = EXAMPLES / script
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
def test_example_smoke(script: str) -> None:
    proc = _run_example(script, timeout=120)
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.parametrize("script,extras", OPTIONAL_SCRIPTS)
def test_example_optional(script: str, extras: list[str]) -> None:
    pytest.importorskip("sklearn" if "sklearn" in str(script) else "torch")
    if extras:
        for ext in extras:
            if ext == "lightning":
                from pmh.integrations.lightning import lightning_available

                if not lightning_available():
                    pytest.skip("lightning not installed or broken in this environment")
            if ext == "hf":
                pytest.importorskip("transformers")
    path = EXAMPLES / script
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
