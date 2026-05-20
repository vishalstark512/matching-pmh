"""Thin pmh-train CLI: try, doctor, evaluate, route."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_cli_doctor(capsys):
    from pmh.cli.main import main

    assert main(["doctor", "--stack", "pytorch"]) == 0
    assert "doctor" in capsys.readouterr().out.lower()


def test_cli_route_menu(capsys):
    from pmh.cli.main import main

    assert main(["route"]) == 0
    assert "pose" in capsys.readouterr().out.lower() or "vision" in capsys.readouterr().out.lower()


def test_cli_route_task(capsys):
    from pmh.cli.main import main

    assert main(["route", "--task", "vision_classification", "--quiet"]) == 0
    assert "WHAT CHANGES" in capsys.readouterr().out


def test_cli_route_wizard_non_interactive(capsys):
    from pmh.cli.main import main

    assert main(["route", "--wizard", "--non-interactive", "--stack", "pytorch"]) == 0
    assert "PMHTrainer" in capsys.readouterr().out


def test_cli_route_wizard_requires_stack_when_non_interactive():
    from pmh.cli.main import main

    assert main(["route", "--wizard", "--non-interactive"]) == 2


def test_cli_try_quick(capsys):
    from pmh.cli.main import main

    code = main(["try", "--quick", "--no-falsification"])
    out = capsys.readouterr().out
    assert code in (0, 2)
    assert "Deploy holdout" in out or "deploy holdout" in out.lower()
    assert "SHIP" in out or "ship" in out.lower()


def test_cli_try_sklearn(capsys):
    from pmh.cli.main import main

    code = main([
        "try",
        "--stack",
        "sklearn",
        "--quick",
        "--n-per-domain",
        "60",
        "--no-falsification",
        "--no-coral",
    ])
    out = capsys.readouterr().out
    assert code in (0, 2)
    assert "Suggested" in out or "domain_shift" in out or "subspace" in out


def test_cli_evaluate_demo(capsys):
    from pmh.cli.main import main

    code = main(["evaluate", "--demo", "--n-per-domain", "80", "--no-coral"])
    out = capsys.readouterr().out
    assert code in (0, 2)
    assert "Step 5" in out or "matched" in out
    assert "deploy holdout" in out.lower() or "baseline=" in out


def test_cli_evaluate_npy(tmp_path):
    from pmh.cli.main import main

    n, d = 60, 12
    rng = np.random.default_rng(1)
    xs = rng.standard_normal((n, d)).astype(np.float32)
    xt = xs + 0.1
    y = rng.integers(0, 3, n)
    sp = tmp_path / "s.npy"
    tp = tmp_path / "t.npy"
    yp = tmp_path / "y.npy"
    np.save(sp, xs)
    np.save(tp, xt)
    np.save(yp, y)
    code = main([
        "evaluate",
        "--source-npy",
        str(sp),
        "--target-npy",
        str(tp),
        "--source-labels",
        str(yp),
        "--target-labels",
        str(yp),
        "--no-coral",
        "--no-falsification",
    ])
    assert code == 0


def test_cli_evaluate_pytorch_demo(capsys):
    from pmh.cli.main import main

    code = main([
        "evaluate",
        "--stack",
        "pytorch",
        "--demo",
        "--n-per-domain",
        "80",
        "--epochs",
        "1",
        "--no-falsification",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "Deploy holdout" in out or "ERM baseline" in out


@pytest.mark.skipif(sys.platform.startswith("win"), reason="console script may be uninstalled in editable-only env")
def test_console_script_doctor():
    subprocess.run([sys.executable, "-m", "pmh.cli.main", "doctor"], check=True)


def test_removed_commands_fail():
    from pmh.cli.main import main

    for argv in (["list-methods"], ["estimate"], ["benchmark"]):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2
