"""pmh-train evaluate command and PyTorch eval bundles."""

from __future__ import annotations

import numpy as np


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
        "--source-npy", str(sp),
        "--target-npy", str(tp),
        "--source-labels", str(yp),
        "--target-labels", str(yp),
        "--no-coral",
        "--no-falsification",
    ])
    assert code == 0


def test_cli_evaluate_pytorch_demo(capsys):
    from pmh.cli.main import main

    code = main([
        "evaluate",
        "--stack", "pytorch",
        "--demo",
        "--n-per-domain", "80",
        "--epochs", "1",
        "--no-falsification",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "baseline=" in out or "Target" in out


def test_pytorch_demo_loaders():
    from pmh.pytorch_eval import pytorch_demo_loaders

    b = pytorch_demo_loaders(n=60, batch_size=8, seed=0)
    assert b.d_in == 32
    assert b.n_classes == 2
    assert len(b.val_loader.dataset) > 0


def test_pytorch_eval_bundle_from_arrays():
    from pmh.pytorch_eval import pytorch_eval_bundle_from_arrays

    rng = np.random.default_rng(0)
    n, d = 80, 12
    xs = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 3, n)
    xt = xs + 0.2
    b = pytorch_eval_bundle_from_arrays(xs, y, xt, y.copy(), val_fraction=0.3, seed=0)
    assert b.d_in == d
    assert b.n_classes >= 3


def test_pmh_config_from_preset():
    from pmh.config import PMHConfig
    from pmh.pytorch_eval import pmh_config_from_preset

    assert pmh_config_from_preset("balanced").weight == PMHConfig.balanced().weight
    assert pmh_config_from_preset("conservative").cap_ratio == PMHConfig.conservative().cap_ratio
