"""``pmh-train evaluate`` — Step 5 report (sklearn arrays or PyTorch loaders)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_sklearn_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if args.demo:
        from pmh import load_g2_demo_arrays

        return load_g2_demo_arrays(n=args.n_per_domain, seed=args.seed)

    if args.source_npy and args.target_npy:
        x_src = np.load(args.source_npy)
        x_tgt = np.load(args.target_npy)
        y_src = (
            np.load(args.source_labels)
            if args.source_labels
            else np.zeros(len(x_src), dtype=np.int64)
        )
        y_tgt = (
            np.load(args.target_labels)
            if args.target_labels
            else np.zeros(len(x_tgt), dtype=np.int64)
        )
        return x_src, y_src, x_tgt, y_tgt

    if args.source_dir and args.target_dir:
        from pmh.data_adapters import load_domain_dirs

        xs, ys, xt, yt = load_domain_dirs(args.source_dir, args.target_dir)
        if ys is None or yt is None:
            raise SystemExit(
                "evaluate: need labels in domain folders or pass --source-labels / --target-labels"
            )
        return xs, ys, xt, yt

    raise SystemExit(_usage_sklearn())


def _usage_sklearn() -> str:
    return (
        "sklearn evaluate: --demo, or --source-npy/--target-npy (+ labels), "
        "or --source-dir/--target-dir"
    )


def _usage_pytorch() -> str:
    return (
        "pytorch evaluate: --demo, or --source-dir/--target-dir (features.npy + labels.npy), "
        "or --source-npy/--target-npy with labels"
    )


def _run_sklearn(args: argparse.Namespace) -> int:
    from pmh import check_applicability, evaluate_baseline_vs_pmh

    x_src, y_src, x_tgt, y_tgt = _load_sklearn_arrays(args)
    app = check_applicability(
        stack="sklearn",
        n_source=len(x_src),
        n_target=len(x_tgt),
        feature_dim=x_src.shape[1],
        has_target_labels=True,
    )
    print(app.summary())
    print()
    report = evaluate_baseline_vs_pmh(
        x_source=x_src,
        y_source=y_src,
        x_target=x_tgt,
        y_target=y_tgt,
        rank=args.rank,
        seed=args.seed,
        test_size=args.test_size,
        nuisance=args.nuisance,
        compare_to=() if args.no_coral else ("coral",),
        include_falsification=not args.no_falsification,
    )
    print(report.summary())
    return 0 if (report.step5_ok() is not False) else 2


def _run_pytorch(args: argparse.Namespace) -> int:
    from pmh import check_applicability, evaluate_robust_fit
    from pmh.pytorch_eval import (
        pmh_config_from_preset,
        pytorch_demo_loaders,
        pytorch_eval_bundle_from_arrays,
    )

    if args.demo:
        bundle = pytorch_demo_loaders(n=args.n_per_domain, batch_size=args.batch_size, seed=args.seed)
    elif (args.source_dir and args.target_dir) or (args.source_npy and args.target_npy):
        if args.source_dir and args.target_dir:
            from pmh.data_adapters import load_domain_dirs

            xs, ys, xt, yt = load_domain_dirs(args.source_dir, args.target_dir)
            if ys is None or yt is None:
                raise SystemExit("pytorch evaluate: need labels.npy in both domain folders")
        else:
            xs = np.load(args.source_npy)
            xt = np.load(args.target_npy)
            ys = np.load(args.source_labels) if args.source_labels else None
            yt = np.load(args.target_labels) if args.target_labels else None
            if ys is None or yt is None:
                raise SystemExit("pytorch evaluate: pass --source-labels and --target-labels with .npy")
        bundle = pytorch_eval_bundle_from_arrays(
            xs, ys, xt, yt,
            val_fraction=args.test_size,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    else:
        raise SystemExit(_usage_pytorch())

    pmh_cfg = pmh_config_from_preset(args.pmh_preset)
    if args.weight is not None or args.cap_ratio is not None:
        from pmh.config import PMHConfig

        d = pmh_cfg.to_dict()
        if args.weight is not None:
            d["weight"] = args.weight
        if args.cap_ratio is not None:
            d["cap_ratio"] = args.cap_ratio
        pmh_cfg = PMHConfig.from_dict(d)

    app = check_applicability(
        stack="pytorch",
        n_source=len(bundle.source_batches.dataset),
        n_target=len(bundle.target_batches.dataset),
        feature_dim=bundle.d_in,
        has_target_labels=True,
    )
    print(app.summary())
    print(f"  model: MLP d_in={bundle.d_in}  classes={bundle.n_classes}")
    print()

    report = evaluate_robust_fit(
        bundle.model,
        bundle.train_loader,
        bundle.val_loader,
        source_batches=bundle.source_batches,
        target_batches=bundle.target_batches,
        hook=bundle.encoder,
        head=bundle.head,
        epochs=args.epochs,
        rank=args.rank,
        nuisance=args.nuisance,
        pmh_config=pmh_cfg,
        seed=args.seed,
        include_falsification=not args.no_falsification,
        falsification_test_size=args.test_size,
    )
    print(report.summary())
    return 0 if (report.step5_ok() is not False) else 2


def run_evaluate(args: argparse.Namespace) -> int:
    from pmh.adoption import format_recipe_banner

    print(format_recipe_banner())
    print()

    stack = getattr(args, "stack", "sklearn")
    if stack == "pytorch":
        return _run_pytorch(args)
    return _run_sklearn(args)


def add_evaluate_parser(sub) -> None:
    p = sub.add_parser(
        "evaluate",
        help="Step 5 report: sklearn (default) or PyTorch (--stack pytorch)",
    )
    p.add_argument(
        "--stack",
        choices=("sklearn", "pytorch"),
        default="sklearn",
        help="sklearn: frozen features; pytorch: MLP on features or synthetic demo",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Synthetic demo (Office-31-style for sklearn, domain-shift for pytorch)",
    )
    p.add_argument("--source-npy", type=Path, default=None)
    p.add_argument("--target-npy", type=Path, default=None)
    p.add_argument("--source-labels", type=Path, default=None)
    p.add_argument("--target-labels", type=Path, default=None)
    p.add_argument("--source-dir", type=Path, default=None, help="Folder with features.npy (+ labels)")
    p.add_argument("--target-dir", type=Path, default=None)
    p.add_argument("--rank", type=int, default=16, help="Subspace rank (geometry estimate)")
    p.add_argument("--nuisance", default=None, help="Nuisance key (default: auto)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.35, dest="test_size")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs (pytorch only)")
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size", help="Batch size (pytorch)")
    p.add_argument(
        "--pmh-preset",
        choices=("conservative", "balanced", "aggressive", "finetune_llm"),
        default="balanced",
        dest="pmh_preset",
        help="PMHConfig preset (pytorch only)",
    )
    p.add_argument("--weight", type=float, default=None, help="Override PMHConfig.weight")
    p.add_argument("--cap-ratio", type=float, default=None, dest="cap_ratio", help="Override cap_ratio")
    p.add_argument("--no-falsification", action="store_true", help="Skip Step 5 arms (faster)")
    p.add_argument("--no-coral", action="store_true", help="Skip CORAL baseline (sklearn)")
    p.add_argument(
        "--n-per-domain",
        type=int,
        default=500,
        dest="n_per_domain",
        help="Samples per domain when --demo",
    )
    p.set_defaults(func=run_evaluate)
