"""``pmh-train try`` — golden path: one command, ship / don't ship report."""

from __future__ import annotations

import argparse
import os


def _maybe_save_html(args: argparse.Namespace, report) -> None:
    if not getattr(args, "html", None):
        return
    path = report.save_html(args.html)
    print(f"HTML report: {path}")


def _quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    if args.n_per_domain > 200:
        args.n_per_domain = 120
    if args.epochs > 2:
        args.epochs = 2
    if args.stack == "pytorch" and args.max_steps > 4:
        args.max_steps = 4
    if args.stack == "multilayer" and args.max_steps > 4:
        args.max_steps = 4


def _run_sklearn_try(args: argparse.Namespace) -> int:
    from pmh import evaluate_baseline_vs_pmh, infer_applicability, load_g2_demo_arrays

    xs, ys, xt, yt = load_g2_demo_arrays(n=args.n_per_domain, seed=args.seed)
    app = infer_applicability(
        stack="sklearn",
        n_source=len(xs),
        n_target=len(xt),
        feature_dim=xs.shape[1],
        has_target_labels=True,
    )
    print(app.summary())
    print()
    report = evaluate_baseline_vs_pmh(
        x_source=xs,
        y_source=ys,
        x_target=xt,
        y_target=yt,
        rank=args.rank,
        seed=args.seed,
        nuisance=args.nuisance,
        compare_to=() if args.no_coral else ("coral",),
        include_falsification=not args.no_falsification,
    )
    print(report.deploy_summary())
    print()
    print(report.ship_verdict())
    _maybe_save_html(args, report)
    return 0 if report.step5_ok() is not False else 2


def _run_pytorch_try(args: argparse.Namespace) -> int:
    from pmh import infer_applicability, try_pmh
    from pmh.pytorch_eval import pytorch_demo_loaders

    bundle = pytorch_demo_loaders(
        n=args.n_per_domain,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    app = infer_applicability(
        stack="pytorch",
        has_target_domain=True,
        has_target_labels=False,
    )
    print(app.summary())
    print(f"  demo MLP: d_in={bundle.d_in}  classes={bundle.n_classes}")
    print()
    report = try_pmh(
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
        max_steps_per_epoch=args.max_steps if args.max_steps > 0 else None,
        include_falsification=not args.no_falsification,
    )
    print(report.deploy_summary())
    print()
    print(report.ship_verdict())
    _maybe_save_html(args, report)
    return 0 if report.step5_ok() is not False else 2


def _run_multilayer_try(args: argparse.Namespace) -> int:
    """T4B-style RGB CNN feature-diff demo (not tabular)."""
    import torch

    from pmh import PMHConfig, PMHTrainer
    from pmh.developer import RobustFitResult, check_applicability, evaluate_robust_fit
    from pmh.pytorch_eval import pytorch_multilayer_vision_demo_loaders

    torch.manual_seed(args.seed)
    bundle = pytorch_multilayer_vision_demo_loaders(
        n=args.n_per_domain,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    m = bundle.model.to(device)
    layer_names = ("conv1", "conv2")
    app = check_applicability(has_target_domain=True)
    print(app.summary())
    print(f"  RGB CNN multilayer demo, classes={bundle.n_classes}")
    print()
    trainer = PMHTrainer(
        m,
        hook=bundle.encoder,
        head=m.head,
        nuisance="domain_shift",
        rank=args.rank,
        pmh_config=PMHConfig.golden_path(),
        train_mode="feature_diff",
        forward_features=m.forward_features,
        layer_names=layer_names,
        head_layer="conv2",
        device=device,
    )
    trainer.estimate_multilayer(
        bundle.source_batches,
        bundle.target_batches,
        max_batches=4 if args.quick else 20,
    )
    stats = trainer.fit(
        bundle.train_loader,
        source_batches=bundle.source_batches,
        target_batches=bundle.target_batches,
        epochs=args.epochs,
        max_steps_per_epoch=args.max_steps if args.max_steps > 0 else None,
        reestimate=False,
    )
    print("train steps:", int(stats.get("n_steps", 0)))
    report = evaluate_robust_fit(
        m,
        bundle.train_loader,
        bundle.val_loader,
        source_batches=bundle.source_batches,
        target_batches=bundle.target_batches,
        hook=bundle.encoder,
        head=m.head,
        epochs=1,
        pmh_result=RobustFitResult(
            trainer=trainer,
            stats=stats,
            applicability=app,
            hook_used=bundle.encoder,
            preflight=trainer.artifact_.preflight if trainer.artifact_ else None,
        ),
        include_falsification=not args.no_falsification,
        max_steps_per_epoch=args.max_steps if args.max_steps > 0 else None,
    )
    print(report.deploy_summary())
    print()
    print(report.ship_verdict())
    _maybe_save_html(args, report)
    return 0 if report.step5_ok() is not False else 2


def run_try(args: argparse.Namespace) -> int:
    from pmh.adoption import format_recipe_banner

    if args.quick:
        os.environ.setdefault("PMH_QUICK", "1")

    _quick_defaults(args)
    print(format_recipe_banner())
    print("pmh-train try — auto shift type, deploy holdout, ship verdict")
    print()

    if args.stack == "sklearn":
        return _run_sklearn_try(args)
    if args.stack == "multilayer":
        return _run_multilayer_try(args)
    return _run_pytorch_try(args)


def add_try_parser(sub) -> None:
    p = sub.add_parser(
        "try",
        help="Golden path: demo train + Step 5 deploy report (ship / don't ship)",
    )
    p.add_argument(
        "--stack",
        choices=("pytorch", "sklearn", "multilayer"),
        default="pytorch",
        help="pytorch: tabular domain MLP; sklearn: frozen features; multilayer: RGB CNN T4B",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Small data, few steps (~1 min CPU)",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--nuisance", default=None, help="Leave unset to auto-pick from data flags")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument("--n-per-domain", type=int, default=400, dest="n_per_domain")
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        dest="max_steps",
        help="Cap steps per epoch (pytorch/multilayer); 0 = no cap",
    )
    p.add_argument("--no-falsification", action="store_true", help="Skip control arms (faster)")
    p.add_argument("--no-coral", action="store_true", help="Skip CORAL (sklearn only)")
    p.add_argument("--cpu", action="store_true", help="Force CPU (multilayer demo)")
    p.add_argument(
        "--html",
        metavar="PATH",
        default=None,
        help="Write deploy Step 5 report as HTML (e.g. report.html)",
    )
    p.set_defaults(func=run_try)
