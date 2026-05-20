"""pmh-train: estimate Sigma_task and run matched-PMH jobs from JSON configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npy(path: str | Path) -> np.ndarray:
    return np.load(path)


def _cmd_list_methods(_: argparse.Namespace) -> int:
    from pmh.catalog import list_methods
    from pmh.subtypes import format_subtype_line

    print(f"{'Method':<6} {'Name':<28} {'Typical blocks':<20} Required inputs")
    print("-" * 90)
    for spec in list_methods():
        req = ", ".join(spec.required_data) or "(config only)"
        print(f"{spec.method:<6} {spec.name:<28} {spec.typical_tasks:<20} {req}")
        print(f"         {format_subtype_line(spec.method)}")
    print("\nNew here: pmh-train recipe  |  pmh-train shifts  |  pmh-train evaluate --demo")
    print("Paper block presets: pmh-train list-presets")
    print("Use: pmh-train estimate --config job.json")
    print("       pmh-train benchmark --config examples/configs/benchmark_sklearn.json")
    print("Samples: examples/configs/")
    return 0


def _cmd_shifts(_: argparse.Namespace) -> int:
    from pmh import format_shift_types

    print(format_shift_types())
    print("Detail: docs/WHAT_IS_DEPLOYMENT_SHIFT.md")
    print("Pick task: pmh-train route --list")
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    from pmh.task_router import explain_task, format_search_results, format_task_menu

    if getattr(args, "search", None):
        print(format_search_results(args.search))
        return 0
    if args.list:
        print(format_task_menu(short=False))
        return 0
    if args.task is None:
        print(format_task_menu(short=True))
        from pmh.adoption import format_recipe_banner

        print()
        print(format_recipe_banner())
        print("\nExample: pmh-train route --task pose_or_keypoints")
        print("Docs:    docs/FIVE_STEP_RECIPE.md")
        return 0
    try:
        print(explain_task(args.task))
    except KeyError as e:
        print(e, file=sys.stderr)
        return 2
    if not args.quiet:
        print("\nThen: pmh-train wizard --task", args.task)
    return 0


def _cmd_wizard(args: argparse.Namespace) -> int:
    from pmh.onboarding import run_wizard

    if args.non_interactive and args.stack is None and args.task is None:
        print("wizard --non-interactive requires --stack or --task", file=sys.stderr)
        return 2
    run_wizard(
        stack=args.stack,
        task_id=args.task,
        has_target_domain=args.target_domain,
        has_target_labels=args.target_labels,
        has_frozen_features=args.frozen_features,
        has_style_pairs=args.style_pairs,
        interactive=not args.non_interactive,
    )
    return 0


def _cmd_list_presets(_: argparse.Namespace) -> int:
    from pmh.benchmark.presets import PRESETS

    print(f"{'Preset':<22} {'Block':<6} {'Lemma':<5} {'Mode':<10} rank  PMH w/cap")
    print("-" * 78)
    for pid in sorted(PRESETS):
        p = PRESETS[pid]
        w = p.pmh_config.weight
        c = p.pmh_config.cap_ratio
        print(
            f"{pid:<22} {p.paper_type:<6} {p.lemma:<5} {p.application_mode:<10} "
            f"{p.default_rank:<4}  {w:.2f}/{c:.2f}"
        )
    from pmh.benchmark.presets import SUBTYPE_TO_BLOCK_PRESET

    print("\nSubtype defaults:", ", ".join(f"{k}→{v}" for k, v in SUBTYPE_TO_BLOCK_PRESET.items()))
    print("Sklearn: compare_arms_sklearn(..., preset='t1_office31_sklearn')")
    print("         or preset from get_subtype_preset('D1').block_id")
    print("PyTorch: compare_arms(..., preset='t4_domain_d4', include_geometry=True)")
    print("Validate: pmh-train validate -c examples/configs/validate_sklearn_synthetic.json")
    print("Estimate: pmh-train estimate --source-dir A/ --target-dir B/ -o artifacts/run")
    print("Doctor: pmh-train doctor")
    return 0


def _resolve_features(data: dict[str, Any], key: str, alt: str) -> torch.Tensor:
    if key in data:
        v = data[key]
        if isinstance(v, str):
            return torch.from_numpy(_load_npy(v)).float()
        return torch.as_tensor(v).float()
    if alt in data:
        return torch.from_numpy(_load_npy(data[alt])).float()
    raise KeyError(key)


def _estimate_from_job(job: dict[str, Any]) -> Any:
    from pmh.artifact import SigmaTaskEstimate
    from pmh.catalog import config_from_job, validate_job_data
    from pmh.estimate import estimate_from_config
    from pmh.numpy_api import estimate_sigma_task_numpy

    est_block = job.get("estimator", job)
    data = job.get("data", {})
    method = str(est_block.get("method", "D4")).upper()
    missing = validate_job_data(method, data)
    if missing:
        raise ValueError(f"job data missing for {method}: {missing}")

    cfg = config_from_job(est_block)

    if method == "D2":
        dim = int(data.get("dim", data.get("representation_dim")))
        cfg.dim = dim
        cfg.noise_level = float(data.get("noise_level", cfg.noise_level or 0.1))
        artifact = estimate_from_config(cfg)

    elif method == "D3":
        aug = _resolve_features(data, "aug_deltas", "aug_npy")
        artifact = estimate_from_config(cfg, aug_deltas=aug)

    elif method == "D4":
        src = _resolve_features(data, "source_features", "source_npy")
        tgt = _resolve_features(data, "target_features", "target_npy")
        artifact = estimate_from_config(cfg, src, tgt)

    elif method == "D1":
        if all(k in data for k in ("source_labels", "target_labels")):
            xs = _load_npy(data.get("source_npy", data["source_features"]))
            ys = np.load(data["source_labels"]) if isinstance(data["source_labels"], str) else np.asarray(
                data["source_labels"]
            )
            xt = _load_npy(data.get("target_npy", data["target_features"]))
            yt = np.load(data["target_labels"]) if isinstance(data["target_labels"], str) else np.asarray(
                data["target_labels"]
            )
            artifact = estimate_sigma_task_numpy(xs, ys, xt, yt, config=cfg)
        else:
            src = _resolve_features(data, "source_features", "source_npy")
            tgt = _resolve_features(data, "target_features", "target_npy")
            artifact = estimate_from_config(cfg, src, tgt)

    elif method == "D5":
        feats = _resolve_features(data, "features", "features_npy")
        if cfg.nuisance_indices is None:
            raise ValueError("D5 job needs estimator.nuisance_indices")
        artifact = estimate_from_config(cfg, feats)

    elif method == "D6":
        raise ValueError("D6 benchmark via CLI not supported; use run_benchmark_protocol in Python")

    elif method == "D7":
        raise ValueError("D7 estimate via CLI needs style JSONL — see configs/d7_style_estimate.json")

    else:
        raise ValueError(method)

    out = job.get("output", "artifacts/sigma_task")
    if isinstance(out, str):
        artifact.save(out)
        print(f"saved {out}")
    return artifact


def _estimate_from_dirs(
    source_dir: Path,
    target_dir: Path,
    *,
    method: str,
    rank: int,
    output: str,
) -> Any:
    from pmh.config import SigmaTaskConfig
    from pmh.data_adapters import load_domain_dirs
    from pmh.numpy_api import estimate_sigma_task_numpy

    xs, ys, xt, yt = load_domain_dirs(source_dir, target_dir)
    m = method.upper()
    if m == "D1":
        if ys is None or yt is None:
            raise ValueError("D1 estimate needs labels.npy (or y.npy) in both domain folders")
        art = estimate_sigma_task_numpy(
            xs, ys, xt, yt, config=SigmaTaskConfig.for_subspace(rank=rank)
        )
    else:
        art = estimate_sigma_task_numpy(xs, xt, config=SigmaTaskConfig.for_domain(rank=rank))
    art.save(output)
    print(f"saved {output}")
    return art


def _cmd_estimate(args: argparse.Namespace) -> int:
    if args.config:
        artifact = _estimate_from_job(_load_json(Path(args.config)))
    elif getattr(args, "source_dir", None) and getattr(args, "target_dir", None):
        artifact = _estimate_from_dirs(
            Path(args.source_dir),
            Path(args.target_dir),
            method=args.method,
            rank=args.rank,
            output=args.output,
        )
    elif getattr(args, "source_npy", None) and getattr(args, "target_npy", None):
        job: dict[str, Any] = {
            "estimator": {"method": args.method, "rank": args.rank, "shrinkage": 1e-6},
            "data": {
                "source_npy": str(args.source_npy),
                "target_npy": str(args.target_npy),
            },
            "output": args.output,
        }
        if getattr(args, "source_labels", None):
            job["data"]["source_labels"] = str(args.source_labels)
        if getattr(args, "target_labels", None):
            job["data"]["target_labels"] = str(args.target_labels)
        artifact = _estimate_from_job(job)
    else:
        print(
            "estimate needs --config, or --source-npy + --target-npy, "
            "or --source-dir + --target-dir",
            file=sys.stderr,
        )
        return 2
    print(f"  method={artifact.method}  dim={artifact.dim}  preflight={artifact.preflight}")
    if artifact.eigengap is not None:
        print(f"  eigengap={artifact.eigengap:.4f}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from pmh.doctor import run_doctor

    artifact = str(args.artifact) if getattr(args, "artifact", None) else None
    rep = run_doctor(
        stack=args.stack,
        artifact_path=artifact,
        rank=getattr(args, "rank", None),
    )
    print(rep.summary())
    return 0 if rep.ok else 1


def _cmd_preflight(args: argparse.Namespace) -> int:
    from pmh.artifact import SigmaTaskEstimate
    from pmh.diagnostics import eigengap_ratio
    from pmh.preflight import preflight_eigengap

    art = SigmaTaskEstimate.load(args.artifact)
    rank = args.rank or art.config.rank or 1
    gamma = eigengap_ratio(art.sigma, rank)
    status, stored_gap = preflight_eigengap(art.sigma, rank)
    print(f"artifact={args.artifact}  method={art.method}  rank={rank}  gamma_r={gamma:.4f}")
    print(f"preflight={status.value}  stored={art.preflight}  stored_eigengap={art.eigengap}")
    if stored_gap is not None:
        print(f"recomputed_eigengap={stored_gap:.4f}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from pmh.benchmark import benchmark_to_markdown, load_benchmark_report, write_benchmark_report

    if args.report:
        data = load_benchmark_report(args.report)
        print(benchmark_to_markdown(data))
        return 0

    if not args.config:
        print("benchmark needs --config job.json or --report results/benchmark/benchmark.json", file=sys.stderr)
        return 1

    job = _load_json(Path(args.config))
    protocol = job.get("protocol", "sklearn")
    out = Path(job.get("output", "results/benchmark"))
    arms = job.get("arms")

    if protocol == "sklearn":
        from pmh.benchmark.sklearn_protocol import run_sklearn_benchmark, synthetic_office31_features

        data = job.get("data", {})
        if data.get("mode") == "synthetic_office31":
            n = int(data.get("n_per_domain", 400))
            seed = int(data.get("seed", 0))
            x_a, y, x_d, y2 = synthetic_office31_features(n, seed=seed)
        else:
            x_a = _load_npy(data["source_npy"])
            y = np.load(data["source_labels"]) if isinstance(data["source_labels"], str) else np.asarray(
                data["source_labels"]
            )
            x_d = _load_npy(data["target_npy"])
            y2 = (
                np.load(data["target_labels"])
                if isinstance(data["target_labels"], str)
                else np.asarray(data["target_labels"])
            )
        rank = int(job.get("estimator", {}).get("rank", 16))
        want_coral = arms is None or "coral" in [str(a).lower() for a in arms]
        result = run_sklearn_benchmark(x_a, y, x_d, y2, rank=rank, include_coral=want_coral)
    else:
        print(
            "PyTorch multi-arm compare: python examples/20_compare_training_arms.py",
            file=sys.stderr,
        )
        return 1

    paths = write_benchmark_report(result, out)
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")
    for arm, row in result.arms.items():
        print(f"  {arm:10s}  {row.val_metric:.4f}  ({row.metric_name})")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Run sklearn falsification benchmark and exit 0 only if matched wins controls."""
    from pmh.benchmark.validate import validate_falsification
    from pmh.benchmark.report import write_benchmark_report
    from pmh.benchmark.sklearn_protocol import run_sklearn_benchmark, synthetic_office31_features

    if args.config:
        job = _load_json(Path(args.config))
        out = Path(job.get("output", "results/validate"))
        protocol = job.get("protocol", "sklearn")
        if protocol == "pytorch_smoke":
            from pmh.benchmark.pytorch_smoke import run_pytorch_benchmark_smoke

            est = job.get("estimator", {})
            result = run_pytorch_benchmark_smoke(
                rank=int(est.get("rank", 8)),
                epochs=est.get("epochs"),
            )
        else:
            data = job.get("data", {})
            if data.get("mode") == "synthetic_office31":
                n = int(data.get("n_per_domain", 400))
                seed = int(data.get("seed", 0))
                x_a, y, x_d, y2 = synthetic_office31_features(n, seed=seed)
            else:
                x_a = _load_npy(data["source_npy"])
                y = (
                    np.load(data["source_labels"])
                    if isinstance(data["source_labels"], str)
                    else np.asarray(data["source_labels"])
                )
                x_d = _load_npy(data["target_npy"])
                y2 = (
                    np.load(data["target_labels"])
                    if isinstance(data["target_labels"], str)
                    else np.asarray(data["target_labels"])
                )
            est = job.get("estimator", {})
            rank = int(est.get("rank", 16))
            preset = est.get("preset")
            bench_kw: dict[str, Any] = {"rank": rank, "include_coral": False}
            if preset:
                from pmh.benchmark.presets import get_preset, get_subtype_preset

                try:
                    p = get_subtype_preset(str(preset))
                except KeyError:
                    p = get_preset(str(preset))
                bench_kw.update(p.sklearn_benchmark)
                bench_kw.setdefault("rank", p.default_rank)
            result = run_sklearn_benchmark(x_a, y, x_d, y2, **bench_kw)
    elif args.report:
        from pmh.benchmark.report import load_benchmark_report

        data = load_benchmark_report(args.report)
        rep = validate_falsification(data, min_margin=args.min_margin)
        print(rep.summary())
        return 0 if rep.passed else 1
    else:
        print("validate needs --config or --report", file=sys.stderr)
        return 1

    paths = write_benchmark_report(result, out, stem="validate")
    rep = validate_falsification(result, min_margin=args.min_margin)
    print(rep.summary())
    print(f"Report: {paths['json']}")
    return 0 if rep.passed else 1


def _cmd_run(args: argparse.Namespace) -> int:
    job = _load_json(Path(args.config))
    pmh_block = job.get("pmh", {})
    artifact_path = job.get("artifact") or pmh_block.get("artifact")
    if not artifact_path:
        print("run job needs 'artifact' path to saved Sigma_task (.pt)", file=sys.stderr)
        return 1
    from pmh.artifact import SigmaTaskEstimate
    from pmh.config import PMHConfig

    art = SigmaTaskEstimate.load(artifact_path)
    pcfg = PMHConfig.from_dict(pmh_block) if pmh_block else PMHConfig()
    print("Matched PMH training recipe")
    print(f"  artifact: {artifact_path}")
    print(f"  method:   {art.method}  dim={art.dim}  preflight={art.preflight}")
    print(f"  weight:   {pcfg.weight}  cap_ratio={pcfg.cap_ratio}  warmup_epochs={pcfg.warmup_epochs}")
    print("  controls: train also with mode=wrong_w and mode=isotropic (see pmh.benchmark)")
    train = job.get("training", {})
    if train.get("backend") == "hf_trainer":
        print("  backend:  Hugging Face PMHTrainer (see examples/11_dpo_lora_style_pmh.py)")
    elif train.get("backend") == "lightning":
        print("  backend:  Lightning (see examples/09_lightning_module.py)")
    else:
        print("  backend:  PMHLoss — see examples/20_compare_training_arms.py")
    if args.dry_run:
        print("(dry-run: no training executed)")
    return 0


def _cmd_recipe(args: argparse.Namespace) -> int:
    from pmh.recipe import format_five_step_guide, plan_recipe

    print(
        format_five_step_guide(
            task_id=getattr(args, "task", None),
            stack=getattr(args, "stack", None),
        )
    )
    if getattr(args, "identify", False):
        flags = {
            "has_source_labels": args.source_labels,
            "has_target_labels": args.target_labels,
            "has_target_domain": args.target_domain,
            "has_augmentation_modes": args.augmentation_modes,
            "has_style_pairs": args.style_pairs,
            "has_temporal_sequences": args.temporal,
            "has_nuisance_indices": args.compositional,
            "noise_level_known": args.isotropic_noise,
        }
        shift = plan_recipe(
            stack=args.stack or "pytorch",
            identify_flags=flags,
        ).shift
        print()
        print("Identification (Step 1):")
        print(f"  {shift.assumption} / {shift.method} → nuisance={shift.nuisance!r}")
        print(f"  {shift.title}: {shift.reason}")
        print(f"  application_mode={shift.application_mode!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from pmh.adoption import RECIPE_ONE_LINER

    parser = argparse.ArgumentParser(
        prog="pmh-train",
        description=f"Matched PMH — {RECIPE_ONE_LINER}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_recipe = sub.add_parser(
        "recipe",
        help="Print the 5-step matching-principle recipe (paper §7)",
    )
    p_recipe.add_argument("--task", default=None, help="Append task profile from route")
    p_recipe.add_argument(
        "--stack",
        choices=("pytorch", "sklearn", "hf"),
        default=None,
        help="Show suggested nuisance + mode for stack",
    )
    p_recipe.add_argument(
        "--identify",
        action="store_true",
        help="Run Step 1 from flags below",
    )
    p_recipe.add_argument("--source-labels", action="store_true", default=True)
    p_recipe.add_argument("--no-source-labels", action="store_false", dest="source_labels")
    p_recipe.add_argument("--target-labels", action="store_true", default=False)
    p_recipe.add_argument("--target-domain", action="store_true", default=True)
    p_recipe.add_argument("--no-target-domain", action="store_false", dest="target_domain")
    p_recipe.add_argument("--augmentation-modes", action="store_true", default=False)
    p_recipe.add_argument("--style-pairs", action="store_true", default=False)
    p_recipe.add_argument("--temporal", action="store_true", default=False)
    p_recipe.add_argument("--compositional", action="store_true", default=False)
    p_recipe.add_argument("--isotropic-noise", action="store_true", default=False)
    p_recipe.set_defaults(func=_cmd_recipe)

    p_shifts = sub.add_parser(
        "shifts",
        help="Plain English: what deployment shift types exist (nuisance= API keys)",
    )
    p_shifts.set_defaults(func=_cmd_shifts)

    p_list = sub.add_parser("list-methods", help="List estimator IDs D1--D7 (technical)")
    p_list.set_defaults(func=_cmd_list_methods)

    p_presets = sub.add_parser(
        "list-presets",
        help="List paper block presets (T1 Office-31, T4 domain, T7A style, …)",
    )
    p_presets.set_defaults(func=_cmd_list_presets)

    p_route = sub.add_parser(
        "route",
        help="Route your ML task to one PMH path (pose, vision, LLM, …)",
    )
    p_route.add_argument(
        "--task",
        default=None,
        help="Task id, e.g. pose_or_keypoints, vision_classification",
    )
    p_route.add_argument(
        "--search",
        default=None,
        metavar="KEYWORD",
        help="Find applications by keyword (pose, hospital, blur, temporal, …)",
    )
    p_route.add_argument("--list", action="store_true", help="Print task menu")
    p_route.add_argument("--quiet", action="store_true", help="No follow-up hints")
    p_route.set_defaults(func=_cmd_route)

    p_wiz = sub.add_parser(
        "wizard",
        help="Interactive setup guide (stack, data, copy-paste snippet)",
    )
    p_wiz.add_argument(
        "--stack",
        choices=("pytorch", "sklearn", "hf"),
        default=None,
        help="Skip questionnaire when set",
    )
    p_wiz.add_argument("--target-domain", action="store_true", default=None)
    p_wiz.add_argument("--no-target-domain", action="store_false", dest="target_domain")
    p_wiz.add_argument("--target-labels", action="store_true", default=None)
    p_wiz.add_argument("--frozen-features", action="store_true", default=None)
    p_wiz.add_argument("--style-pairs", action="store_true", default=None)
    p_wiz.add_argument(
        "--task",
        default=None,
        help="Task id from pmh-train route --list (e.g. pose_or_keypoints)",
    )
    p_wiz.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use flags only (requires --stack or --task)",
    )
    p_wiz.set_defaults(func=_cmd_wizard, target_domain=True)

    p_est = sub.add_parser("estimate", help="Estimate Sigma_task (JSON job or .npy / folders)")
    p_est.add_argument("--config", "-c", type=Path, default=None)
    p_est.add_argument("--source-npy", type=Path, default=None, help="Source features [N,d]")
    p_est.add_argument("--target-npy", type=Path, default=None, help="Target features [M,d]")
    p_est.add_argument("--source-labels", type=Path, default=None)
    p_est.add_argument("--target-labels", type=Path, default=None)
    p_est.add_argument("--source-dir", type=Path, default=None, help="Folder with features.npy")
    p_est.add_argument("--target-dir", type=Path, default=None, help="Folder with features.npy")
    p_est.add_argument("--method", default="D4", help="D1--D7 (default D4)")
    p_est.add_argument("--rank", type=int, default=16)
    p_est.add_argument("--output", "-o", default="artifacts/sigma_task", help="Save path stem")
    p_est.set_defaults(func=_cmd_estimate)

    p_doc = sub.add_parser("doctor", help="Check install and optional extras (G1/G1b/G2/G3)")
    p_doc.add_argument(
        "--stack",
        choices=("pytorch", "sklearn", "hf"),
        default="pytorch",
    )
    p_doc.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Optional saved SigmaTaskEstimate — report preflight before Step 5",
    )
    p_doc.add_argument("--rank", type=int, default=None, help="Rank for preflight when using --artifact")
    p_doc.set_defaults(func=_cmd_doctor)

    from pmh.cli.evaluate import add_evaluate_parser

    add_evaluate_parser(sub)

    p_pf = sub.add_parser("preflight", help="Eigengap for saved artifact")
    p_pf.add_argument("artifact", type=Path)
    p_pf.add_argument("--rank", type=int, default=None)
    p_pf.set_defaults(func=_cmd_preflight)

    p_bench = sub.add_parser(
        "benchmark",
        help="Run B0/matched/wrong-W/isotropic comparison (sklearn) or print saved report",
    )
    p_bench.add_argument("--config", "-c", type=Path, default=None, help="benchmark job JSON")
    p_bench.add_argument(
        "--report",
        "-r",
        type=Path,
        default=None,
        help="Print markdown table from saved benchmark.json",
    )
    p_bench.set_defaults(func=_cmd_benchmark)

    p_val = sub.add_parser(
        "validate",
        help="Falsification pass/fail (sklearn arms); exit 1 if matched loses to wrong-W/isotropic",
    )
    p_val.add_argument("--config", "-c", type=Path, default=None, help="benchmark-style JSON job")
    p_val.add_argument(
        "--report",
        "-r",
        type=Path,
        default=None,
        help="Validate existing benchmark/validate.json",
    )
    p_val.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Required accuracy gap matched minus control",
    )
    p_val.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="Validate training job JSON")
    p_run.add_argument("--config", "-c", required=True, type=Path)
    p_run.add_argument("--dry-run", action="store_true", default=True)
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
