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

    print(f"{'Method':<6} {'Name':<28} {'Typical blocks':<20} Required inputs")
    print("-" * 90)
    for spec in list_methods():
        req = ", ".join(spec.required_data) or "(config only)"
        print(f"{spec.method:<6} {spec.name:<28} {spec.typical_tasks:<20} {req}")
    print("\nUse: pmh-train estimate --config job.json")
    print("Samples: examples/configs/")
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
    from pmh.config import SigmaTaskConfig
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
        artifact = estimate_from_config(cfg, aug)

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
        cfg.nuisance_indices = list(data["nuisance_indices"])
        artifact = estimate_from_config(cfg, feats)

    elif method == "D6":
        if "sequences_npy" in data:
            seq = torch.from_numpy(_load_npy(data["sequences_npy"]))
        else:
            seq = torch.as_tensor(data["sequences"])
        artifact = estimate_from_config(cfg, seq)

    elif method == "D7":
        if "deltas_npy" in data:
            deltas = torch.from_numpy(_load_npy(data["deltas_npy"]))
            artifact = estimate_from_config(cfg, deltas)
        else:
            from pmh.integrations.huggingface import estimate_style_sigma, load_style_pairs_jsonl

            path = Path(data["style_jsonl"])
            pairs = load_style_pairs_jsonl(path, max_pairs=data.get("max_pairs"))
            model_id = data.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")
            import os

            os.environ.setdefault("USE_TF", "0")
            os.environ.setdefault("USE_FLAX", "0")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            artifact = estimate_style_sigma(
                pairs,
                model,
                tokenizer,
                rank=int(cfg.rank or 128),
                batch_size=int(data.get("batch_size", 4)),
                max_length=int(data.get("max_length", 512)),
            )
    else:
        raise ValueError(method)

    assert isinstance(artifact, SigmaTaskEstimate)
    return artifact


def _cmd_estimate(args: argparse.Namespace) -> int:
    job = _load_json(Path(args.config))
    artifact = _estimate_from_job(job)
    out = Path(job.get("output", "sigma_task"))
    pt = artifact.save(out)
    print(f"Saved {pt}")
    print(f"  method={artifact.method}  dim={artifact.dim}  preflight={artifact.preflight}")
    if artifact.eigengap is not None:
        print(f"  eigengap={artifact.eigengap:.4f}")
    return 0


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


def _cmd_run(args: argparse.Namespace) -> int:
    """Validate a training job JSON and print the recipe (integrate in your trainer)."""
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
    train = job.get("training", {})
    if train.get("backend") == "hf_trainer":
        print("  backend:  Hugging Face PMHTrainer (see examples/11_dpo_lora_style_pmh.py)")
    elif train.get("backend") == "lightning":
        print("  backend:  Lightning + add_pmh_to_loss (see examples/09_lightning_module.py)")
    else:
        print("  backend:  PMHLoss on backbone(h) — see examples/01_domain_shift_d4.py")
    if args.dry_run:
        print("(dry-run: no training executed)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pmh-train",
        description="Estimate Sigma_task (D1--D7) and inspect matched-PMH training jobs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-methods", help="List nuisance types D1--D7 and required inputs")
    p_list.set_defaults(func=_cmd_list_methods)

    p_est = sub.add_parser("estimate", help="Estimate Sigma_task from a JSON job file")
    p_est.add_argument("--config", "-c", required=True, type=Path, help="Job JSON path")
    p_est.set_defaults(func=_cmd_estimate)

    p_pf = sub.add_parser("preflight", help="Show eigengap for a saved artifact")
    p_pf.add_argument("artifact", type=Path, help=".pt artifact path")
    p_pf.add_argument("--rank", type=int, default=None)
    p_pf.set_defaults(func=_cmd_preflight)

    p_run = sub.add_parser("run", help="Validate training job JSON and print recipe")
    p_run.add_argument("--config", "-c", required=True, type=Path)
    p_run.add_argument("--dry-run", action="store_true", default=True)
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
