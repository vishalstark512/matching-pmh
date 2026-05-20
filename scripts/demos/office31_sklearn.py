#!/usr/bin/env python3
"""Classical domain shift with PMHMatcher + compare_arms_sklearn (synthetic or Office-31)."""

from __future__ import annotations

import argparse

import numpy as np


def _synthetic(n: int = 400, d: int = 512, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((d, 32)).astype(np.float32)
    q, _ = np.linalg.qr(w)
    x_a = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 31, n)
    nuisance = (x_a @ q) @ q.T
    x_d = x_a + 1.5 * nuisance + 0.05 * rng.standard_normal((n, d)).astype(np.float32)
    return x_a, y, x_d, y.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--office31-root", type=str, default=None)
    parser.add_argument("--source", type=str, default="amazon")
    parser.add_argument("--target", type=str, default="dslr")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--report", type=str, default=None, help="Write benchmark.json/md here")
    args = parser.parse_args()

    if args.office31_root:
        from pmh.datasets.office31 import extract_office31_features

        print(f"Office-31: {args.source} -> {args.target}")
        x_src, y_src = extract_office31_features(
            args.office31_root, args.source, max_samples=args.max_samples, seed=args.seed
        )
        x_tgt, y_tgt = extract_office31_features(
            args.office31_root, args.target, max_samples=args.max_samples, seed=args.seed + 1
        )
    else:
        print("Synthetic features (use --office31-root for real Office-31).")
        x_src, y_src, x_tgt, y_tgt = _synthetic(seed=args.seed)

    from pmh import PMHMatcher, compare_arms_sklearn, suggest_nuisance

    sug = suggest_nuisance(
        has_source_labels=True,
        has_target_labels=True,
        has_target_domain=True,
    )
    print(f"suggest_nuisance -> {sug.nuisance} ({sug.reason})")

    matcher = PMHMatcher(nuisance=sug.nuisance, rank=args.rank, seed=args.seed)
    matcher.fit(x_src, y_src, x_tgt, y_tgt)
    print(f"artifact preflight={matcher.artifact_.preflight}  eigengap={matcher.artifact_.eigengap}")

    result = compare_arms_sklearn(
        x_src, y_src, x_tgt, y_tgt,
        rank=args.rank,
        include_coral=True,
        report_dir=args.report,
        seed=args.seed,
    )
    print("\n--- Arms (target accuracy) ---")
    for arm, run in result.arms.items():
        print(f"  {arm:10s}  {run.val_metric:.3f}")


if __name__ == "__main__":
    main()
