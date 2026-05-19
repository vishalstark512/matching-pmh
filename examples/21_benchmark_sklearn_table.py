#!/usr/bin/env python3
"""Standard sklearn benchmark: B0 vs matched vs wrong-W vs isotropic (+ CORAL).

Reports **target accuracy** and geometry (**TDI_cls**, **D_N/D_S**) — same arms as
``compare_arms_sklearn`` / falsification walkthrough 08.

Usage
-----
    python examples/21_benchmark_sklearn_table.py
    python examples/21_benchmark_sklearn_table.py --office31-root /path/to/office31
    python examples/21_benchmark_sklearn_table.py --report results/sklearn_benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="PMH sklearn benchmark table")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--office31-root", type=str, default=None)
    parser.add_argument("--source", type=str, default="amazon")
    parser.add_argument("--target", type=str, default="dslr")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--report", type=str, default=None, help="Write benchmark.json/md")
    parser.add_argument("--no-coral", action="store_true")
    args = parser.parse_args()

    if args.office31_root:
        from pmh.datasets.office31 import extract_office31_features

        print(f"Dataset: Office-31  {args.source} -> {args.target}")
        x_src, y_src = extract_office31_features(
            args.office31_root,
            args.source,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        x_tgt, y_tgt = extract_office31_features(
            args.office31_root,
            args.target,
            max_samples=args.max_samples,
            seed=args.seed + 1,
        )
    else:
        from pmh.benchmark.sklearn_protocol import synthetic_office31_features

        print("Dataset: synthetic Office-31-style shift (use --office31-root for real data)")
        x_src, y_src, x_tgt, y_tgt = synthetic_office31_features(
            n=min(args.max_samples, 600),
            seed=args.seed,
            d=128,
        )

    from pmh import compare_arms_sklearn
    from pmh.benchmark.report import benchmark_to_markdown

    result = compare_arms_sklearn(
        x_src,
        y_src,
        x_tgt,
        y_tgt,
        rank=args.rank,
        include_coral=not args.no_coral,
        include_geometry=True,
        seed=args.seed,
        report_dir=args.report,
    )

    print("\n" + benchmark_to_markdown(result.to_dict(), title="Sklearn benchmark (accuracy + TDI)"))

    if args.report:
        print(f"\nWrote {Path(args.report)}/benchmark.json and benchmark.md")


if __name__ == "__main__":
    main()
