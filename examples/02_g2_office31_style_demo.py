#!/usr/bin/env python3
"""G2 newbie path: Office-31-style synthetic features + Step 5 report (no download).

  python examples/02_g2_office31_style_demo.py
  pmh-train doctor --stack sklearn
"""

from __future__ import annotations

from pmh import check_applicability, evaluate_baseline_vs_pmh, load_g2_demo_arrays


def main() -> None:
    x_source, y_source, x_target, y_target = load_g2_demo_arrays(n=500, seed=0)
    print(check_applicability(
        stack="sklearn",
        n_source=len(x_source),
        n_target=len(x_target),
        has_target_labels=True,
    ).summary())
    print()
    report = evaluate_baseline_vs_pmh(
        x_source=x_source,
        y_source=y_source,
        x_target=x_target,
        y_target=y_target,
        compare_to=("coral",),
    )
    print(report.summary())
    print()
    print("Next: swap arrays for your embeddings — docs/GOLDEN_PATHS.md#g2")
    print("Real Office-31: python examples/06_office31_sklearn.py --office31-root PATH")


if __name__ == "__main__":
    main()
