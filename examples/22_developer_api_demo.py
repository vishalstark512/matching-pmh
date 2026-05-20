#!/usr/bin/env python3
"""Developer API demo: check_applicability, robust_fit, evaluate_baseline_vs_pmh."""

from __future__ import annotations

import numpy as np


def main() -> None:
    from pmh import check_applicability, evaluate_baseline_vs_pmh

    print("=== sklearn evaluate_baseline_vs_pmh ===\n")
    rng = np.random.default_rng(0)
    n, d = 200, 24
    xs = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 3, n)
    q, _ = np.linalg.qr(rng.standard_normal((d, 5)))
    xt = xs + (xs @ q) @ q.T

    app = check_applicability(stack="sklearn", n_source=n, n_target=n, feature_dim=d)
    print(app.summary(), "\n")

    try:
        rep = evaluate_baseline_vs_pmh(
            x_source=xs, y_source=y, x_target=xt, y_target=y, compare_to=()
        )
        print(rep.summary())
    except ImportError:
        print('Install sklearn: pip install "matching-pmh[sklearn]"')

    print("\n=== PyTorch: run examples/00_first_run_domain_shift.py or use robust_fit ===")
    print("pmh-train wizard")


if __name__ == "__main__":
    main()
