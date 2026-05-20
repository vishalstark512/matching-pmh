"""G2 golden path — replace arrays with your embeddings."""

import numpy as np

from pmh import check_applicability, evaluate_baseline_vs_pmh


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 300, 64
    x_src = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 4, n)
    q, _ = np.linalg.qr(rng.standard_normal((d, 6)))
    x_tgt = x_src + (x_src @ q) @ q.T

    print(check_applicability(stack="sklearn", n_source=n, n_target=n, feature_dim=d).summary())

    report = evaluate_baseline_vs_pmh(
        x_source=x_src,
        y_source=y,
        x_target=x_tgt,
        y_target=y,
        compare_to=("coral",),
    )
    print(report.summary())


if __name__ == "__main__":
    main()
