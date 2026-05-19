#!/usr/bin/env python3
"""Classical domain shift (Office-31 style) with NumPy + scikit-learn.

Default: **synthetic** 512-d features (no download).
Use ``--office31-root PATH`` for real ResNet-18 features (amazon vs dslr).

Reports B0, matched PMH (D1 projection), wrong-W, and **CORAL** (Sun & Saenko 2016).
Paper reference (Office-31): CORAL SVM ~25.2% vs matched PMH ~23.3% when eigengap ~1.03.
"""

from __future__ import annotations

import argparse

import numpy as np


def _synthetic_office31(n: int = 400, d: int = 512, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((d, 32)).astype(np.float32)
    q, _ = np.linalg.qr(w)
    x_a = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 31, n)
    nuisance = (x_a @ q) @ q.T
    x_d = x_a + 1.5 * nuisance + 0.05 * rng.standard_normal((n, d)).astype(np.float32)
    return x_a, y, x_d, y.copy()


def _load_office31(
    root: str,
    *,
    source: str = "amazon",
    target: str = "dslr",
    max_samples: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, ...]:
    from pmh.datasets.office31 import extract_office31_features

    print(f"Extracting Office-31 features: {source} vs {target} from {root}")
    x_src, y_src = extract_office31_features(root, source, max_samples=max_samples, seed=seed)
    x_tgt, y_tgt = extract_office31_features(root, target, max_samples=max_samples, seed=seed + 1)
    return x_src, y_src, x_tgt, y_tgt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--office31-root", type=str, default=None)
    parser.add_argument("--source", type=str, default="amazon")
    parser.add_argument("--target", type=str, default="dslr")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument(
        "--classifier",
        choices=("logistic", "svm"),
        default="logistic",
        help="Classifier on frozen features (paper Office-31 uses SVM)",
    )
    args = parser.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
    except ImportError as exc:
        raise SystemExit("pip install scikit-learn") from exc

    from pmh.baselines.coral import coral_align
    from pmh.config import SigmaTaskConfig
    from pmh.numpy_api import estimate_sigma_task_numpy
    from pmh.sklearn_match import MatchedSubspaceProjector, project_onto_complement

    if args.office31_root:
        x_src, y_src, x_tgt, y_tgt = _load_office31(
            args.office31_root,
            source=args.source,
            target=args.target,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    else:
        print("Synthetic features (pass --office31-root for real Office-31).")
        x_src, y_src, x_tgt, y_tgt = _synthetic_office31(seed=args.seed)

    xa_tr, xa_te, ya_tr, ya_te = train_test_split(x_src, y_src, test_size=0.3, random_state=args.seed)
    xd_tr, xd_te, yd_tr, yd_te = train_test_split(x_tgt, y_tgt, test_size=0.3, random_state=args.seed)

    artifact = estimate_sigma_task_numpy(
        x_src, y_src, x_tgt, y_tgt,
        config=SigmaTaskConfig.for_subspace(rank=args.rank),
    )
    print(f"preflight={artifact.preflight}  eigengap={artifact.eigengap}")

    proj = MatchedSubspaceProjector(rank=args.rank, seed=args.seed).fit(x_src, y_src, x_tgt, y_tgt)

    def make_clf():
        if args.classifier == "svm":
            return SVC(kernel="linear", C=1.0)
        return LogisticRegression(max_iter=500, C=1.0)

    def eval_clf(name: str, x_tr, y_tr, x_te, y_te) -> float:
        clf = make_clf()
        clf.fit(x_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(x_te))
        print(f"  {name}: target acc = {acc:.3f}")
        return acc

    print("B0 (raw, train source -> test target):")
    eval_clf("B0", xa_tr, ya_tr, xd_te, yd_te)

    print("Matched PMH (D1 subspace projection):")
    eval_clf("matched", proj.transform(xa_tr), ya_tr, proj.transform(xd_te), yd_te)

    rng = np.random.default_rng(args.seed + 99)
    w_wrong = rng.standard_normal((x_src.shape[1], args.rank)).astype(np.float32)
    q_wrong, _ = np.linalg.qr(w_wrong)
    print("Wrong-W control:")
    eval_clf(
        "wrong-W",
        project_onto_complement(xa_tr, q_wrong),
        ya_tr,
        project_onto_complement(xd_te, q_wrong),
        yd_te,
    )

    print("CORAL (align source stats to target, train source -> test target):")
    x_src_coral, _ = coral_align(x_src, x_tgt)
    xc_tr, _, yc_tr, _ = train_test_split(x_src_coral, y_src, test_size=0.3, random_state=args.seed)
    eval_clf("CORAL", xc_tr, yc_tr, xd_te, yd_te)

    if args.office31_root:
        print(
            "\nPaper note (amazon->dslr, linear SVM on ResNet-18): "
            "CORAL ~0.252 vs matched PMH ~0.233 when eigengap ~1.03; "
            "re-run with --classifier svm to compare."
        )


if __name__ == "__main__":
    main()
