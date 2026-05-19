"""NumPy/sklearn four-arm protocol on frozen features."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def synthetic_office31_features(
    n: int = 400,
    *,
    seed: int = 0,
    d: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic domain shift (T1 / example 06 spirit)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((d, 12)).astype(np.float32)
    q, _ = np.linalg.qr(w)
    x_a = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 10, n)
    nuisance = (x_a @ q) @ q.T
    x_d = x_a + 1.5 * nuisance + 0.05 * rng.standard_normal((n, d)).astype(np.float32)
    return x_a, y, x_d, y.copy()

from pmh.benchmark.protocol import ArmRunResult, BenchmarkResult
from pmh.baselines.coral import coral_align
from pmh.config import SigmaTaskConfig
from pmh.numpy_api import estimate_sigma_task_numpy
from pmh.sklearn_match import MatchedSubspaceProjector, project_onto_complement


def run_sklearn_benchmark(
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_tgt: np.ndarray,
    y_tgt: np.ndarray,
    *,
    rank: int = 16,
    classifier_factory: Callable[[], Any] | None = None,
    seed: int = 0,
    include_coral: bool = True,
) -> BenchmarkResult:
    """Train/test on target domain: B0, matched projection, wrong-W, isotropic-style subspace, CORAL.

    Matches ``T1/classical_pmh`` and ``examples/06_office31_sklearn.py`` logic.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    clf_factory = classifier_factory or (lambda: LogisticRegression(max_iter=500, C=1.0))

    xa_tr, xa_te, ya_tr, ya_te = train_test_split(x_src, y_src, test_size=0.3, random_state=seed)
    xd_tr, xd_te, yd_tr, yd_te = train_test_split(x_tgt, y_tgt, test_size=0.3, random_state=seed)

    artifact = estimate_sigma_task_numpy(
        x_src, y_src, x_tgt, y_tgt,
        config=SigmaTaskConfig.for_subspace(rank=rank),
    )
    out = BenchmarkResult(
        artifact_method=artifact.method,
        artifact_preflight=artifact.preflight,
        artifact_eigengap=artifact.eigengap,
    )
    proj = MatchedSubspaceProjector(rank=rank, seed=seed).fit(x_src, y_src, x_tgt, y_tgt)

    rng = np.random.default_rng(seed + 99)
    w_wrong = rng.standard_normal((x_src.shape[1], rank)).astype(np.float32)
    q_wrong, _ = np.linalg.qr(w_wrong)

    def _eval(name: str, xtr, ytr, xte) -> ArmRunResult:
        clf = clf_factory()
        clf.fit(xtr, ytr)
        acc = float(accuracy_score(yd_te, clf.predict(xte)))
        return ArmRunResult(arm=name, val_metric=acc, metric_name="target_accuracy")

    out.arms["b0"] = _eval("b0", xa_tr, ya_tr, xd_te)
    out.arms["matched"] = _eval("matched", proj.transform(xa_tr), ya_tr, proj.transform(xd_te))
    out.arms["wrong_w"] = _eval(
        "wrong_w",
        project_onto_complement(xa_tr, q_wrong),
        ya_tr,
        project_onto_complement(xd_te, q_wrong),
    )
    # Isotropic-style: complement of full estimated subspace (same rank as matched for fair rank)
    u_full, _, _ = np.linalg.svd(x_tgt - x_tgt.mean(0), full_matrices=False)
    q_iso = u_full[:rank].T.astype(np.float32)
    out.arms["isotropic"] = _eval(
        "isotropic",
        project_onto_complement(xa_tr, q_iso),
        ya_tr,
        project_onto_complement(xd_te, q_iso),
    )

    if include_coral:
        x_src_coral, _ = coral_align(x_src, x_tgt)
        xc_tr, _, yc_tr, _ = train_test_split(x_src_coral, y_src, test_size=0.3, random_state=seed)
        out.arms["coral"] = _eval("coral", xc_tr, yc_tr, xd_te)

    return out
