"""NumPy/sklearn four-arm protocol on frozen features."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from pmh.benchmark.protocol import ArmRunResult, BenchmarkResult
from pmh.baselines.coral import coral_align
from pmh.config import SigmaTaskConfig
from pmh.numpy_api import estimate_sigma_task_numpy
from pmh.sklearn_match import (
    MatchedSubspaceProjector,
    domain_d4_subspace_numpy,
    project_onto_complement,
    wrong_w_subspace_numpy,
)
from pmh.tdi import geometry_report


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


def _clamp_office_sizes(
    n_src: int,
    n_tgt: int,
    n_train: int,
    n_pool: int,
    n_test: int,
) -> tuple[int, int, int]:
    n_train = min(n_train, n_src)
    need = n_pool + n_test
    if need > n_tgt:
        scale = n_tgt / need
        n_pool = max(50, int(n_pool * scale))
        n_test = max(50, n_tgt - n_pool)
    n_pool = min(n_pool, n_tgt - 1)
    n_test = min(n_test, n_tgt - n_pool)
    return n_train, n_pool, n_test


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
    include_geometry: bool = True,
    paper_protocol: bool = True,
    n_train_src: int = 1500,
    n_target_pool: int = 200,
    n_test: int = 250,
    n_pairs_per_class: int = 40,
    test_size: float = 0.3,
) -> BenchmarkResult:
    """Train on source, test on held-out target: B0, matched, wrong-W, isotropic, CORAL.

    **paper_protocol=True** (default, matches ``T1/classical_pmh/office31_pmh.py``):

    - Estimate :math:`\\hat W` on source train + **target pool only** (no test leakage).
    - Test on a disjoint target slice (default pool=200, test=250).
    - D1 includes per-class mean shifts in the delta matrix.
    - wrong-W: random subspace orthogonalized to matched :math:`\\hat W`.
    - isotropic: top-:math:`r` directions of **D4** domain Gram (unmatched nuisance), not target PCA.

    **paper_protocol=False**: legacy random ``train_test_split`` on both domains (not recommended).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    clf_factory = classifier_factory or (lambda: LogisticRegression(max_iter=500, C=1.0))

    if paper_protocol:
        nt, npool, nte = _clamp_office_sizes(
            len(x_src), len(x_tgt), n_train_src, n_target_pool, n_test
        )
        xa_tr, ya_tr = x_src[:nt], y_src[:nt]
        x_pool, y_pool = x_tgt[:npool], y_tgt[:npool]
        xd_te, yd_te = x_tgt[npool : npool + nte], y_tgt[npool : npool + nte]
        w_fit_src, w_fit_y = xa_tr, ya_tr
        w_fit_tgt, w_fit_yt = x_pool, y_pool
    else:
        xa_tr, xa_te, ya_tr, ya_te = train_test_split(
            x_src, y_src, test_size=test_size, random_state=seed
        )
        xd_tr, xd_te, yd_tr, yd_te = train_test_split(
            x_tgt, y_tgt, test_size=test_size, random_state=seed
        )
        w_fit_src, w_fit_y = x_src, y_src
        w_fit_tgt, w_fit_yt = x_tgt, y_tgt
        x_pool, y_pool = xd_tr, yd_tr

    artifact = estimate_sigma_task_numpy(
        w_fit_src,
        w_fit_y,
        w_fit_tgt,
        w_fit_yt,
        config=SigmaTaskConfig.for_subspace(rank=rank),
    )
    out = BenchmarkResult(
        artifact_method=artifact.method,
        artifact_preflight=artifact.preflight,
        artifact_eigengap=artifact.eigengap,
    )
    proj = MatchedSubspaceProjector(
        rank=rank, seed=seed, n_pairs_per_class=n_pairs_per_class
    ).fit(w_fit_src, w_fit_y, w_fit_tgt, w_fit_yt)
    w_matched = proj.w_
    assert w_matched is not None

    q_wrong = wrong_w_subspace_numpy(w_matched, rank, seed=seed + 99)
    q_d4 = domain_d4_subspace_numpy(w_fit_src, w_fit_tgt, rank)

    def _geom(x_probe: np.ndarray, w: np.ndarray | None) -> dict[str, float | None]:
        if not include_geometry:
            return {}
        rep = geometry_report(x_probe, yd_te, w=w, seed=seed)
        return rep.to_dict()

    def _eval(
        name: str,
        xtr,
        ytr,
        xte,
        *,
        w_for_drift: np.ndarray | None = None,
    ) -> ArmRunResult:
        clf = clf_factory()
        clf.fit(xtr, ytr)
        acc = float(accuracy_score(yd_te, clf.predict(xte)))
        return ArmRunResult(
            arm=name,
            val_metric=acc,
            metric_name="target_accuracy",
            geometry=_geom(xte, w_for_drift),
        )

    out.arms["b0"] = _eval("b0", xa_tr, ya_tr, xd_te, w_for_drift=w_matched)
    out.arms["matched"] = _eval(
        "matched",
        proj.transform(xa_tr),
        ya_tr,
        proj.transform(xd_te),
        w_for_drift=w_matched,
    )
    out.arms["wrong_w"] = _eval(
        "wrong_w",
        project_onto_complement(xa_tr, q_wrong),
        ya_tr,
        project_onto_complement(xd_te, q_wrong),
        w_for_drift=q_wrong,
    )
    out.arms["isotropic"] = _eval(
        "isotropic",
        project_onto_complement(xa_tr, q_d4),
        ya_tr,
        project_onto_complement(xd_te, q_d4),
        w_for_drift=q_d4,
    )

    if include_coral:
        x_src_coral, _ = coral_align(w_fit_src, x_pool)
        out.arms["coral"] = _eval("coral", x_src_coral[: len(xa_tr)], ya_tr, xd_te, w_for_drift=w_matched)

    if paper_protocol:
        out.notes.append(
            f"Protocol: T1 Office-31 style — train n={len(xa_tr)}, "
            f"target pool n={len(x_pool)} (W only), test n={len(xd_te)}."
        )
    if include_geometry:
        out.notes.append(
            "Geometry: tdi_cls = class-layout TDI (lower better); "
            "D_N/D_S = directional drift along nuisance basis W."
        )

    return out


def run_sklearn_benchmark_multi_seed(
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_tgt: np.ndarray,
    y_tgt: np.ndarray,
    *,
    seeds: list[int] | tuple[int, ...],
    **kwargs: Any,
) -> BenchmarkResult:
    """Run :func:`run_sklearn_benchmark` per seed; aggregate mean val_metric per arm."""
    runs = [
        run_sklearn_benchmark(x_src, y_src, x_tgt, y_tgt, seed=int(sd), **kwargs)
        for sd in seeds
    ]
    base = runs[0]
    out = BenchmarkResult(
        artifact_method=base.artifact_method,
        artifact_preflight=base.artifact_preflight,
        artifact_eigengap=base.artifact_eigengap,
    )
    out.notes.extend(base.notes)
    out.notes.append(f"Multi-seed: {list(seeds)} — val_metric is mean over seeds.")

    arms = set()
    for r in runs:
        arms.update(r.arms.keys())
    for arm in arms:
        metrics = [r.arms[arm].val_metric for r in runs if arm in r.arms and r.arms[arm].val_metric is not None]
        if not metrics:
            continue
        mean_m = float(np.mean(metrics))
        std_m = float(np.std(metrics)) if len(metrics) > 1 else 0.0
        geom = runs[0].arms[arm].geometry if arm in runs[0].arms else {}
        out.arms[arm] = ArmRunResult(
            arm=arm,
            val_metric=mean_m,
            metric_name=f"target_accuracy_mean (std={std_m:.4f}, n={len(metrics)})",
            geometry=geom,
        )
    return out
