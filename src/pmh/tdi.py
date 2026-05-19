"""Trajectory / layout TDI and directional drift (paper Section 6, diagnostic metrics).

These metrics are **label-free or label-structured geometry probes** — complementary to
task accuracy. They were used throughout the Grand Unification replication code
(``Paper2/T3/Task3A/tdi_utils.py``, ``T4/*/tdi.py``, ``T6/*/eval_*``); this module
is the packaged API for ``matching-pmh`` users.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "TDIReport",
    "tdi_cls",
    "tdi_layout",
    "tdi_feature_isotropic",
    "directional_drift_numpy",
    "geometry_report",
]


@dataclass
class TDIReport:
    """Collected geometry metrics for one representation map."""

    tdi_cls: float | None = None
    tdi_feature_iso: float | None = None
    d_n: float | None = None
    d_s: float | None = None
    d_n_over_d_s: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "tdi_cls": self.tdi_cls,
            "tdi_feature_iso": self.tdi_feature_iso,
            "D_N": self.d_n,
            "D_S": self.d_s,
            "D_N_over_D_S": self.d_n_over_d_s,
        }


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def tdi_cls(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    max_per_class: int = 200,
    seed: int = 42,
) -> float:
    """Class-layout :math:`\\mathrm{TDI}_0^{\\mathrm{cls}}` (Eq. layout in paper §6).

    Ratio of mean intra-class pairwise distance to mean inter-class centroid distance
    on L2-normalised embeddings. **Lower is better.**

    Same formula as ``Paper2/T3/Task3A/tdi_utils.compute_tdi`` and
    ``T6/task6B/eval_tdi.compute_tdi_cls``.
    """
    return tdi_layout(embeddings, labels, max_per_class=max_per_class, seed=seed)


def tdi_layout(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    max_per_class: int = 200,
    seed: int = 42,
) -> float:
    """Alias for :func:`tdi_cls` (layout / Style TDI on labelled embeddings)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    emb = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    classes = np.unique(labels)
    if len(classes) < 2:
        return float("nan")

    per_class: dict[int, np.ndarray] = {}
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        per_class[int(c)] = emb[idx]

    intra_dists: list[float] = []
    for feats in per_class.values():
        if len(feats) < 2:
            continue
        diff = feats[:, None, :] - feats[None, :, :]
        sq = (diff**2).sum(axis=2)
        triu = sq[np.triu_indices(len(feats), k=1)]
        intra_dists.append(float(np.sqrt(triu.mean())))
    intra = float(np.mean(intra_dists)) if intra_dists else 0.0

    centroids = {c: feats.mean(axis=0) for c, feats in per_class.items()}
    keys = list(centroids.keys())
    inter_dists: list[float] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter_dists.append(
                float(np.linalg.norm(centroids[keys[i]] - centroids[keys[j]]))
            )
    inter = float(np.mean(inter_dists)) if inter_dists else 1e-8
    return float(intra / max(inter, 1e-8))


def tdi_feature_isotropic(
    features: np.ndarray,
    *,
    sigma: float = 0.01,
    n_noise: int = 32,
    seed: int = 0,
) -> float:
    """Feature-space isotropic sensitivity proxy (trajectory TDI spirit on frozen :math:`\\phi(x)=x`).

    For deep encoders, use layer-averaged input-Gaussian probes in PyTorch (see paper
    Eq.~\\eqref{eq:tdi-trajectory}); this NumPy form is for **frozen features** and
    sklearn benchmarks.

    Computes
    :math:`\\mathbb{E}_{x,\\delta}[\\|x+\\delta-x\\|^2] / \\mathbb{E}_x[\\|x\\|^2]`
    with :math:`\\delta \\sim \\mathcal{N}(0, \\sigma^2 I)`.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    denom = float((x**2).sum(axis=1).mean()) + 1e-8
    numer_acc = 0.0
    count = 0
    for row in x:
        for _ in range(n_noise):
            delta = rng.standard_normal(row.shape[0], dtype=np.float32) * sigma
            diff = delta
            numer_acc += float((diff**2).sum())
            count += 1
    return numer_acc / max(count, 1) / denom


def directional_drift_numpy(
    features: np.ndarray,
    w: np.ndarray,
    *,
    sigma: float = 0.1,
    n_noise: int = 64,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Directional drift :math:`D_N`, :math:`D_S`, and ratio (paper §6).

    Parameters
    ----------
    features
        Probe rows ``[n, d]`` (typically target-domain embeddings).
    w
        Nuisance basis ``[d, r]`` (columns need not be orthonormal; QR applied internally).

    Returns
    -------
    d_n, d_s, d_n_over_d_s
        Mean perturbation norms along :math:`\\hat W` and its orthogonal complement.
    """
    x = np.asarray(features, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    if x.ndim != 2 or w.ndim != 2 or w.shape[0] != x.shape[1]:
        raise ValueError("features [n,d] and w [d,r] required with matching d")

    q, _ = np.linalg.qr(w, mode="reduced")
    p_w = q @ q.T
    d = x.shape[1]
    p_perp = np.eye(d, dtype=np.float32) - p_w

    rng = np.random.default_rng(seed)
    dn_vals: list[float] = []
    ds_vals: list[float] = []
    n_probe = min(len(x), 256)
    idx = rng.choice(len(x), n_probe, replace=False) if len(x) > n_probe else np.arange(len(x))

    for i in idx:
        for _ in range(n_noise):
            z = rng.standard_normal(d, dtype=np.float32) * sigma
            dn_vals.append(float(np.linalg.norm(p_w @ z)))
            ds_vals.append(float(np.linalg.norm(p_perp @ z)))

    d_n = float(np.mean(dn_vals))
    d_s = float(np.mean(ds_vals))
    return d_n, d_s, d_n / max(d_s, 1e-12)


def geometry_report(
    embeddings: np.ndarray,
    labels: np.ndarray | None,
    w: np.ndarray | None = None,
    *,
    sigma_iso: float = 0.01,
    sigma_drift: float = 0.1,
    seed: int = 0,
) -> TDIReport:
    """Compute layout TDI (+ optional isotropic / directional drift) on one embedding matrix."""
    rep = TDIReport()
    rep.tdi_feature_iso = tdi_feature_isotropic(embeddings, sigma=sigma_iso, seed=seed)
    if labels is not None:
        rep.tdi_cls = tdi_cls(embeddings, labels, seed=seed)
    if w is not None:
        rep.d_n, rep.d_s, rep.d_n_over_d_s = directional_drift_numpy(
            embeddings, w, sigma=sigma_drift, seed=seed
        )
    return rep
