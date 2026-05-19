"""Trajectory / layout TDI and directional drift (paper Section 6, diagnostic metrics).

These metrics are **label-free or label-structured geometry probes** — complementary to
task accuracy. They were used throughout the Grand Unification replication code
(``Paper2/T3/Task3A/tdi_utils.py``, ``T4/*/tdi.py``, ``T6/*/eval_*``); this module
is the packaged API for ``matching-pmh`` users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "TDIReport",
    "tdi_cls",
    "tdi_layout",
    "tdi_feature_isotropic",
    "trajectory_tdi_layerwise",
    "trajectory_tdi_encoder",
    "directional_drift_numpy",
    "geometry_report",
]


@dataclass
class TDIReport:
    """Collected geometry metrics for one representation map."""

    tdi_cls: float | None = None
    tdi_feature_iso: float | None = None
    trajectory_tdi: float | None = None
    tdi_per_layer: list[float] | None = None
    d_n: float | None = None
    d_s: float | None = None
    d_n_over_d_s: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "tdi_cls": self.tdi_cls,
            "tdi_feature_iso": self.tdi_feature_iso,
            "trajectory_tdi": self.trajectory_tdi,
            "tdi_per_layer": self.tdi_per_layer,
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


def trajectory_tdi_layerwise(
    layers_clean: list[np.ndarray],
    layers_perturbed: list[np.ndarray],
) -> tuple[float, list[float]]:
    """Layer-averaged trajectory TDI (paper Eq. trajectory; Task 2A ``compute_paper_tdi_layerwise``).

    For each depth ``ell``, computes mean_i ||phi_ell(x_i + delta_i) - phi_ell(x_i)||^2 / mean_i ||phi_ell(x_i)||^2,
    then averages across layers. **Lower is better.**

    Parameters
    ----------
    layers_clean, layers_perturbed
        Lists of paired ``[N, d_ell]`` arrays in the same sample order.
    """
    if len(layers_clean) != len(layers_perturbed):
        raise ValueError(
            f"Layer count mismatch: clean={len(layers_clean)}, perturbed={len(layers_perturbed)}"
        )
    ratios: list[float] = []
    for c, p in zip(layers_clean, layers_perturbed):
        cc = np.asarray(c, dtype=np.float64)
        pp = np.asarray(p, dtype=np.float64)
        if cc.shape != pp.shape:
            raise ValueError(f"Shape mismatch: clean={cc.shape}, perturbed={pp.shape}")
        num = np.square(np.linalg.norm(pp - cc, axis=1))
        den = np.square(np.linalg.norm(cc, axis=1))
        den_mean = float(np.mean(den))
        if den_mean <= 0:
            ratios.append(0.0)
        else:
            ratios.append(float(np.mean(num) / den_mean))
    tdi = float(np.mean(ratios)) if ratios else 0.0
    return tdi, ratios


def trajectory_tdi_encoder(
    model: Any,
    encoder: Any,
    batches: Any,
    *,
    sigma: float = 0.01,
    max_batches: int = 20,
    device: Any = None,
    seed: int = 0,
) -> dict[str, float | list[float] | int]:
    """Estimate trajectory TDI@sigma with isotropic input noise on a PyTorch encoder.

    Default: single representation ``h = encoder(x)`` (final layer / CLS).
    Perturbs inputs ``x' = x + sigma * eps``, recomputes ``h`` on the same batches.

    Matches paper probe at ``sigma=0.01`` (T2A-style); use :func:`trajectory_tdi_layerwise`
    directly when you have per-layer CLS tensors from timm/ViT hooks.
    """
    import torch

    if device is None:
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else "cpu"
    dev = torch.device(device)
    model.eval()

    clean_parts: list[np.ndarray] = []
    pert_parts: list[np.ndarray] = []
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    n_batches = 0

    for item in batches:
        if max_batches is not None and n_batches >= max_batches:
            break
        x = item[0] if isinstance(item, (tuple, list)) else item
        x = x.to(dev)
        eps = torch.randn(x.shape, generator=gen, device=dev, dtype=x.dtype) * sigma
        x_pert = x + eps

        with torch.no_grad():
            h0 = encoder(x).detach().float().cpu().numpy()
            h1 = encoder(x_pert).detach().float().cpu().numpy()
        if h0.ndim != 2:
            raise ValueError(f"encoder must return [B, d], got {h0.shape}")
        clean_parts.append(h0)
        pert_parts.append(h1)
        n_batches += 1

    if not clean_parts:
        raise ValueError("probe_batches yielded no data")

    h_clean = np.concatenate(clean_parts, axis=0)
    h_pert = np.concatenate(pert_parts, axis=0)
    tdi, per_layer = trajectory_tdi_layerwise([h_clean], [h_pert])
    return {
        "trajectory_tdi": tdi,
        "tdi_per_layer": per_layer,
        "sigma": float(sigma),
        "n_samples": int(h_clean.shape[0]),
    }


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
