"""Bring your own geometry: deltas, W, or saved artifacts (any subtype D1–D7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from pmh.artifact import SigmaTaskEstimate
from pmh.calibrate import subspace_artifact_from_deltas
from pmh.config import Method, SigmaTaskConfig
from pmh.numpy_api import estimate_cross_domain_subspace_numpy

MethodLike = str


def artifact_from_deltas(
    deltas: np.ndarray | torch.Tensor,
    *,
    method: MethodLike = "D7",
    rank: int = 16,
    shrinkage: float = 1e-6,
    metadata: dict[str, Any] | None = None,
) -> SigmaTaskEstimate:
    """Stacked delta rows [N, d] → :class:`SigmaTaskEstimate` (D3/D6/D7 refinements)."""
    m: Method = SigmaTaskConfig(method=method).method
    return subspace_artifact_from_deltas(
        deltas, rank=rank, method=m, shrinkage=shrinkage, metadata=metadata
    )


def artifact_from_w(
    w: np.ndarray | torch.Tensor,
    *,
    method: MethodLike = "D1",
    shrinkage: float = 1e-6,
) -> SigmaTaskEstimate:
    """Build Σ = WWᵀ from an orthonormal or general basis W [d, r]."""
    key = str(method).strip().upper()
    if not key.startswith("D"):
        key = f"D{key}"
    if isinstance(w, torch.Tensor):
        w_np = w.detach().float().cpu().numpy()
    else:
        w_np = np.asarray(w, dtype=np.float32)
    if w_np.ndim != 2:
        raise ValueError("w must be [feature_dim, rank]")
    r = int(w_np.shape[1])
    sigma = (w_np @ w_np.T).astype(np.float32)
    sigma = 0.5 * (sigma + sigma.T) + shrinkage * np.eye(sigma.shape[0], dtype=np.float32)
    if key == "D1":
        cfg = SigmaTaskConfig.for_subspace(rank=r, shrinkage=shrinkage)
    elif key == "D2":
        cfg = SigmaTaskConfig.for_isotropic(sigma.shape[0], shrinkage=shrinkage)
    else:
        cfg = SigmaTaskConfig(method=key, rank=r, shrinkage=shrinkage)  # type: ignore[arg-type]
    return SigmaTaskEstimate(
        sigma=torch.from_numpy(sigma),
        method=cfg.method,
        config=cfg,
        metadata={"w": torch.from_numpy(w_np)},
    )


def load_w_numpy(path: str | Path) -> np.ndarray:
    """Load W from ``.npy`` (shape [d, r])."""
    w = np.load(path)
    if w.ndim != 2:
        raise ValueError(f"expected W [d, r], got shape {w.shape}")
    return w.astype(np.float32)


def estimate_custom(
    *,
    deltas: np.ndarray | torch.Tensor | None = None,
    w: np.ndarray | torch.Tensor | None = None,
    w_path: str | Path | None = None,
    method: MethodLike = "D4",
    rank: int = 16,
    shrinkage: float = 1e-6,
    # D1 labeled path
    x_src: np.ndarray | torch.Tensor | None = None,
    y_src: np.ndarray | None = None,
    x_tgt: np.ndarray | torch.Tensor | None = None,
    y_tgt: np.ndarray | None = None,
    seed: int = 0,
) -> SigmaTaskEstimate:
    """One entry point for custom identification within a subtype.

    Provide exactly one of: ``deltas``, ``w`` / ``w_path``, or labeled D1 arrays.
    """
    if w_path is not None:
        w = load_w_numpy(w_path)
    if deltas is not None:
        return artifact_from_deltas(deltas, method=method, rank=rank, shrinkage=shrinkage)
    if w is not None:
        return artifact_from_w(w, method=method, shrinkage=shrinkage)
    if x_src is not None and x_tgt is not None:
        xs = x_src.numpy() if isinstance(x_src, torch.Tensor) else np.asarray(x_src, dtype=np.float32)
        xt = x_tgt.numpy() if isinstance(x_tgt, torch.Tensor) else np.asarray(x_tgt, dtype=np.float32)
        if y_src is not None and y_tgt is not None:
            w_hat = estimate_cross_domain_subspace_numpy(
                xs, np.asarray(y_src), xt, np.asarray(y_tgt), rank=rank, seed=seed
            )
            return artifact_from_w(w_hat, method="D1", shrinkage=shrinkage)
        from pmh.numpy_api import estimate_sigma_task_numpy

        return estimate_sigma_task_numpy(
            xs, xt, config=SigmaTaskConfig.for_domain(rank=rank, shrinkage=shrinkage)
        )
    raise ValueError(
        "estimate_custom: pass deltas=, w=/w_path=, or x_src/x_tgt (+ labels for D1)"
    )
