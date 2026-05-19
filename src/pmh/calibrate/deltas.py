"""Build Sigma_task / W from stacked delta rows (PGD, style, aug, etc.)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pmh.artifact import SigmaTaskEstimate
from pmh.config import SigmaTaskConfig
from pmh.estimators.d7_alignment import estimate_d7


def subspace_from_stacked_deltas(
    deltas: np.ndarray | torch.Tensor,
    rank: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Top-`rank` right singular vectors of delta matrix G [N, d] → W [d, r]."""
    if isinstance(deltas, torch.Tensor):
        g = deltas.detach().float().cpu().numpy()
    else:
        g = np.asarray(deltas, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError("deltas must be [N, d]")
    g = g - g.mean(0, keepdims=True)
    g /= np.sqrt(max(len(g), 1))
    _, _, vt = np.linalg.svd(g, full_matrices=False)
    r = min(rank, vt.shape[0])
    return vt[:r].T.astype(np.float32)


def subspace_artifact_from_deltas(
    deltas: np.ndarray | torch.Tensor,
    *,
    rank: int,
    method: str = "D7",
    shrinkage: float = 1e-6,
    metadata: dict[str, Any] | None = None,
) -> SigmaTaskEstimate:
    """Package stacked deltas as :class:`SigmaTaskEstimate` (T7B PGD, custom probes)."""
    w = subspace_from_stacked_deltas(deltas, rank)
    if isinstance(deltas, torch.Tensor):
        d_t = deltas.detach().float()
    else:
        d_t = torch.from_numpy(np.asarray(deltas, dtype=np.float32))
    sigma = estimate_d7(d_t, rank=rank, shrinkage=shrinkage)
    meta = dict(metadata or {})
    meta["w"] = torch.from_numpy(w)
    cfg = (
        SigmaTaskConfig.for_alignment(rank=rank, shrinkage=shrinkage)
        if method == "D7"
        else SigmaTaskConfig(method=method, rank=rank, shrinkage=shrinkage)  # type: ignore[arg-type]
    )
    return SigmaTaskEstimate(sigma=sigma, method=method, config=cfg, metadata=meta)
