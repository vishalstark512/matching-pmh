"""Gradient-SVD subspace (Paper2 Task3A protocol; not default D3 aug-delta)."""

from __future__ import annotations

import numpy as np

from pmh.artifact import SigmaTaskEstimate
from pmh.calibrate.deltas import subspace_artifact_from_deltas


def gradient_subspace_numpy(
    gradients: np.ndarray,
    *,
    rank: int = 16,
    method: str = "D3",
) -> tuple[np.ndarray, SigmaTaskEstimate]:
    """SVD on stacked input gradients [N, d] → W (Task3A ``calibrate_subspace`` spirit)."""
    art = subspace_artifact_from_deltas(gradients, rank=rank, method=method)
    w = art.metadata["w"]
    w_np = w.cpu().numpy() if hasattr(w, "cpu") else np.asarray(w)
    return w_np.astype(np.float32), art
