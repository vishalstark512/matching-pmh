"""Content-residual temporal subspace (Paper2 Task6A protocol)."""

from __future__ import annotations

import numpy as np

from pmh.calibrate.deltas import subspace_artifact_from_deltas


def content_residual_subspace(
    sequences: np.ndarray,
    *,
    rank: int = 32,
    source: str = "content",
) -> tuple[np.ndarray, "SigmaTaskEstimate"]:
    """PCA on pooled token-group residuals (6A) or consecutive diffs (6B-style).

    Parameters
    ----------
    sequences
        [N, T, d] token trajectories or [N, d] per-step features.
    source
        ``content`` — mean-pool over time, subtract class/content mean per group;
        ``temporal`` — consecutive differences along T.
    """
    x = np.asarray(sequences, dtype=np.float32)
    if x.ndim == 2:
        if source == "temporal":
            raise ValueError("temporal source needs [N, T, d] with T>1")
        deltas = x - x.mean(0, keepdims=True)
        flat = deltas
    else:
        if source == "content":
            pooled = x.mean(axis=1)
            flat = pooled - pooled.mean(0, keepdims=True)
        else:
            diffs = x[:, 1:, :] - x[:, :-1, :]
            flat = diffs.reshape(-1, x.shape[-1])
    art = subspace_artifact_from_deltas(flat, rank=rank, method="D6")
    w = art.metadata["w"]
    w_np = w.cpu().numpy() if hasattr(w, "cpu") else np.asarray(w)
    return w_np, art
