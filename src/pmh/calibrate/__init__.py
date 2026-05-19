"""Task-specific calibration helpers (thin wrappers over paper protocols)."""

from pmh.calibrate.deltas import subspace_artifact_from_deltas
from pmh.calibrate.content_residual import content_residual_subspace
from pmh.calibrate.gradient import gradient_subspace_numpy
from pmh.calibrate.style import style_gram_from_deltas

__all__ = [
    "subspace_artifact_from_deltas",
    "style_gram_from_deltas",
    "gradient_subspace_numpy",
    "content_residual_subspace",
]
