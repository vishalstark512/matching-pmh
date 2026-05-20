"""Classical-ML helpers (T1): projection + soft k-NN."""

from pmh.classical.soft_knn import (
    compare_knn_hard_vs_soft,
    cv_softlift_alpha,
    knn_target_accuracy,
    softlift,
)

__all__ = [
    "softlift",
    "knn_target_accuracy",
    "cv_softlift_alpha",
    "compare_knn_hard_vs_soft",
]
