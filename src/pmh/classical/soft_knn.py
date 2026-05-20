"""Soft Mahalanobis k-NN for T1 (hard projection hurts distance geometry)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def softlift(
    x: np.ndarray,
    w: np.ndarray,
    *,
    alpha: float,
    beta: float = 1.0,
) -> np.ndarray:
    """Apply L = sqrt(beta) P_perp + sqrt(alpha) P_W (paper T1 soft_knn.py).

    Euclidean distance in lifted space equals Mahalanobis with
    M = beta P_perp + alpha P_W.
    """
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    proj_w = (x @ w) @ w.T
    proj_perp = x - proj_w
    return (np.sqrt(beta) * proj_perp + np.sqrt(alpha) * proj_w).astype(np.float32)


def knn_target_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    k: int = 5,
) -> float:
    """k-NN accuracy on held-out target rows."""
    clf = KNeighborsClassifier(n_neighbors=k, algorithm="brute", metric="euclidean")
    clf.fit(x_train, y_train)
    return float(accuracy_score(y_test, clf.predict(x_test)))


def cv_softlift_alpha(
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    w: np.ndarray,
    *,
    alphas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    k: int = 5,
) -> tuple[float, dict[float, float]]:
    """Pick alpha maximizing validation k-NN accuracy after softlift."""
    scores: dict[float, float] = {}
    for a in alphas:
        scores[a] = knn_target_accuracy(
            softlift(x_src, w, alpha=a),
            y_src,
            softlift(x_val, w, alpha=a),
            y_val,
            k=k,
        )
    best = max(scores, key=scores.get)
    return best, scores


def compare_knn_hard_vs_soft(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    w: np.ndarray,
    *,
    val_fraction: float = 0.25,
    k: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """B0 raw, hard project, softlift (CV alpha) on a target train/test split."""
    from sklearn.model_selection import train_test_split

    x_tr, x_te, y_tr, y_te = train_test_split(
        x_target, y_target, test_size=val_fraction, random_state=seed
    )
    x_val, x_te2, y_val, y_te2 = train_test_split(
        x_tr, y_tr, test_size=0.4, random_state=seed + 1
    )
    x_src_fit = np.vstack([x_source, x_val])
    y_src_fit = np.concatenate([y_source, y_val])

    out = {
        "b0": knn_target_accuracy(x_src_fit, y_src_fit, x_te2, y_te2, k=k),
        "matched_hard": knn_target_accuracy(
            softlift(x_src_fit, w, alpha=0.0),
            y_src_fit,
            softlift(x_te2, w, alpha=0.0),
            y_te2,
            k=k,
        ),
    }
    best_a, _ = cv_softlift_alpha(x_src_fit, y_src_fit, x_val, y_val, w, k=k)
    out["matched_soft"] = knn_target_accuracy(
        softlift(x_src_fit, w, alpha=best_a),
        y_src_fit,
        softlift(x_te2, w, alpha=best_a),
        y_te2,
        k=k,
    )
    out["best_alpha"] = best_a
    return out
