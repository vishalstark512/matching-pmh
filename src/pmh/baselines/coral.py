"""CORAL: Correlation Alignment (Sun & Saenko, 2016)."""

from __future__ import annotations

import numpy as np


def coral_align(
    x_src: np.ndarray,
    x_tgt: np.ndarray,
    *,
    ridge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Align source second-order statistics to target.

    Returns ``(x_src_aligned, x_tgt)``; train on aligned source, test on raw target.
    """
    x_src = np.asarray(x_src, dtype=np.float32)
    x_tgt = np.asarray(x_tgt, dtype=np.float32)
    mu_s = x_src.mean(0, keepdims=True)
    mu_t = x_tgt.mean(0, keepdims=True)
    xs = x_src - mu_s
    xt = x_tgt - mu_t
    n_s = max(len(xs), 2)
    n_t = max(len(xt), 2)
    d = xs.shape[1]
    c_s = (xs.T @ xs) / (n_s - 1) + ridge * np.eye(d, dtype=np.float32)
    c_t = (xt.T @ xt) / (n_t - 1) + ridge * np.eye(d, dtype=np.float32)
    us, ss, _ = np.linalg.svd(c_s)
    ut, st, _ = np.linalg.svd(c_t)
    ss = np.clip(ss, 1e-6, None)
    st = np.clip(st, 1e-6, None)
    c_s_inv_sqrt = (us * (ss ** -0.5)) @ us.T
    c_t_sqrt = (ut * (st ** 0.5)) @ ut.T
    m = (c_s_inv_sqrt @ c_t_sqrt).astype(np.float32)
    x_src_aligned = xs @ m + mu_t
    return x_src_aligned.astype(np.float32), x_tgt.astype(np.float32)
