"""NumPy-facing estimators (no PyTorch required for estimation)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pmh.artifact import SigmaTaskEstimate
from pmh.config import SigmaTaskConfig
from pmh.preflight import preflight_eigengap


def gram_from_diff_numpy(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Lemma D4 Gram: (1/N) D^T D for centred source - target rows."""
    s = np.asarray(source, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    if s.ndim != 2 or t.ndim != 2:
        raise ValueError("source and target must be [N, d]")
    s = s - s.mean(0, keepdims=True)
    t = t - t.mean(0, keepdims=True)
    n = min(len(s), len(t))
    diff = s[:n] - t[:n]
    return (diff.T @ diff) / max(n, 1)


def estimate_cross_domain_subspace_numpy(
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_tgt: np.ndarray,
    y_tgt: np.ndarray,
    *,
    rank: int = 16,
    seed: int = 0,
    n_pairs_per_class: int = 100,
    include_mean_shift: bool = True,
) -> np.ndarray:
    """Lemma D1: top-`rank` directions from class-aligned cross-domain deltas.

    Stacks class-mean shifts :math:`\\mu_T^c - \\mu_S^c` (when ``include_mean_shift``)
  and centered same-class pair deltas, then takes top right singular vectors (T1 protocol).
    """
    rng = np.random.default_rng(seed)
    classes = np.intersect1d(np.unique(y_src), np.unique(y_tgt))
    deltas: list[np.ndarray] = []
    for c in classes:
        idx_s = np.where(y_src == c)[0]
        idx_t = np.where(y_tgt == c)[0]
        if len(idx_s) == 0 or len(idx_t) == 0:
            continue
        if include_mean_shift:
            deltas.append((x_tgt[idx_t].mean(0) - x_src[idx_s].mean(0))[None, :].astype(np.float32))
        n_pairs = min(n_pairs_per_class, len(idx_s), len(idx_t))
        ps = rng.choice(idx_s, n_pairs, replace=len(idx_s) < n_pairs)
        pt = rng.choice(idx_t, n_pairs, replace=len(idx_t) < n_pairs)
        d = x_tgt[pt].astype(np.float32) - x_src[ps].astype(np.float32)
        d -= d.mean(0, keepdims=True)
        deltas.append(d)
    if not deltas:
        raise ValueError("no class-aligned pairs between domains")
    g = np.concatenate(deltas, axis=0)
    g /= np.sqrt(max(len(g), 1))
    _, _, vt = np.linalg.svd(g, full_matrices=False)
    r = min(rank, vt.shape[0])
    return vt[:r].T.astype(np.float32)


def _truncate_rank_numpy(cov: np.ndarray, rank: int) -> np.ndarray:
    evals, evecs = np.linalg.eigh(cov)
    r = min(rank, cov.shape[0])
    top_e = evecs[:, -r:]
    top_l = np.clip(evals[-r:], 0.0, None)
    return (top_e * top_l) @ top_e.T


def estimate_sigma_task_numpy(
    *args: Any,
    config: SigmaTaskConfig | None = None,
    method: str = "D4",
    rank: int | None = None,
    shrinkage: float = 1e-6,
    **kwargs: Any,
) -> SigmaTaskEstimate:
    """Estimate Sigma_task from NumPy arrays; returns :class:`SigmaTaskEstimate`."""
    if config is None:
        config = SigmaTaskConfig(method=method, rank=rank, shrinkage=shrinkage, **{
            k: kwargs[k]
            for k in ("dim", "noise_level", "nuisance_indices")
            if k in kwargs
        })

    m = config.method
    eigengap = None
    preflight = None

    if m == "D2":
        if config.dim is None:
            raise ValueError("D2 requires dim")
        nl = float(config.noise_level or 0.1)
        sigma = (nl**2) * np.eye(config.dim, dtype=np.float32)
    elif m in ("D1", "D4"):
        if m == "D1":
            if len(args) < 4:
                raise ValueError("D1: pass x_src, y_src, x_tgt, y_tgt")
            x_src, y_src, x_tgt, y_tgt = args[0], args[1], args[2], args[3]
            if config.rank is None:
                raise ValueError("D1 requires rank")
            w = estimate_cross_domain_subspace_numpy(
                x_src, y_src, x_tgt, y_tgt, rank=config.rank
            )
            sigma = (w @ w.T).astype(np.float32)
            preflight_cov = sigma
        else:
            if len(args) >= 4:
                x_src, x_tgt = args[0], args[2]
            elif len(args) >= 2:
                x_src, x_tgt = args[0], args[1]
            else:
                raise ValueError("D4: pass (x_src, x_tgt) or (x_src, y_src, x_tgt, y_tgt)")
            sigma = gram_from_diff_numpy(x_src, x_tgt)
            if config.rank is not None:
                sigma = _truncate_rank_numpy(sigma, config.rank)
            preflight_cov = sigma
        import torch

        status, eigengap = preflight_eigengap(
            torch.from_numpy(preflight_cov), config.rank or 1
        )
        preflight = status.value
    elif m == "D5":
        if config.nuisance_indices is None:
            raise ValueError("D5 requires nuisance_indices")
        x = np.asarray(args[0], dtype=np.float32)
        idx = np.asarray(config.nuisance_indices, dtype=int)
        block = x[:, idx] - x[:, idx].mean(0, keepdims=True)
        cov = (block.T @ block) / max(len(block), 1)
        sigma = np.zeros((x.shape[1], x.shape[1]), dtype=np.float32)
        sigma[np.ix_(idx, idx)] = cov
    else:
        raise ValueError(f"NumPy path supports D1, D2, D4, D5; got {m}")

    sigma = sigma.astype(np.float32)
    sigma = 0.5 * (sigma + sigma.T) + shrinkage * np.eye(sigma.shape[0], dtype=np.float32)

    import torch

    return SigmaTaskEstimate(
        sigma=torch.from_numpy(sigma),
        method=m,
        config=config,
        eigengap=eigengap,
        preflight=preflight,
    )


def torch_from_numpy(arr: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(np.asarray(arr, dtype=np.float32))
