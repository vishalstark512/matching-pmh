"""Unified entry point: estimate_sigma_task(..., method='Dk')."""

from __future__ import annotations

from typing import Any

import torch

from pmh.artifact import SigmaTaskEstimate
from pmh.config import Method, SigmaTaskConfig
from pmh.estimators import (
    estimate_d1,
    estimate_d2,
    estimate_d3,
    estimate_d4,
    estimate_d5,
    estimate_d6,
    estimate_d7,
)
from pmh.estimators.d4_domain import gram_from_diff
from pmh.preflight import preflight_eigengap


def estimate_from_config(
    config: SigmaTaskConfig,
    *args: Any,
    **kwargs: Any,
) -> SigmaTaskEstimate:
    """Estimate using a :class:`SigmaTaskConfig` and return a saveable artifact."""
    sigma = _dispatch(config, *args, **kwargs)
    eigengap: float | None = None
    preflight: str | None = None
    rank = config.rank or 1

    if config.method in ("D1", "D4", "D7") and config.rank is not None:
        cov = gram_from_diff(args[0], args[1]) if config.method in ("D1", "D4") and len(args) >= 2 else sigma
        if config.method == "D7" and isinstance(args[0], torch.Tensor):
            cov = sigma
        if config.method in ("D1", "D4"):

            status, eigengap = preflight_eigengap(cov, rank)
            preflight = status.value

    meta = dict(config.extra)
    meta.update(kwargs.get("metadata", {}))
    return SigmaTaskEstimate(
        sigma=sigma,
        method=config.method,
        config=config,
        eigengap=eigengap,
        preflight=preflight,
        metadata=meta,
    )


def estimate_sigma_task(
    *args: Any,
    method: Method | str = "D4",
    rank: int | None = None,
    shrinkage: float = 1e-6,
    return_artifact: bool = False,
    config: SigmaTaskConfig | None = None,
    **kwargs: Any,
) -> torch.Tensor | SigmaTaskEstimate:
    """Estimate deployment nuisance covariance Sigma_task (Lemmas D1--D7).

    Pass a :class:`SigmaTaskConfig` as ``config=`` **or** use ``method=`` + kwargs.

    Set ``return_artifact=True`` to get :class:`SigmaTaskEstimate` (save/load, eigengap).
    """
    if config is None:
        cfg_keys = {
            "dim",
            "noise_level",
            "nuisance_indices",
            "encoder",
            "device",
            "dtype",
            "extra",
        }
        cfg_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in cfg_keys}
        config = SigmaTaskConfig(
            method=method,  # type: ignore[arg-type]
            rank=rank,
            shrinkage=shrinkage,
            **cfg_kw,
        )
    else:
        if rank is not None:
            config.rank = rank
        config.shrinkage = shrinkage

    artifact = estimate_from_config(config, *args, **kwargs)
    return artifact if return_artifact else artifact.sigma


def _dispatch(config: SigmaTaskConfig, *args: Any, **kwargs: Any) -> torch.Tensor:
    m = config.method
    shrinkage = config.shrinkage
    dtype = getattr(torch, config.dtype, torch.float32)
    device = torch.device(config.device) if config.device else None

    if m == "D2":
        if config.dim is None:
            raise ValueError("D2 requires config.dim")
        nl = config.noise_level if config.noise_level is not None else 0.1
        return estimate_d2(
            dim=int(config.dim),
            noise_level=float(nl),
            device=device,
            dtype=dtype,
        )

    if m == "D3":
        aug = kwargs.get("aug_deltas")
        if aug is None and args:
            aug = args[0]
        if aug is None:
            raise ValueError("D3: pass aug_deltas")
        return estimate_d3(aug, shrinkage=shrinkage)

    if m in ("D1", "D4"):
        if len(args) < 2:
            raise ValueError(f"{m}: pass source and target features [N, d]")
        fn = estimate_d1 if m == "D1" else estimate_d4
        kw = {"rank": int(config.rank), "shrinkage": shrinkage} if m == "D1" else {
            "rank": config.rank,
            "shrinkage": shrinkage,
        }
        return fn(args[0], args[1], **kw)

    if m == "D5":
        if len(args) < 1:
            raise ValueError("D5: pass features [N, d]")
        if config.nuisance_indices is None:
            raise ValueError("D5: config.nuisance_indices required")
        return estimate_d5(args[0], config.nuisance_indices, shrinkage=shrinkage)

    if m == "D6":
        seq = args[0] if args else kwargs.get("sequences")
        if seq is None:
            raise ValueError("D6: pass sequences")
        return estimate_d6(seq, shrinkage=shrinkage)

    if m == "D7":
        data = args[0] if args else kwargs.get("style_pairs")
        if data is None:
            raise ValueError("D7: pass style_pairs")
        enc = config.encoder or kwargs.get("encoder")
        return estimate_d7(
            data,
            encoder=enc,
            rank=config.rank if config.rank is not None else 128,
            shrinkage=shrinkage,
        )

    raise ValueError(f"Unknown method {m!r}")
