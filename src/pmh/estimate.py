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
    sigma, dispatch_meta = _dispatch(config, *args, **kwargs)
    eigengap: float | None = None
    preflight: str | None = None
    rank = config.rank or 1

    if config.method in ("D1", "D4", "D7") and config.rank is not None:
        if config.method == "D1":
            cov = sigma
        elif config.method == "D4" and len(args) >= 2:
            cov = gram_from_diff(args[0], args[1])
        elif config.method == "D7":
            cov = sigma
        else:
            cov = None
        if cov is not None:
            status, eigengap = preflight_eigengap(cov, rank)
            preflight = status.value

    meta = dict(config.extra)
    meta.update(kwargs.get("metadata", {}))
    meta.update(dispatch_meta)
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

    **D1** requires ``x_src, y_src, x_tgt, y_tgt`` (four positional tensors) or
    ``x_src, x_tgt`` with ``y_src=``, ``y_tgt=`` keywords.

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


def _dispatch(
    config: SigmaTaskConfig, *args: Any, **kwargs: Any
) -> tuple[torch.Tensor, dict[str, Any]]:
    m = config.method
    shrinkage = config.shrinkage
    dtype = getattr(torch, config.dtype, torch.float32)
    device = torch.device(config.device) if config.device else None
    meta: dict[str, Any] = {}

    if m == "D2":
        if config.dim is None:
            raise ValueError("D2 requires config.dim")
        nl = config.noise_level if config.noise_level is not None else 0.1
        return estimate_d2(
            dim=int(config.dim),
            noise_level=float(nl),
            device=device,
            dtype=dtype,
        ), meta

    if m == "D3":
        aug = kwargs.get("aug_deltas")
        if aug is None and args:
            aug = args[0]
        if aug is None:
            raise ValueError("D3: pass aug_deltas")
        return estimate_d3(aug, shrinkage=shrinkage), meta

    if m == "D1":
        if config.rank is None:
            raise ValueError("D1 requires config.rank")
        if len(args) >= 4:
            x_src, y_src, x_tgt, y_tgt = args[0], args[1], args[2], args[3]
        elif len(args) >= 2:
            x_src, x_tgt = args[0], args[1]
            y_src = kwargs.get("y_src")
            y_tgt = kwargs.get("y_tgt")
            if y_src is None or y_tgt is None:
                raise ValueError(
                    "D1 requires class labels on both domains: pass "
                    "(x_src, y_src, x_tgt, y_tgt) or (x_src, x_tgt, y_src=..., y_tgt=...). "
                    "For unlabeled domain Gram use method='D4'."
                )
        else:
            raise ValueError(
                "D1: pass (x_src, y_src, x_tgt, y_tgt). For unlabeled features use method='D4'."
            )
        sigma, w = estimate_d1(
            x_src,
            y_src,
            x_tgt,
            y_tgt,
            rank=int(config.rank),
            shrinkage=shrinkage,
            seed=int(kwargs.get("seed", 0)),
            n_pairs_per_class=int(kwargs.get("n_pairs_per_class", 100)),
            include_mean_shift=bool(kwargs.get("include_mean_shift", True)),
        )
        meta["w"] = w
        return sigma, meta

    if m == "D4":
        if len(args) < 2:
            raise ValueError("D4: pass source and target features [N, d]")
        return (
            estimate_d4(
                args[0],
                args[1],
                rank=config.rank,
                shrinkage=shrinkage,
            ),
            meta,
        )

    if m == "D5":
        if len(args) < 1:
            raise ValueError("D5: pass features [N, d]")
        if config.nuisance_indices is None:
            raise ValueError("D5: config.nuisance_indices required")
        return estimate_d5(args[0], config.nuisance_indices, shrinkage=shrinkage), meta

    if m == "D6":
        seq = args[0] if args else kwargs.get("sequences")
        if seq is None:
            raise ValueError("D6: pass sequences")
        return estimate_d6(seq, shrinkage=shrinkage), meta

    if m == "D7":
        data = args[0] if args else kwargs.get("style_pairs")
        if data is None:
            raise ValueError("D7: pass style_pairs")
        enc = config.encoder or kwargs.get("encoder")
        return (
            estimate_d7(
                data,
                encoder=enc,
                rank=config.rank if config.rank is not None else 128,
                shrinkage=shrinkage,
            ),
            meta,
        )

    raise ValueError(f"Unknown method {m!r}")
