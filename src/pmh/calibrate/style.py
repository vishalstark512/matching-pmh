"""D7 style-pair Gram (T7A)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from pmh.artifact import SigmaTaskEstimate
from pmh.config import SigmaTaskConfig
from pmh.estimators.d7_alignment import estimate_d7


def style_gram_from_deltas(
    deltas: torch.Tensor,
    *,
    rank: int = 128,
    shrinkage: float = 0.1,
    metadata: dict[str, Any] | None = None,
) -> SigmaTaskEstimate:
    """Covariance of style embedding deltas [M, d] (paper T7A behavioral protocol)."""
    sigma = estimate_d7(deltas, rank=rank, shrinkage=shrinkage)
    meta = dict(metadata or {})
    _, _, vt = torch.linalg.svd(deltas.float() - deltas.mean(0, keepdim=True), full_matrices=False)
    r = min(rank, vt.shape[0])
    meta["w"] = vt[:r].T.contiguous()
    return SigmaTaskEstimate(
        sigma=sigma,
        method="D7",
        config=SigmaTaskConfig.for_alignment(rank=rank, shrinkage=shrinkage),
        metadata=meta,
    )


def style_gram_from_jsonl(
    records: Sequence[dict[str, Any]],
    encoder,
    *,
    rank: int = 128,
    shrinkage: float = 0.1,
) -> SigmaTaskEstimate:
    """Estimate from HF-style JSONL records (prompt + style_variants)."""
    sigma = estimate_d7(records, encoder=encoder, rank=rank, shrinkage=shrinkage)
    return SigmaTaskEstimate(
        sigma=sigma,
        method="D7",
        config=SigmaTaskConfig.for_alignment(rank=rank, shrinkage=shrinkage),
    )
