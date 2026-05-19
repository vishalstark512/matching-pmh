"""Eigengap pre-flight checks before training."""

from __future__ import annotations

from enum import Enum

import torch

from pmh.config import PreflightConfig
from pmh.diagnostics import eigengap_ratio


class PreflightStatus(str, Enum):
    PASS = "pass"
    MARGINAL = "marginal"
    FAIL = "fail"


def preflight_eigengap(
    cov: torch.Tensor,
    rank: int,
    *,
    config: PreflightConfig | None = None,
) -> tuple[PreflightStatus, float]:
    """Classify eigengap gamma_r (paper Section 5.1).

    Returns (status, gamma_r).
    """
    cfg = config or PreflightConfig()
    gamma = eigengap_ratio(cov, rank)
    if gamma >= cfg.pass_ratio:
        return PreflightStatus.PASS, gamma
    if gamma >= cfg.fail_ratio:
        return PreflightStatus.MARGINAL, gamma
    return PreflightStatus.FAIL, gamma
