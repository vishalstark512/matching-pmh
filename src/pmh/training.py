"""Training helpers: PMH loss module and total loss assembly."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from pmh.artifact import SigmaTaskEstimate
from pmh.config import PMHConfig
from pmh.controls import wrong_W_projector
from pmh.penalty import cap_pmh_term, pmh_penalty_on_rep


class PMHLoss(nn.Module):
    """Differentiable PMH term on representations ``h = encoder(x)``.

    Parameters
    ----------
    estimate : SigmaTaskEstimate or Tensor
        PSD matrix or loaded artifact.
    config : PMHConfig
        Weight, cap, probes, warmup schedule.
    mode : str
        ``'matched'`` | ``'wrong_w'`` | ``'isotropic'`` (uses sigma = sigma.trace()/d * I for iso).
    wrong_rank : int
        Rank for random wrong-W control (Lemma C).
    """

    def __init__(
        self,
        estimate: SigmaTaskEstimate | torch.Tensor,
        config: PMHConfig | None = None,
        *,
        mode: Literal["matched", "wrong_w", "isotropic"] = "matched",
        wrong_rank: int = 32,
    ) -> None:
        super().__init__()
        self.config = config or PMHConfig()
        if isinstance(estimate, SigmaTaskEstimate):
            sigma = estimate.sigma
            self.estimate_meta = estimate
        else:
            sigma = estimate
            self.estimate_meta = None
        self.register_buffer("sigma", sigma.detach())
        self.mode = mode
        self.wrong_rank = wrong_rank
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _sigma_for_mode(self, h: torch.Tensor) -> torch.Tensor:
        d = h.shape[-1]
        if self.mode == "matched":
            return self.sigma
        if self.mode == "isotropic":
            tr = self.sigma.trace().item() / max(d, 1)
            return tr * torch.eye(d, device=h.device, dtype=h.dtype)
        # wrong_w — rank cannot exceed representation dimension
        rank = min(self.wrong_rank, d)
        U = wrong_W_projector(d, rank, device=h.device, dtype=h.dtype)
        return U @ U.T

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = self.config.pmh_weight_for_epoch(self._epoch) * self.config.weight
        if w <= 0:
            return h.new_zeros(())
        pen = pmh_penalty_on_rep(
            h,
            self._sigma_for_mode(h),
            n_probes=self.config.n_probes,
            shrinkage=self.config.shrinkage,
        )
        return w * pen

    def capped_total(
        self,
        task_loss: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(total_loss, raw_pmh_term)`` with cap applied."""
        raw = self.forward(h)
        if self.config.cap_ratio > 0:
            raw = cap_pmh_term(
                raw,
                task_loss,
                cap_ratio=self.config.cap_ratio,
                basis=self.config.cap_basis,
            )
        return task_loss + raw, raw
