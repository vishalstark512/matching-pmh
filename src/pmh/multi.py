"""Multiple nuisance geometries (additive matched PMH)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from pmh.artifact import SigmaTaskEstimate
from pmh.config import PMHConfig
from pmh.training import PMHLoss


class MultiPMHLoss(nn.Module):
    """Sum of matched PMH penalties from several ``Sigma_task`` estimates.

    Use when deployment has **hybrid** nuisances (e.g. domain shift + augmentations):
    estimate one artifact per story, then one ``MultiPMHLoss`` in training.

    Parameters
    ----------
    estimates : sequence of SigmaTaskEstimate or PMHLoss
    configs : optional per-term PMHConfig (else shared ``config``)
    weights : optional per-term scalar multipliers on top of PMHConfig.weight
    """

    def __init__(
        self,
        estimates: Sequence[SigmaTaskEstimate | PMHLoss],
        config: PMHConfig | None = None,
        *,
        configs: Sequence[PMHConfig] | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self._terms: nn.ModuleList = nn.ModuleList()
        for i, est in enumerate(estimates):
            if isinstance(est, PMHLoss):
                self._terms.append(est)
            else:
                cfg = configs[i] if configs is not None else config
                self._terms.append(PMHLoss(est, cfg))
        self._weights = list(weights) if weights is not None else [1.0] * len(self._terms)
        if len(self._weights) != len(self._terms):
            raise ValueError("weights length must match number of estimates")
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        for t in self._terms:
            t.set_epoch(epoch)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        total = h.new_zeros(())
        for term, w in zip(self._terms, self._weights):
            total = total + w * term.forward(h)
        return total

    def capped_total(
        self,
        task_loss: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cap to the **combined** raw PMH sum (single cap vs task loss)."""
        raw = self.forward(h)
        cfg = self._terms[0].config if self._terms else None
        if cfg is not None and cfg.cap_basis == "task" and cfg.pmh_max_task_ratio > 0:
            from pmh.loss_budget import budget_pmh_to_task_loss

            raw, _ = budget_pmh_to_task_loss(raw, task_loss, cfg)
        elif cfg is not None and cfg.cap_ratio > 0:
            from pmh.penalty import cap_pmh_term

            raw = cap_pmh_term(
                raw,
                task_loss,
                cap_ratio=cfg.cap_ratio,
                basis=cfg.cap_basis,
            )
        return task_loss + raw, raw
