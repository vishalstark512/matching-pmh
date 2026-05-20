"""Training helpers: PMH loss module and total loss assembly."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from pmh.artifact import SigmaTaskEstimate
from pmh.config import PMHConfig
from pmh.loss_budget import PMHLossBudget, budget_pmh_to_task_loss
from pmh.penalty import pmh_penalty_on_rep

# Training falsification modes (see docs/PAPER_ALIGNMENT.md)
# - matched: Phase-A Sigma_task
# - wrong_w: random subspace orthogonal to matched W (Lemma C; T1 protocol)
# - trace_iso / isotropic: (trace/d)*I from matched Sigma — NOT D2 nuisance, NOT sklearn D4 arm
PMHMode = Literal["matched", "wrong_w", "isotropic", "trace_iso"]


class PMHLoss(nn.Module):
    """Differentiable PMH term on representations ``h = encoder(x)``.

    Parameters
    ----------
    estimate : SigmaTaskEstimate or Tensor
        PSD matrix or loaded artifact (matched ``Sigma`` from Phase A).
    config : PMHConfig
        Weight, cap, probes, warmup schedule.
    mode : str
        ``'matched'`` | ``'wrong_w'`` | ``'trace_iso'`` (alias ``'isotropic'``).
    wrong_rank : int
        Rank for wrong-W control (Lemma C).
    wrong_seed : int
        RNG seed for wrong-W subspace (reproducible falsification).
    """

    def __init__(
        self,
        estimate: SigmaTaskEstimate | torch.Tensor,
        config: PMHConfig | None = None,
        *,
        mode: PMHMode = "matched",
        wrong_rank: int = 32,
        wrong_seed: int = 0,
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
        self.mode: PMHMode = "trace_iso" if mode == "isotropic" else mode  # type: ignore[assignment]
        self.wrong_rank = wrong_rank
        self.wrong_seed = wrong_seed
        self._epoch = 0
        self.last_budget: PMHLossBudget | None = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _matched_basis(self, d: int) -> torch.Tensor:
        """Top-r eigenvectors of matched ``sigma`` as W in R^{d x r}."""
        if self.estimate_meta is not None and "w" in self.estimate_meta.metadata:
            w = self.estimate_meta.metadata["w"]
            if isinstance(w, torch.Tensor) and w.shape[0] == d:
                r = min(self.wrong_rank, w.shape[1])
                return w[:, :r].to(device=self.sigma.device, dtype=self.sigma.dtype)
        _, evecs = torch.linalg.eigh(self.sigma.float())
        r = min(self.wrong_rank, d)
        return evecs[:, -r:].to(dtype=self.sigma.dtype)

    def _wrong_w_sigma(self, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Lemma C: Sigma' = Q Q^T with Q orthogonal to matched W."""
        rank = min(self.wrong_rank, d)
        w = self._matched_basis(d)
        gen = torch.Generator(device=device)
        gen.manual_seed(self.wrong_seed)
        m = torch.randn(d, rank, device=device, dtype=dtype, generator=gen)
        residual = m - w @ (w.T @ m)
        q, _ = torch.linalg.qr(residual)
        r_eff = min(rank, q.shape[1])
        q = q[:, :r_eff]
        return q @ q.T

    def _sigma_for_mode(self, h: torch.Tensor) -> torch.Tensor:
        d = h.shape[-1]
        if self.mode == "matched":
            return self.sigma
        if self.mode == "trace_iso":
            tr = self.sigma.trace().item() / max(d, 1)
            return tr * torch.eye(d, device=h.device, dtype=h.dtype)
        return self._wrong_w_sigma(d, h.device, h.dtype)

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
        """Return ``(total_loss, pmh_applied)`` with task-ratio cap (5--30% band)."""
        raw = self.forward(h)
        max_r = self.config.pmh_max_task_ratio
        if self.config.cap_basis == "task" and max_r > 0:
            applied, self.last_budget = budget_pmh_to_task_loss(raw, task_loss, self.config)
        elif self.config.cap_ratio > 0:
            from pmh.penalty import cap_pmh_term

            applied = cap_pmh_term(
                raw,
                task_loss,
                cap_ratio=self.config.cap_ratio,
                basis=self.config.cap_basis,
            )
            self.last_budget = None
        else:
            applied = raw
            self.last_budget = None
        return task_loss + applied, applied
