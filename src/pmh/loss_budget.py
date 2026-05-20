"""PMH vs task loss scale control (target ~5--30% of task loss on the PMH term)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pmh.config import PMHConfig


@dataclass(frozen=True)
class PMHLossBudget:
    """Diagnostics after applying the task-loss budget to the PMH term."""

    task_loss: float
    pmh_raw: float
    pmh_applied: float
    pmh_task_ratio: float
    capped: bool
    underpowered: bool
    overpowered: bool

    def summary_line(self) -> str:
        pct = 100.0 * self.pmh_task_ratio
        flags = []
        if self.capped:
            flags.append("capped")
        if self.underpowered:
            flags.append(f"below {100*0.05:.0f}% target")
        if self.overpowered:
            flags.append("over limit")
        extra = f" ({', '.join(flags)})" if flags else ""
        return f"PMH/task = {pct:.1f}%{extra}"


def pmh_task_ratio(pmh_term: torch.Tensor, task_loss: torch.Tensor) -> float:
    """PMH term divided by task loss (detached scalars)."""
    t = float(task_loss.detach())
    p = float(pmh_term.detach())
    if t < 1e-12:
        return 0.0
    return p / t


def budget_pmh_to_task_loss(
    pmh_term: torch.Tensor,
    task_loss: torch.Tensor,
    config: PMHConfig,
) -> tuple[torch.Tensor, PMHLossBudget]:
    """Hard-cap PMH so it does not exceed ``config.pmh_max_task_ratio`` × task loss.

    Default training target: PMH term is **5--30%** of task loss (see ``PMHConfig``).
    """
    lt = task_loss.detach().float()
    raw = pmh_term.float()
    max_r = float(config.pmh_max_task_ratio)
    min_r = float(config.pmh_min_task_ratio)

    if max_r <= 0:
        applied = raw
        capped = False
    else:
        cap_val = max_r * lt
        applied = torch.minimum(raw, cap_val)
        capped = bool((raw > cap_val + 1e-8).item())

    ratio = pmh_task_ratio(applied, task_loss)
    diag = PMHLossBudget(
        task_loss=float(lt),
        pmh_raw=float(raw),
        pmh_applied=float(applied.detach()),
        pmh_task_ratio=ratio,
        capped=capped,
        underpowered=ratio < min_r and min_r > 0,
        overpowered=capped,
    )
    return applied.to(dtype=pmh_term.dtype), diag


def format_loss_balance_line(diag: PMHLossBudget) -> str:
    return (
        f"task_loss={diag.task_loss:.4f}  pmh={diag.pmh_applied:.4f}  "
        f"{diag.summary_line()}"
    )


@torch.no_grad()
def suggest_pmh_weight(
    encoder,
    x: torch.Tensor,
    sigma: torch.Tensor,
    task_loss: torch.Tensor,
    *,
    target_ratio: float = 0.15,
    n_probes: int = 4,
    shrinkage: float = 1e-6,
) -> float:
    """One-shot weight so ``weight * penalty ≈ target_ratio × task_loss`` (probe batch)."""
    from pmh.penalty import pmh_penalty_on_rep

    h = encoder(x) if not isinstance(x, torch.Tensor) or x.shape[-1] != sigma.shape[0] else x
    if h.shape[-1] != sigma.shape[0]:
        h = encoder(x)
    pen = float(pmh_penalty_on_rep(h, sigma, n_probes=n_probes, shrinkage=shrinkage).detach())
    t = float(task_loss.detach())
    if pen < 1e-12 or t < 1e-12:
        return 0.3
    return max(1e-4, min(2.0, target_ratio * t / pen))
