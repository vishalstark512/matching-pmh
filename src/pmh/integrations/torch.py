"""PyTorch training-loop integration for matched PMH."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from pmh.artifact import SigmaTaskEstimate
from pmh.config import PMHConfig
from pmh.training import PMHLoss

try:
    from pmh.multi import MultiPMHLoss

    _PMHLossLike = PMHLoss | MultiPMHLoss
except ImportError:
    _PMHLossLike = PMHLoss  # type: ignore[misc,assignment]

Encoder = Callable[[torch.Tensor], torch.Tensor]
Batch = torch.Tensor | tuple[torch.Tensor, ...]
BatchIterable = Iterable[Batch]


@dataclass
class PMHStepResult:
    """Scalars from one training step."""

    task_loss: float
    pmh_loss: float
    total_loss: float
    pmh_task_ratio: float | None = None
    pmh_capped: bool = False
    pmh_underpowered: bool = False


@dataclass
class PMHEpochStats:
    """Aggregated metrics after an epoch."""

    task_loss: float = 0.0
    pmh_loss: float = 0.0
    total_loss: float = 0.0
    n_steps: int = 0
    _ratio_sum: float = 0.0
    _ratio_n: int = 0

    def update(self, step: PMHStepResult) -> None:
        self.task_loss += step.task_loss
        self.pmh_loss += step.pmh_loss
        self.total_loss += step.total_loss
        self.n_steps += 1
        if step.pmh_task_ratio is not None:
            self._ratio_sum += step.pmh_task_ratio
            self._ratio_n += 1

    def averages(self) -> dict[str, float]:
        n = max(self.n_steps, 1)
        out = {
            "task_loss": self.task_loss / n,
            "pmh_loss": self.pmh_loss / n,
            "total_loss": self.total_loss / n,
            "n_steps": float(self.n_steps),
        }
        if self._ratio_n > 0:
            out["pmh_task_ratio"] = self._ratio_sum / self._ratio_n
        return out


class PMHCallback:
    """Hook matched PMH into a manual PyTorch training loop.

    Parameters
    ----------
    pmh_loss : PMHLoss
        Configured penalty (matched / wrong_w / isotropic).
    encoder : callable
        ``h = encoder(x)`` representation used for PMH (often ``model.backbone``).
    head : callable, optional
        If given, ``logits = head(h)``; otherwise ``encoder`` must return logits.
    """

    def __init__(
        self,
        pmh_loss: _PMHLossLike,
        encoder: Encoder,
        *,
        head: Callable[[torch.Tensor], torch.Tensor] | None = None,
        task_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.pmh_loss = pmh_loss
        self.encoder = encoder
        self.head = head
        self.task_loss_fn = task_loss_fn or nn.functional.cross_entropy
        self._epoch = 0
        self._stats = PMHEpochStats()

    @classmethod
    def from_artifact(
        cls,
        artifact: SigmaTaskEstimate,
        encoder: Encoder,
        pmh_config: PMHConfig | None = None,
        **kwargs: Any,
    ) -> PMHCallback:
        return cls(PMHLoss(artifact, pmh_config), encoder, **kwargs)

    def on_epoch_start(self, epoch: int) -> None:
        self._epoch = epoch
        self.pmh_loss.set_epoch(epoch)
        self._stats = PMHEpochStats()

    def on_epoch_end(self) -> dict[str, float]:
        return self._stats.averages()

    def training_step(
        self,
        batch: Batch,
        *,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PMHStepResult]:
        """Compute capped ``task + PMH`` loss for one batch.

        ``batch`` may be ``x`` or ``(x, y, ...)``.
        """
        if isinstance(batch, (tuple, list)):
            x = batch[0]
            y = targets if targets is not None else batch[1]
        else:
            x = batch
            if targets is None:
                raise ValueError("pass targets= when batch is only features")
            y = targets

        h = self.encoder(x)
        logits = self.head(h) if self.head is not None else h
        task = self.task_loss_fn(logits, y)
        total, pmh_raw = self.pmh_loss.capped_total(task, h)
        budget = getattr(self.pmh_loss, "last_budget", None)
        ratio = budget.pmh_task_ratio if budget is not None else None
        if ratio is None and float(task.detach()) > 1e-12:
            ratio = float(pmh_raw.detach()) / float(task.detach())
        step = PMHStepResult(
            task_loss=float(task.detach()),
            pmh_loss=float(pmh_raw.detach()),
            total_loss=float(total.detach()),
            pmh_task_ratio=ratio,
            pmh_capped=budget.capped if budget else False,
            pmh_underpowered=budget.underpowered if budget else False,
        )
        if budget is not None and budget.underpowered:
            cfg = getattr(self.pmh_loss, "config", None)
            if cfg is not None and getattr(cfg, "warn_underpowered_pmh", True):
                import warnings

                warnings.warn(
                    f"PMH term is {100*budget.pmh_task_ratio:.1f}% of task loss "
                    f"(target {100*cfg.pmh_min_task_ratio:.0f}--"
                    f"{100*cfg.pmh_max_task_ratio:.0f}%). "
                    "Increase PMHConfig.weight or check hook/estimate.",
                    stacklevel=2,
                )
        self._stats.update(step)
        return total, step


def train_epoch_with_pmh(
    model: nn.Module,
    callback: PMHCallback,
    dataloader: BatchIterable,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    device: torch.device | str | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Run one training epoch: ``zero_grad → backward → step`` per batch."""
    dev = torch.device(device) if device is not None else None
    model.train()
    callback.on_epoch_start(epoch)
    steps = 0
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = tuple(t.to(dev) if dev and torch.is_tensor(t) else t for t in batch)
        elif torch.is_tensor(batch) and dev is not None:
            batch = batch.to(dev)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = callback.training_step(batch)
        loss.backward()
        optimizer.step()
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return callback.on_epoch_end()


def infinite_loader(loader: Iterable[Batch]) -> Iterator[Batch]:
    """Cycle a DataLoader indefinitely (for small examples)."""
    while True:
        yield from loader
