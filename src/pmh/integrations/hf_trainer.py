"""Hugging Face ``Trainer`` integration for matched PMH."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from pmh.artifact import SigmaTaskEstimate
from pmh.config import PMHConfig
from pmh.training import PMHLoss

RepresentationFn = Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor]


def _require_transformers() -> tuple[Any, Any]:
    import os

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    try:
        from transformers import Trainer, TrainerCallback
    except ImportError as exc:
        raise ImportError(
            'HF Trainer integration requires transformers. '
            'pip install "matching-pmh[hf]"'
        ) from exc
    return Trainer, TrainerCallback


def compute_pmh_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    pmh_loss: PMHLoss,
    *,
    representation_fn: RepresentationFn | None = None,
    task_loss_fn: Callable[[Any, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Task + capped PMH without importing ``Trainer`` (useful for custom loops)."""
    labels = inputs.get("labels")
    outputs = model(**inputs)
    if task_loss_fn is not None:
        task_loss = task_loss_fn(outputs, labels)
    else:
        logits = getattr(outputs, "logits", None)
        if logits is None or labels is None:
            raise ValueError("need logits and labels for default task loss")
        task_loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
    rep_fn = representation_fn or default_representation_fn
    h = rep_fn(model, inputs)
    total, pmh_term = pmh_loss.capped_total(task_loss, h)
    return total, task_loss, pmh_term


def default_representation_fn(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    hidden_state_index: int = -1,
    pool: str = "last",
) -> torch.Tensor:
    """Last-token (or mean-pooled) hidden state from ``output_hidden_states``."""
    attention_mask = inputs.get("attention_mask")
    out = model(**{**inputs, "output_hidden_states": True})
    hidden = out.hidden_states[hidden_state_index].float()
    if hidden.dim() == 2:
        return hidden
    if attention_mask is not None:
        if pool == "last":
            last_pos = attention_mask.sum(dim=1).clamp(min=1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), last_pos]
        if pool == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return hidden[:, -1, :]


def _build_epoch_callback(pmh_loss: PMHLoss, trainer_callback_base: type) -> Any:
    class _EpochPMHCallback(trainer_callback_base):
        def on_epoch_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            epoch = int(state.epoch) if state.epoch is not None else 0
            pmh_loss.set_epoch(epoch)

    return _EpochPMHCallback()


def create_pmh_trainer_class() -> type:
    Trainer, TrainerCallback = _require_transformers()

    class PMHTrainer(Trainer):
        """``Trainer`` with capped PMH on representations."""

        def __init__(
            self,
            pmh_loss: PMHLoss,
            *,
            representation_fn: RepresentationFn | None = None,
            task_loss_fn: Callable[[Any, torch.Tensor], torch.Tensor] | None = None,
            pmh_log_key: str = "pmh_loss",
            **kwargs: Any,
        ) -> None:
            self.pmh_loss = pmh_loss
            self.representation_fn = representation_fn or default_representation_fn
            self.task_loss_fn = task_loss_fn
            self.pmh_log_key = pmh_log_key
            callbacks = list(kwargs.pop("callbacks", None) or [])
            callbacks.append(_build_epoch_callback(pmh_loss, TrainerCallback))
            kwargs["callbacks"] = callbacks
            super().__init__(**kwargs)

        @classmethod
        def from_artifact(
            cls,
            artifact: SigmaTaskEstimate,
            pmh_config: PMHConfig | None = None,
            **kwargs: Any,
        ) -> PMHTrainer:
            return cls(PMHLoss(artifact, pmh_config), **kwargs)

        def compute_loss(
            self,
            model: nn.Module,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
            **kwargs: Any,
        ) -> torch.Tensor | tuple[torch.Tensor, Any]:
            labels = inputs.get("labels")
            outputs = model(**inputs)
            if self.task_loss_fn is not None:
                task_loss = self.task_loss_fn(outputs, labels)
            else:
                if labels is None:
                    raise ValueError("inputs must include labels for default CE loss")
                logits = getattr(outputs, "logits", None)
                if logits is None:
                    raise ValueError("model outputs must expose .logits for default task_loss_fn")
                task_loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            total, task_loss, pmh_term = compute_pmh_training_loss(
                model,
                inputs,
                self.pmh_loss,
                representation_fn=self.representation_fn,
                task_loss_fn=self.task_loss_fn,
            )
            del task_loss
            self.log({self.pmh_log_key: pmh_term.detach()})
            return (total, outputs) if return_outputs else total

    return PMHTrainer


# Lazy singleton for imports
PMHTrainer = None  # set on first access


def get_pmh_trainer() -> type:
    global PMHTrainer
    if PMHTrainer is None:
        PMHTrainer = create_pmh_trainer_class()
    return PMHTrainer
