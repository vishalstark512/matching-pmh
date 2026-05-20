"""E1 multiscale domain PMH (paper T4B): per-layer Gram + feature-diff train.

Use when deploy shift is visual domain / texture and you have **labeled** source and
target mini-batches (class-aligned Gram per layer). Default ``PMHTrainer`` + single-hook
``PMHLoss`` is the Jacobian path; this module wires the paper's **feature-diff** loop.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pmh.config import PMHConfig
from pmh.estimators.d4_domain import estimate_d4_from_paired_diffs
from pmh.features import collect_domain_paired_diffs
from pmh.integrations.torch import PMHEpochStats, PMHStepResult
from pmh.vision.multilayer import MultiLayerPMHLoss, gram_sample_noise, make_noise_forward

EncoderDict = Callable[[torch.Tensor], dict[str, torch.Tensor]]
HeadFn = Callable[[torch.Tensor], torch.Tensor]


@torch.no_grad()
def estimate_multilayer_domain_sigmas(
    forward_features: EncoderDict,
    source_batches: Iterable,
    target_batches: Iterable,
    layer_names: Sequence[str],
    *,
    rank: int = 64,
    max_batches: int = 50,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Class-aligned D4 Gram per layer (paper E1_multiscale estimate phase)."""

    def _enc_layer(layer: str):
        def _fn(x: torch.Tensor) -> torch.Tensor:
            feats = forward_features(x)
            t = feats[layer]
            if t.dim() > 2:
                t = t.flatten(1)
            return t

        return _fn

    sigmas: dict[str, torch.Tensor] = {}
    for name in layer_names:
        diff, _ = collect_domain_paired_diffs(
            _enc_layer(name),
            source_batches,
            target_batches,
            align_by_class=True,
            max_batches=max_batches,
            device=device,
        )
        sigmas[name] = estimate_d4_from_paired_diffs(diff, rank=rank)
    return sigmas


def build_multilayer_domain_trainer(
    model: nn.Module,
    forward_features: EncoderDict,
    layer_sigmas: Mapping[str, torch.Tensor],
    layer_names: Sequence[str],
    *,
    pmh_config: PMHConfig | None = None,
    noise_std: float = 0.05,
    noise_rank: int = 64,
) -> tuple[MultiLayerPMHLoss, Callable[[torch.Tensor], dict[str, torch.Tensor]]]:
    """Return ``(loss, noisy_forward)`` for one training step (clean vs noisy features)."""
    noise_fns = {
        k: (lambda feats, g=g, s=noise_std, r=noise_rank: gram_sample_noise(feats, g, s, rank=r))
        for k, g in layer_sigmas.items()
        if k in layer_names
    }
    noisy_forward = make_noise_forward(model, noise_fns, forward_features)
    loss = MultiLayerPMHLoss(layer_names, pmh_config or PMHConfig())
    return loss, noisy_forward


class FeatureDiffCallback:
    """Paper E1_multiscale train step: clean features (anchor) vs Gram-noisy features."""

    def __init__(
        self,
        model: nn.Module,
        forward_features: EncoderDict,
        noisy_forward: EncoderDict,
        pmh_loss: MultiLayerPMHLoss,
        *,
        head: HeadFn | None = None,
        head_layer: str | None = None,
        layer_names: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.forward_features = forward_features
        self.noisy_forward = noisy_forward
        self.pmh_loss = pmh_loss
        self.head = head
        self.layer_names = tuple(layer_names or pmh_loss.layer_names)
        self.head_layer = head_layer or (self.layer_names[-1] if self.layer_names else None)
        self._epoch = 0
        self._stats = PMHEpochStats()

    def on_epoch_start(self, epoch: int) -> None:
        self._epoch = epoch
        self.pmh_loss.set_epoch(epoch)
        self._stats = PMHEpochStats()

    def on_epoch_end(self) -> dict[str, float]:
        return self._stats.averages()

    def training_step(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, PMHStepResult]:
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        else:
            raise ValueError("feature_diff training expects (x, y) batches")
        feats_clean = self.forward_features(x)
        feats_anchor = {k: v.detach() for k, v in feats_clean.items()}
        feats_noisy = self.noisy_forward(x)
        if self.head_layer is None or self.head_layer not in feats_clean:
            raise KeyError(f"head_layer {self.head_layer!r} missing from forward_features")
        h = feats_clean[self.head_layer]
        logits = self.head(h) if self.head is not None else h
        task = F.cross_entropy(logits, y)
        total, pmh_raw = self.pmh_loss.capped_total(task, feats_anchor, feats_noisy)
        step = PMHStepResult(
            task_loss=float(task.detach()),
            pmh_loss=float(pmh_raw.detach()),
            total_loss=float(total.detach()),
        )
        self._stats.update(step)
        return total, step


def train_epoch_feature_diff(
    callback: FeatureDiffCallback,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    device: torch.device | str | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    """One epoch of feature-diff PMH (clean anchor + noisy forward)."""
    dev = torch.device(device) if device is not None else None
    callback.model.train()
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
