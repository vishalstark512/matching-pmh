"""Multilayer feature-difference PMH (ResNet / ViT style)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn as nn

from pmh.config import PMHConfig
from pmh.penalty import cap_pmh_term, pmh_penalty_feature_diff


def gram_sample_noise(
    feats: torch.Tensor,
    gram: torch.Tensor,
    std: float,
    *,
    rank: int = 64,
) -> torch.Tensor:
    """Sample feature noise with covariance proportional to ``gram`` (domain D4 sampling)."""
    c = gram.shape[0]
    r = min(rank, c)
    try:
        evals, evecs = torch.linalg.eigh(gram.float())
        evals = evals[-r:].clamp(min=0.0)
        evecs = evecs[:, -r:]
    except Exception:
        return torch.randn_like(feats) * std

    total = evals.sum().item()
    if total < 1e-12:
        return torch.randn_like(feats) * std
    scale = std / (total**0.5 + 1e-12)

    if feats.dim() == 4:
        b, ch, h, w = feats.shape
        z = torch.randn(b, h * w, r, device=feats.device, dtype=torch.float32)
        w_mat = (evecs * evals.sqrt()).T.to(feats.device)
        delta = (z @ w_mat).permute(0, 2, 1).view(b, ch, h, w) * scale
    else:
        z = torch.randn(feats.shape[0], r, device=feats.device, dtype=torch.float32)
        w_mat = (evecs * evals.sqrt()).T.to(feats.device)
        delta = (z @ w_mat) * scale
    return delta.to(dtype=feats.dtype)


class MultiLayerPMHLoss(nn.Module):
    """Average PMH feature-diff across named layers (vision blocks).

    Parameters
    ----------
    layer_names : sequence of str
        Keys present in clean/noisy feature dicts.
    config : PMHConfig
        Global weight, cap (applied outside on total), warmup.
    normalize : bool
        L2-normalise pooled features per layer before MSE.
    """

    def __init__(
        self,
        layer_names: Sequence[str],
        config: PMHConfig | None = None,
        *,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.layer_names = tuple(layer_names)
        self.config = config or PMHConfig()
        self.normalize = normalize
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def forward(
        self,
        feats_clean: Mapping[str, torch.Tensor],
        feats_noisy: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        w = self.config.pmh_weight_for_epoch(self._epoch) * self.config.weight
        if w <= 0:
            ref = next(iter(feats_clean.values()))
            return ref.new_zeros(())

        losses: list[torch.Tensor] = []
        for name in self.layer_names:
            if name not in feats_clean or name not in feats_noisy:
                continue
            losses.append(
                pmh_penalty_feature_diff(
                    feats_clean[name],
                    feats_noisy[name],
                    normalize=self.normalize,
                )
            )
        if not losses:
            raise KeyError(f"no layers matched {self.layer_names}")
        return w * torch.stack(losses).mean()

    def capped_total(
        self,
        task_loss: torch.Tensor,
        feats_clean: Mapping[str, torch.Tensor],
        feats_noisy: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.forward(feats_clean, feats_noisy)
        if self.config.cap_basis == "task" and self.config.pmh_max_task_ratio > 0:
            from pmh.loss_budget import budget_pmh_to_task_loss

            raw, _ = budget_pmh_to_task_loss(raw, task_loss, self.config)
        elif self.config.cap_ratio > 0:
            raw = cap_pmh_term(
                raw,
                task_loss,
                cap_ratio=self.config.cap_ratio,
                basis=self.config.cap_basis,
            )
        return task_loss + raw, raw


def make_noise_forward(
    model: nn.Module,
    noise_fns: Mapping[str, Callable[[torch.Tensor], torch.Tensor]],
    forward_features: Callable[[torch.Tensor], dict[str, torch.Tensor]],
) -> Callable[[torch.Tensor], dict[str, torch.Tensor]]:
    """Wrap ``forward_features`` with per-layer additive noise hooks."""

    def _forward(x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = forward_features(x)
        out = {}
        for k, v in feats.items():
            if k in noise_fns:
                out[k] = v + noise_fns[k](v)
            else:
                out[k] = v
        return out

    return _forward
