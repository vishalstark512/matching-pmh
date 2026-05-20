"""PGD / adversarial delta collection for T7B-style nuisance estimation."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from pmh.artifact import SigmaTaskEstimate
from pmh.calibrate.deltas import subspace_artifact_from_deltas

Encoder = Callable[[torch.Tensor], torch.Tensor]


def _batch_xy(item: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(item, (tuple, list)) or len(item) < 2:
        raise ValueError("PGD collection expects (x, y) batches")
    return item[0], item[1]


@torch.enable_grad()
def collect_pgd_feature_deltas(
    encoder: Encoder,
    batches: Iterable[torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    epsilon: float = 0.1,
    steps: int = 3,
    max_batches: int | None = 20,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Stack PGD attack deltas ``h_adv - h_clean`` as ``[N, d]`` (T7B estimate phase)."""
    dev = torch.device(device) if device is not None else None
    rows: list[torch.Tensor] = []
    n_batches = 0
    for item in batches:
        x, y = _batch_xy(item)
        if dev is not None:
            x, y = x.to(dev), y.to(dev)
        x_adv = x.detach().clone()
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            h = encoder(x_adv)
            if h.dim() != 2:
                raise ValueError("encoder must return [B, d] for PGD deltas")
            if loss_fn is not None:
                loss = loss_fn(h, y)
            else:
                loss = F.cross_entropy(h, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = (x_adv + epsilon * grad.sign()).detach()
        with torch.no_grad():
            h0 = encoder(x).detach().float()
            h1 = encoder(x_adv).detach().float()
        rows.append((h1 - h0).cpu())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    if not rows:
        raise ValueError("batches yielded no data")
    return torch.cat(rows, dim=0)


def estimate_pgd_subspace(
    encoder: Encoder,
    batches: Iterable[torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    rank: int = 16,
    epsilon: float = 0.1,
    steps: int = 3,
    max_batches: int | None = 20,
    device: torch.device | str | None = None,
    shrinkage: float = 1e-6,
) -> SigmaTaskEstimate:
    """Estimate attack-direction subspace from PGD feature deltas (library T7B path)."""
    deltas = collect_pgd_feature_deltas(
        encoder,
        batches,
        epsilon=epsilon,
        steps=steps,
        max_batches=max_batches,
        device=device,
    )
    return subspace_artifact_from_deltas(
        deltas,
        rank=rank,
        method="D7",
        shrinkage=shrinkage,
        metadata={"source": "pgd", "epsilon": epsilon, "steps": steps},
    )


def estimate_pgd_subspace_from_model(
    model: nn.Module,
    hook: str | nn.Module | Encoder,
    batches: Iterable[torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    head: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None = None,
    rank: int = 16,
    epsilon: float = 0.1,
    steps: int = 3,
    max_batches: int | None = 20,
    device: torch.device | str | None = None,
) -> SigmaTaskEstimate:
    """PGD on inputs; deltas are on hook representations ``h`` (T7B library path)."""
    from pmh.hooks import resolve_hook

    enc = resolve_hook(model, hook)
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    if head is not None:

        def loss_fn(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(head(h), y)  # type: ignore[misc]

    def rep_encoder(x: torch.Tensor) -> torch.Tensor:
        return enc(x)

    deltas = collect_pgd_feature_deltas(
        rep_encoder,
        batches,
        loss_fn=loss_fn,
        epsilon=epsilon,
        steps=steps,
        max_batches=max_batches,
        device=device,
    )
    return subspace_artifact_from_deltas(
        deltas,
        rank=rank,
        method="D7",
        metadata={"source": "pgd", "epsilon": epsilon, "steps": steps},
    )
