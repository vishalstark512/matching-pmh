"""Collect representation batches from PyTorch data pipelines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence

import torch

Encoder = Callable[[torch.Tensor], torch.Tensor]
AugFn = Callable[[torch.Tensor], torch.Tensor]

__all__ = [
    "collect_features",
    "collect_labeled_features",
    "collect_sequence_features",
    "collect_augmentation_deltas",
    "paired_batches",
    "Encoder",
    "AugFn",
]


@torch.no_grad()
def _batch_x(item: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return item[0] if isinstance(item, (tuple, list)) else item


@torch.no_grad()
def collect_features(
    encoder: Encoder,
    batches: Iterable[torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    max_batches: int | None = 50,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Stack encoder outputs into ``[N, d]`` for ``estimate_sigma_task``."""
    dev = torch.device(device) if device is not None else None
    rows: list[torch.Tensor] = []
    n_batches = 0
    for item in batches:
        x = _batch_x(item)
        if dev is not None:
            x = x.to(dev)
        h = encoder(x).detach().float().to(dtype=dtype)
        if h.dim() != 2:
            raise ValueError(f"encoder must return [B, d], got shape {tuple(h.shape)}")
        rows.append(h.cpu())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    if not rows:
        raise ValueError("batches yielded no data")
    return torch.cat(rows, dim=0)


@torch.no_grad()
def collect_labeled_features(
    encoder: Encoder,
    batches: Iterable[torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    max_batches: int | None = 50,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(h, y)`` with ``h`` ``[N, d]``, ``y`` ``[N]``."""
    dev = torch.device(device) if device is not None else None
    hs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    n_batches = 0
    for item in batches:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("labeled batches must be (x, y) tuples")
        x, y = item[0], item[1]
        if dev is not None:
            x, y = x.to(dev), y.to(dev)
        h = encoder(x).detach().float().cpu()
        if h.dim() != 2:
            raise ValueError(f"encoder must return [B, d], got {tuple(h.shape)}")
        hs.append(h)
        ys.append(y.detach().cpu().reshape(-1))
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    if not hs:
        raise ValueError("batches yielded no data")
    return torch.cat(hs, dim=0), torch.cat(ys, dim=0)


@torch.no_grad()
def collect_sequence_features(
    encoder: Encoder,
    batches: Iterable[torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    max_batches: int | None = 50,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Stack per-sequence trajectories ``[N, T, d]`` (encoder must return ``[B, T, d]``)."""
    dev = torch.device(device) if device is not None else None
    seqs: list[torch.Tensor] = []
    n_batches = 0
    for item in batches:
        x = _batch_x(item)
        if dev is not None:
            x = x.to(dev)
        h = encoder(x).detach().float().cpu()
        if h.dim() != 3:
            raise ValueError(f"sequence encoder must return [B, T, d], got {tuple(h.shape)}")
        seqs.append(h)
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    if not seqs:
        raise ValueError("batches yielded no data")
    return torch.cat(seqs, dim=0)


@torch.no_grad()
def collect_augmentation_deltas(
    encoder: Encoder,
    batches: Iterable[torch.Tensor | tuple[torch.Tensor, ...]],
    augmentations: Sequence[AugFn],
    *,
    max_batches: int | None = 50,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Per-mode mean feature deltas ``[K, d]`` for D3 (Lemma D3).

    Each ``aug`` maps ``x -> x'``; delta_k = mean(encoder(aug_k(x)) - encoder(x)).
    """
    if not augmentations:
        raise ValueError("pass at least one augmentation function")
    dev = torch.device(device) if device is not None else None
    sums = None
    counts = 0
    n_batches = 0
    k = len(augmentations)
    for item in batches:
        x = _batch_x(item)
        if dev is not None:
            x = x.to(dev)
        h0 = encoder(x).detach().float()
        if h0.dim() != 2:
            raise ValueError("encoder must return [B, d]")
        if sums is None:
            sums = torch.zeros(k, h0.shape[1], device=h0.device, dtype=h0.dtype)
        for i, aug in enumerate(augmentations):
            xa = aug(x)
            hi = encoder(xa).detach().float()
            sums[i] += (hi - h0).sum(dim=0)
        counts += h0.shape[0]
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    if sums is None or counts == 0:
        raise ValueError("batches yielded no data")
    return (sums / counts).cpu()


def paired_batches(
    loader_a: Iterable,
    loader_b: Iterable,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Zip two loaders into ``(x_source, x_target)`` batches (shorter length)."""
    for a, b in zip(loader_a, loader_b):
        xa = a[0] if isinstance(a, (tuple, list)) else a
        xb = b[0] if isinstance(b, (tuple, list)) else b
        yield xa, xb
