"""Collect representation batches from PyTorch data pipelines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import torch

Encoder = Callable[[torch.Tensor], torch.Tensor]


@torch.no_grad()
def collect_features(
    encoder: Encoder,
    batches: Iterable[torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    max_batches: int | None = 50,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Stack encoder outputs into ``[N, d]`` for ``estimate_sigma_task``.

    Parameters
    ----------
    encoder : callable
        ``phi(x) -> [B, d]`` (e.g. backbone).
    batches : iterable
        Each item is either ``x`` or ``(x, y, ...)``; only ``x`` is used.
    max_batches : int, optional
        Stop after this many batches (None = use all).
    """
    dev = torch.device(device) if device is not None else None
    rows: list[torch.Tensor] = []
    n_batches = 0
    for item in batches:
        x = item[0] if isinstance(item, (tuple, list)) else item
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


def paired_batches(
    loader_a: Iterable,
    loader_b: Iterable,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Zip two loaders into ``(x_source, x_target)`` batches (shorter length)."""
    for a, b in zip(loader_a, loader_b):
        xa = a[0] if isinstance(a, (tuple, list)) else a
        xb = b[0] if isinstance(b, (tuple, list)) else b
        yield xa, xb
