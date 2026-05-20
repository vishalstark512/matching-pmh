"""PyTorch Step 5 helpers: synthetic demo and .npy → loaders + small MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PyTorchEvalBundle:
    """Everything needed for ``evaluate_robust_fit`` on tabular / feature tensors."""

    model: nn.Module
    encoder: nn.Module
    head: nn.Module
    train_loader: DataLoader
    source_batches: DataLoader
    target_batches: DataLoader
    val_loader: DataLoader
    d_in: int
    n_classes: int


class _FeatureMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_classes: int) -> None:
        super().__init__()
        d_h = min(d_hidden, max(8, d_in))
        self.enc = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_h))
        self.head = nn.Linear(d_h, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


def _loader_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.int64)),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def pytorch_eval_bundle_from_arrays(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    *,
    val_fraction: float = 0.35,
    seed: int = 0,
    batch_size: int = 32,
    hidden: int = 64,
) -> PyTorchEvalBundle:
    """Build loaders + MLP from frozen features (site A train, site B deploy holdout)."""
    from sklearn.model_selection import train_test_split

    x_pool, x_val, y_pool, y_val = train_test_split(
        x_target,
        y_target,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_target,
    )
    n_classes = int(max(y_source.max(), y_target.max(), 0) + 1)
    d_in = int(x_source.shape[1])
    model = _FeatureMLP(d_in, hidden, n_classes)
    return PyTorchEvalBundle(
        model=model,
        encoder=model.enc,
        head=model.head,
        train_loader=_loader_xy(x_source, y_source, batch_size=batch_size, shuffle=True),
        source_batches=_loader_xy(x_source, y_source, batch_size=batch_size, shuffle=True),
        target_batches=_loader_xy(x_pool, y_pool, batch_size=batch_size, shuffle=True),
        val_loader=_loader_xy(x_val, y_val, batch_size=batch_size, shuffle=False),
        d_in=d_in,
        n_classes=n_classes,
    )


def pytorch_demo_loaders(
    *,
    n: int = 400,
    batch_size: int = 32,
    seed: int = 0,
) -> PyTorchEvalBundle:
    """Synthetic domain-shift loaders (same spirit as ``examples/00_first_run``)."""
    def _make(n_s: int, shift: float, s: int) -> tuple[np.ndarray, np.ndarray]:
        gen = torch.Generator().manual_seed(s)
        x = torch.randn(n_s, 32, generator=gen) + shift
        y = torch.randint(0, 2, (n_s,), generator=gen)
        return x.numpy().astype(np.float32), y.numpy().astype(np.int64)

    xs, ys = _make(n, 0.0, seed + 1)
    xt, yt = _make(n, 0.8, seed + 2)
    x_train, y_train = _make(n, 0.2, seed + 3)
    x_val, y_val = _make(n // 2, 0.8, seed + 4)

    d_in, n_classes = 32, 2
    model = _FeatureMLP(d_in, 16, n_classes)
    return PyTorchEvalBundle(
        model=model,
        encoder=model.enc,
        head=model.head,
        train_loader=_loader_xy(x_train, y_train, batch_size=batch_size, shuffle=True),
        source_batches=_loader_xy(xs, ys, batch_size=batch_size, shuffle=True),
        target_batches=_loader_xy(xt, yt, batch_size=batch_size, shuffle=True),
        val_loader=_loader_xy(x_val, y_val, batch_size=batch_size, shuffle=False),
        d_in=d_in,
        n_classes=n_classes,
    )


def pmh_config_from_preset(name: str | None) -> Any:
    """Resolve CLI preset name to :class:`PMHConfig`."""
    from pmh.config import PMHConfig

    if name is None or name == "balanced":
        return PMHConfig.balanced()
    if name == "conservative":
        return PMHConfig.conservative()
    if name == "aggressive":
        return PMHConfig.aggressive()
    if name == "finetune_llm":
        return PMHConfig.finetune_llm()
    raise ValueError(f"unknown pmh preset {name!r}; use conservative|balanced|aggressive|finetune_llm")
