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


class _TinyVisionCNN(nn.Module):
    """Small RGB CNN for Type-2 (D2 isotropic) notebook demos — not ImageNet-scale."""

    def __init__(self, n_classes: int = 5, channels: int = 16) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        d = channels
        self.head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


class _TinyVisionMultilayerCNN(nn.Module):
    """Two-stage RGB CNN for T4B feature-diff demos (per-layer Gram)."""

    def __init__(self, n_classes: int = 5, channels: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels * 2, n_classes)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        a = torch.relu(self.conv1(x))
        b = torch.relu(self.conv2(a))
        return {"conv1": self.pool(a).flatten(1), "conv2": self.pool(b).flatten(1)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)["conv2"])


def _vision_tensors(
    n: int,
    n_classes: int,
    *,
    seed: int,
    noise_sigma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    if noise_sigma > 0:
        x = x + noise_sigma * torch.randn_like(x, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return x, y


def _loader_images(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def pytorch_isotropic_demo_loaders(
    *,
    n: int = 400,
    batch_size: int = 32,
    seed: int = 0,
    n_classes: int = 5,
    eval_noise_sigma: float = 0.10,
) -> PyTorchEvalBundle:
    """Synthetic RGB mini-images for D2 isotropic PMH (T2A / T2B notebook demos).

    Train/estimate on clean images; ``val_loader`` applies Gaussian input noise at
    ``eval_noise_sigma`` (paper T2A σ=0.10, T2B σ=0.08).
    """
    x_train, y_train = _vision_tensors(n, n_classes, seed=seed + 1)
    x_src, y_src = _vision_tensors(n, n_classes, seed=seed + 2)
    x_val_clean, y_val = _vision_tensors(n // 2, n_classes, seed=seed + 3)
    g = torch.Generator().manual_seed(seed + 4)
    x_val = x_val_clean + eval_noise_sigma * torch.randn(x_val_clean.shape, generator=g)
    # noisy val uses same labels as clean val (paired by seed+3 labels)
    model = _TinyVisionCNN(n_classes=n_classes)
    return PyTorchEvalBundle(
        model=model,
        encoder=model.enc,
        head=model.head,
        train_loader=_loader_images(x_train, y_train, batch_size=batch_size, shuffle=True),
        source_batches=_loader_images(x_src, y_src, batch_size=batch_size, shuffle=True),
        target_batches=_loader_images(x_src, y_src, batch_size=batch_size, shuffle=True),
        val_loader=_loader_images(x_val, y_val, batch_size=batch_size, shuffle=False),
        d_in=16,
        n_classes=n_classes,
    )


class _SeqModel(nn.Module):
    """GRU sequence encoder returning [B, T, d] for D6 demos."""

    def __init__(self, d_in: int = 8, d_hidden: int = 24, n_classes: int = 3, t_len: int = 12) -> None:
        super().__init__()
        self.t_len = t_len
        self.inp = nn.Linear(d_in, d_hidden)
        self.rnn = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.inp(x))
        return self.head(h[:, -1, :])


class _SequenceHook(nn.Module):
    """Hook module for D6 — keeps [B, T, d] (use ``pool_spatial=False`` on trainer)."""

    def __init__(self, core: _SeqModel) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.core.rnn(self.core.inp(x))
        return h


class _SeqHead(nn.Module):
    """Classifier on last timestep when hook returns [B, T, d]."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 3:
            h = h[:, -1, :]
        return self.linear(h)


@dataclass
class PyTorchSequenceBundle:
    model: nn.Module
    encoder: nn.Module
    head: nn.Module
    train_loader: DataLoader
    sequence_batches: DataLoader
    val_loader: DataLoader


def _seq_tensors(n: int, t_len: int, d_in: int, n_classes: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, t_len, d_in, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return x, y


def pytorch_sequence_demo_loaders(
    *,
    n: int = 400,
    batch_size: int = 16,
    seed: int = 0,
    t_len: int = 12,
    d_in: int = 8,
    n_classes: int = 3,
) -> PyTorchSequenceBundle:
    """Synthetic [B, T, d_in] sequences for temporal (D6) notebooks."""
    x_tr, y_tr = _seq_tensors(n, t_len, d_in, n_classes, seed + 1)
    x_seq, y_seq = _seq_tensors(n, t_len, d_in, n_classes, seed + 2)
    x_va, y_va = _seq_tensors(n // 2, t_len, d_in, n_classes, seed + 3)
    model = _SeqModel(d_in, 24, n_classes, t_len)
    hook = _SequenceHook(model)

    def _ld(x: torch.Tensor, y: torch.Tensor, shuffle: bool) -> DataLoader:
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    return PyTorchSequenceBundle(
        model=model,
        encoder=hook,
        head=_SeqHead(model.head),
        train_loader=_ld(x_tr, y_tr, True),
        sequence_batches=_ld(x_seq, y_seq, True),
        val_loader=_ld(x_va, y_va, False),
    )


def _vision_domain_shift_tensors(
    n: int,
    n_classes: int,
    *,
    seed: int,
    target_brightness: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB 32×32 with optional brightness shift on deploy domain."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g) * 0.5
    if target_brightness != 0.0:
        x = x + target_brightness
    y = torch.randint(0, n_classes, (n,), generator=g)
    return x, y


def pytorch_multilayer_vision_demo_loaders(
    *,
    n: int = 400,
    batch_size: int = 32,
    seed: int = 0,
    n_classes: int = 5,
    target_brightness: float = 0.35,
) -> PyTorchEvalBundle:
    """Synthetic RGB domain shift for T4B ``feature_diff`` (conv1 + conv2 layers)."""
    x_train, y_train = _vision_domain_shift_tensors(n, n_classes, seed=seed + 1)
    x_src, y_src = _vision_domain_shift_tensors(n, n_classes, seed=seed + 2)
    x_tgt, y_tgt = _vision_domain_shift_tensors(
        n, n_classes, seed=seed + 3, target_brightness=target_brightness
    )
    x_val, y_val = _vision_domain_shift_tensors(
        n // 2, n_classes, seed=seed + 4, target_brightness=target_brightness
    )
    model = _TinyVisionMultilayerCNN(n_classes=n_classes)
    enc = nn.Sequential(model.conv1, nn.ReLU(), model.conv2, model.pool, nn.Flatten())

    return PyTorchEvalBundle(
        model=model,
        encoder=enc,
        head=model.head,
        train_loader=_loader_images(x_train, y_train, batch_size=batch_size, shuffle=True),
        source_batches=_loader_images(x_src, y_src, batch_size=batch_size, shuffle=True),
        target_batches=_loader_images(x_tgt, y_tgt, batch_size=batch_size, shuffle=True),
        val_loader=_loader_images(x_val, y_val, batch_size=batch_size, shuffle=False),
        d_in=32,
        n_classes=n_classes,
    )


def pytorch_demo_loaders(
    *,
    n: int = 400,
    batch_size: int = 32,
    seed: int = 0,
) -> PyTorchEvalBundle:
    """Synthetic domain-shift loaders (same spirit as ``scripts/demos/first_run_domain_shift``)."""
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
