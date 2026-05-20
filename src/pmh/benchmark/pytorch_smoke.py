"""Small PyTorch falsification benchmark for CI / ``pmh-train validate``."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, SigmaTaskConfig, collect_features, estimate_from_config
from pmh.benchmark.protocol import BenchmarkResult, run_benchmark_protocol

_QUICK = os.environ.get("PMH_QUICK", "").lower() in ("1", "true", "yes")


class _SmokeNet(nn.Module):
    def __init__(self, d_in: int = 24, d: int = 12, n_classes: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, d), nn.ReLU())
        self.head = nn.Linear(d, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def _tensors(n: int, shift: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 24, generator=g)
    if shift:
        x = x + shift
    y = torch.randint(0, 2, (n,), generator=g)
    return x, y


def run_pytorch_benchmark_smoke(
    *,
    rank: int = 8,
    epochs: int | None = None,
    device: str | torch.device = "cpu",
) -> BenchmarkResult:
    """Train B0/matched/wrong-W/isotropic on synthetic domain shift."""
    dev = torch.device(device)
    ep = epochs if epochs is not None else (2 if _QUICK else 8)
    n_collect = 6 if _QUICK else 20

    net = _SmokeNet().to(dev)
    net.eval()
    batches_s = [_tensors(32, 0.0, i)[0] for i in range(n_collect)]
    batches_t = [_tensors(32, 0.5, i + 50)[0] for i in range(n_collect)]
    h_src = collect_features(lambda x: net.encode(x), batches_s, max_batches=n_collect, device=dev)
    h_tgt = collect_features(lambda x: net.encode(x), batches_t, max_batches=n_collect, device=dev)
    artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=rank), h_src, h_tgt)

    n_tr, n_va = (128, 64) if _QUICK else (400, 200)
    x_tr, y_tr = _tensors(n_tr, 0.0, 1)
    x_va, y_va = _tensors(n_va, 0.6, 2)
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=64)

    def model_factory() -> _SmokeNet:
        return _SmokeNet().to(dev)

    def setup(m: _SmokeNet) -> tuple:
        return m.encode, m.head, torch.optim.Adam(m.parameters(), lr=1e-3)

    return run_benchmark_protocol(
        artifact,
        model_factory,
        setup,
        train_loader,
        val_loader,
        epochs=ep,
        pmh_config=PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=1),
        wrong_rank=rank,
        device=dev,
        max_steps_per_epoch=15 if _QUICK else None,
        shared_init=True,
    )
