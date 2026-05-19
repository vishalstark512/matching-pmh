#!/usr/bin/env python3
"""Walkthrough 16: Finite augmentation modes + D3 (photometric / T2 aug template).

  python examples/18_augmentation_d3.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, estimate_from_config


class SmallEncoder(nn.Module):
    def __init__(self, d: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, d), nn.ReLU(), nn.Linear(d, d))
        self.out_dim = d

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _aug_modes(x: torch.Tensor) -> list[torch.Tensor]:
    return [x, (x * 1.2).clamp(-3, 3), x + 0.15 * torch.randn_like(x)]


def main() -> None:
    torch.manual_seed(5)
    encoder = SmallEncoder()
    x_ref = torch.randn(64, 32)

    encoder.eval()
    with torch.no_grad():
        h0 = encoder.encode(x_ref)
        deltas = []
        for x_aug in _aug_modes(x_ref)[1:]:
            deltas.append((encoder.encode(x_aug) - h0).mean(0))
        aug_stack = torch.stack(deltas, dim=0)

    artifact = estimate_from_config(
        SigmaTaskConfig.for_augmentation(),
        aug_deltas=aug_stack,
    )
    print(f"[estimate] method={artifact.method} sigma shape={tuple(artifact.sigma.shape)}")

    head = nn.Linear(encoder.out_dim, 5)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
    pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=0))

    encoder.train()
    for epoch in range(1, 16):
        pmh.set_epoch(epoch)
        x = torch.randn(32, 32)
        y = torch.randint(0, 5, (32,))
        opt.zero_grad()
        h = encoder.encode(x)
        task = F.cross_entropy(head(h), y)
        total, raw = pmh.capped_total(task, h)
        total.backward()
        opt.step()
        if epoch in (1, 15):
            print(f"epoch {epoch:2d}  task={task.item():.4f}  pmh={raw.item():.4f}")

    print("Vision: stack per-mode mean deltas [K, d] or [K, N, d] from your aug pipeline.")


if __name__ == "__main__":
    main()
