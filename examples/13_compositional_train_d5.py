#!/usr/bin/env python3
"""Walkthrough 5b: D5 compositional Sigma + training loop on a toy encoder.

First 5 of 20 dims are nuisance coordinates; PMH penalizes sensitivity on that block.
See docs/walkthroughs/05-compositional-d5.md.

  python examples/13_compositional_train_d5.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, estimate_from_config


class ToyEncoder(nn.Module):
    def __init__(self, d_in: int = 20, d: int = 20) -> None:
        super().__init__()
        self.net = nn.Linear(d_in, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    torch.manual_seed(1)
    d, k = 20, 5
    nuisance_idx = list(range(k))

    # Phase A: features where nuisance coords carry structured variance
    x = torch.randn(256, d)
    x[:, :k] += 0.8 * torch.randn(256, k)
    artifact = estimate_from_config(
        SigmaTaskConfig.for_compositional(nuisance_idx),
        x,
    )
    print(f"[estimate] preflight={artifact.preflight}  block_norm={artifact.sigma[:k, :k].norm():.4f}")

    encoder = ToyEncoder()
    head = nn.Linear(d, 3)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-2)
    pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=0))

    for epoch in range(1, 21):
        pmh.set_epoch(epoch)
        xb = torch.randn(32, d)
        yb = torch.randint(0, 3, (32,))
        opt.zero_grad()
        h = encoder(xb)
        task = F.cross_entropy(head(h), yb)
        total, raw = pmh.capped_total(task, h)
        total.backward()
        opt.step()
        if epoch in (1, 10, 20):
            print(f"epoch {epoch:2d}  task={task.item():.4f}  pmh={raw.item():.4f}")


if __name__ == "__main__":
    main()
