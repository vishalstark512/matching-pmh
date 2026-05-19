#!/usr/bin/env python3
"""Multilayer feature-diff PMH on a tiny ConvNet (synthetic images)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pmh import PMHConfig, SigmaTaskConfig, estimate_from_config
from pmh.vision import MultiLayerPMHLoss, gram_sample_noise


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        l1 = F.relu(self.conv1(x))
        l2 = F.relu(self.conv2(l1))
        return {"layer1": l1, "layer2": l2, "pool": self.pool(l2).flatten(1)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x)["pool"])


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pmh_layers = ("layer1", "layer2")
    pmh_mod = MultiLayerPMHLoss(pmh_layers, PMHConfig(weight=1.0, cap_ratio=0.3))

    # Estimate per-layer Gram from source vs shifted target batches
    x_s = torch.randn(32, 3, 32, 32, device=device)
    x_t = x_s + 0.3 * torch.randn_like(x_s)
    with torch.no_grad():
        fs = model.forward_features(x_s)
        ft = model.forward_features(x_t)
    grams = {}
    for name in pmh_layers:
        d = fs[name].shape[1]
        flat_s = fs[name].permute(0, 2, 3, 1).reshape(-1, d).float()
        flat_t = ft[name].permute(0, 2, 3, 1).reshape(-1, d).float()
        diff = flat_s - flat_t
        grams[name] = (diff.T @ diff) / max(len(diff), 1)

    noise_sigma = 0.25
    for epoch in range(1, 21):
        pmh_mod.set_epoch(epoch)
        x = torch.randn(16, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (16,), device=device)
        opt.zero_grad()
        feats_clean = model.forward_features(x)
        noise_fns = {
            k: lambda h, g=grams[k]: gram_sample_noise(h, g, noise_sigma, rank=8)
            for k in pmh_layers
        }
        feats_noisy = {
            k: feats_clean[k] + noise_fns[k](feats_clean[k]) for k in pmh_layers
        }
        logits = model.fc(feats_clean["pool"])
        task = F.cross_entropy(logits, y)
        total, pmh_term = pmh_mod.capped_total(task, feats_clean, feats_noisy)
        total.backward()
        opt.step()
        if epoch % 5 == 0:
            print(f"epoch {epoch:2d}  task={task.item():.4f}  pmh={pmh_term.item():.4f}")


if __name__ == "__main__":
    main()
