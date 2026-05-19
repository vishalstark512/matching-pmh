#!/usr/bin/env python3
"""Walkthrough 2: ResNet-18 penultimate features + D4 + PMHLoss (no dataset download).

Synthetic domain shift: target batches get a fixed brightness offset on pixels.
See docs/walkthroughs/02-resnet-vision-d4.md.

  pip install "matching-pmh[vision]"
  python examples/12_resnet_hook_d4.py
  python examples/12_resnet_hook_d4.py --epochs 5 --rank 16
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, collect_features, estimate_from_config


class ConvFallbackEncoder(nn.Module):
    """Tiny CNN if torchvision is unavailable (CPU-friendly walkthrough)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = 16

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class ResNetEncoder(nn.Module):
    """ResNet-18 with penultimate embedding h in R^512."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.out_dim = 512

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


def build_encoder(pretrained: bool = False) -> nn.Module:
    try:
        return ResNetEncoder(pretrained=pretrained)
    except Exception as exc:  # noqa: BLE001 — import or torch/torchvision ABI issues
        print(f"torchvision ResNet unavailable ({exc}); using ConvFallbackEncoder.")
        return ConvFallbackEncoder()


def _batch_iter(n_batches: int, batch: int, shift: float, device: torch.device):
    for _ in range(n_batches):
        x = torch.rand(batch, 3, 224, 224, device=device)
        if shift:
            x = (x + shift).clamp(0.0, 1.0)
        yield x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    encoder = build_encoder(pretrained=args.pretrained).to(device)
    head = nn.Linear(encoder.out_dim, 10).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-4)

    # --- Phase A: D4 on source vs shifted target features ---
    encoder.eval()
    src_batches = list(_batch_iter(12, args.batch, 0.0, device))
    tgt_batches = list(_batch_iter(12, args.batch, 0.35, device))
    h_src = collect_features(lambda x: encoder.encode(x), src_batches, max_batches=12, device=device)
    h_tgt = collect_features(lambda x: encoder.encode(x), tgt_batches, max_batches=12, device=device)

    artifact = estimate_from_config(
        SigmaTaskConfig.for_domain(rank=args.rank),
        h_src,
        h_tgt,
    )
    print(f"[estimate] method={artifact.method} dim={artifact.dim} preflight={artifact.preflight} eigengap={artifact.eigengap:.4f}")

    # --- Phase B: train with matched PMH on h ---
    pmh = PMHLoss(artifact, PMHConfig(weight=0.25, cap_ratio=0.3, warmup_epochs=1))
    encoder.train()
    for epoch in range(1, args.epochs + 1):
        pmh.set_epoch(epoch)
        x = torch.rand(args.batch, 3, 224, 224, device=device)
        y = torch.randint(0, 10, (args.batch,), device=device)
        opt.zero_grad()
        h = encoder.encode(x)
        task = F.cross_entropy(head(h), y)
        total, raw = pmh.capped_total(task, h)
        total.backward()
        opt.step()
        if epoch == 1 or epoch == args.epochs:
            print(f"epoch {epoch:2d}  task={task.item():.4f}  pmh={raw.item():.4f}")

    print("Done. Replace synthetic batches with your source/target DataLoaders; keep encode() as the hook.")


if __name__ == "__main__":
    main()
