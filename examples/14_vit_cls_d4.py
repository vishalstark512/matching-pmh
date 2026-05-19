#!/usr/bin/env python3
"""Walkthrough 12: ViT-style patch encoder + CLS embedding + D4 domain shift.

Paper block T2 (ViT-B/16): patch embedding is the input projection; PMH on CLS/pooled tokens.
No timm dependency — minimal patch encoder for integration template.

  python examples/14_vit_cls_d4.py
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, collect_features, estimate_from_config


class PatchViTEncoder(nn.Module):
    """Patch embed + CLS token + mean pool → h in R^d (ViT hook pattern)."""

    def __init__(self, img: int = 128, patch: int = 16, d: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 2, batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_dim = d

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # [B, d, H', W'] -> [B, T, d]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        b = tokens.shape[0]
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = self.blocks(seq)
        return seq[:, 0, :]  # CLS token as h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


def _batches(n: int, batch: int, brightness: float, device: torch.device):
    for _ in range(n):
        x = torch.rand(batch, 3, 128, 128, device=device)
        if brightness:
            x = (x + brightness).clamp(0.0, 1.0)
        yield x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    encoder = PatchViTEncoder().to(device)
    head = nn.Linear(encoder.out_dim, 10).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=3e-4)

    encoder.eval()
    h_src = collect_features(
        lambda x: encoder.encode(x), _batches(10, 8, 0.0, device), max_batches=10, device=device
    )
    h_tgt = collect_features(
        lambda x: encoder.encode(x), _batches(10, 8, 0.25, device), max_batches=10, device=device
    )
    artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=args.rank), h_src, h_tgt)
    print(f"[estimate] preflight={artifact.preflight} eigengap={artifact.eigengap:.4f}")

    pmh = PMHLoss(artifact, PMHConfig(weight=0.25, cap_ratio=0.3, warmup_epochs=1))
    encoder.train()
    for epoch in range(1, args.epochs + 1):
        pmh.set_epoch(epoch)
        x = torch.rand(16, 3, 128, 128, device=device)
        y = torch.randint(0, 10, (16,), device=device)
        opt.zero_grad()
        h = encoder.encode(x)
        task = F.cross_entropy(head(h), y)
        total, raw = pmh.capped_total(task, h)
        total.backward()
        opt.step()
        if epoch in (1, args.epochs):
            print(f"epoch {epoch}  task={task.item():.4f}  pmh={raw.item():.4f}")

    print("Hook: encoder.encode(x) -> CLS token. Swap PatchViTEncoder for timm ViT; keep D4 workflow.")


if __name__ == "__main__":
    main()
