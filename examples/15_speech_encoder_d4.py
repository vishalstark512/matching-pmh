#!/usr/bin/env python3
"""Walkthrough 13: Speech-style mel encoder + D4 (Whisper T6A template).

Toy 2D mel CNN mimics a conv stem; domain shift = mic gain / noise floor.
Real Whisper: same hook on encoder hidden states, source vs target accents.

  python examples/15_speech_encoder_d4.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, collect_features, estimate_from_config


class MelEncoder(nn.Module):
    """[B, 1, n_mels, T] → pooled embedding h."""

    def __init__(self, n_mels: int = 40, d: int = 48) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, d)
        self.out_dim = d

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        z = self.conv(mel).flatten(1)
        return self.proj(z)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encode(mel)


def _mel_batches(n: int, batch: int, gain: float, device: torch.device):
    for _ in range(n):
        mel = torch.randn(batch, 1, 40, 64, device=device) * 0.1
        if gain:
            mel = mel + gain
        yield mel


def main() -> None:
    torch.manual_seed(2)
    device = torch.device("cpu")
    encoder = MelEncoder().to(device)
    head = nn.Linear(encoder.out_dim, 20)  # pseudo vocab / classes
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    encoder.eval()
    h_src = collect_features(encoder.encode, _mel_batches(12, 8, 0.0, device), max_batches=12)
    h_tgt = collect_features(encoder.encode, _mel_batches(12, 8, 0.6, device), max_batches=12)
    artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=12), h_src, h_tgt)
    print(f"[estimate] method={artifact.method} preflight={artifact.preflight}")

    pmh = PMHLoss(artifact, PMHConfig(weight=0.2, cap_ratio=0.3, warmup_epochs=0))
    encoder.train()
    for step in range(1, 41):
        pmh.set_epoch(step // 10)
        mel = torch.randn(8, 1, 40, 64, device=device)
        y = torch.randint(0, 20, (8,), device=device)
        opt.zero_grad()
        h = encoder.encode(mel)
        task = F.cross_entropy(head(h), y)
        total, raw = pmh.capped_total(task, h)
        total.backward()
        opt.step()
        if step in (1, 40):
            print(f"step {step:2d}  task={task.item():.4f}  pmh={raw.item():.4f}")

    print("Whisper-scale: use encoder hidden states; D4 on studio vs accented clips.")


if __name__ == "__main__":
    main()
