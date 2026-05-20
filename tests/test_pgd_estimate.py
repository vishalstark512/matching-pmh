"""PGD subspace estimate (T7B library path)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh.calibrate.pgd import estimate_pgd_subspace_from_model


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(8, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.enc(x))


def test_estimate_pgd_subspace_from_model():
    torch.manual_seed(0)
    m = Tiny()
    x = torch.randn(32, 8)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    art = estimate_pgd_subspace_from_model(
        m,
        hook=m.enc,
        head=m.head,
        batches=loader,
        rank=3,
        epsilon=0.2,
        steps=2,
        max_batches=2,
    )
    assert art.method == "D7"
    assert art.sigma.shape == (4, 4)
    assert art.metadata.get("source") == "pgd"
