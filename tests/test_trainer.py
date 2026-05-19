"""PMHTrainer and hooks."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, PMHTrainer
from pmh.hooks import resolve_hook, validate_representation
from pmh.suggest import suggest_nuisance


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(8, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(torch.relu(self.enc(x)))


def _loader(n=64, shift=0.0):
    x = torch.randn(n, 8) + shift
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def test_hooks_validate_4d():
    h = torch.randn(2, 8, 3, 3)
    out = validate_representation(h)
    assert out.shape == (2, 8)


def test_resolve_hook_module():
    m = Tiny()
    enc = resolve_hook(m, m.enc)
    x = torch.randn(4, 8)
    assert enc(x).shape == (4, 4)


def test_pmh_trainer_d4(tmp_path):
    torch.manual_seed(0)
    m = Tiny()
    tr = PMHTrainer(
        m,
        hook="enc",
        head=m.head,
        nuisance="domain_shift",
        rank=3,
        pmh_config=PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=0),
        artifact_path=tmp_path / "art.pt",
    )
    stats = tr.fit(
        _loader(shift=0.1),
        source_batches=_loader(shift=0.0),
        target_batches=_loader(shift=0.5),
        epochs=2,
        max_steps_per_epoch=3,
    )
    assert tr.artifact_ is not None
    assert stats["n_steps"] == 3.0


def test_suggest_domain_shift():
    s = suggest_nuisance(has_target_domain=True, has_target_labels=False)
    assert s.method == "D4"
