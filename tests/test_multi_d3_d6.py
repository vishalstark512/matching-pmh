"""MultiPMHLoss, D3 collection, D6 matcher."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, PMHMatcher, PMHTrainer
from pmh.multi import MultiPMHLoss
from pmh.features import collect_augmentation_deltas
from pmh.config import SigmaTaskConfig
from pmh.estimate import estimate_from_config


class Enc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 4)

    def forward(self, x):
        return torch.relu(self.lin(x))


def test_multi_pmh_loss():
    e1 = estimate_from_config(SigmaTaskConfig.for_isotropic(4, 0.1))
    e2 = estimate_from_config(SigmaTaskConfig.for_isotropic(4, 0.05))
    multi = MultiPMHLoss([e1, e2], PMHConfig(weight=0.2, cap_ratio=0.0, warmup_epochs=0))
    h = torch.randn(3, 4, requires_grad=True)
    task = (h**2).mean()
    total, raw = multi.capped_total(task, h)
    total.backward()
    assert h.grad is not None
    assert raw.ndim == 0


def test_collect_aug_deltas():
    enc = Enc()
    data = [(torch.randn(5, 6),) for _ in range(3)]

    def bright(x):
        return x + 0.1

    d = collect_augmentation_deltas(enc, data, [bright], max_batches=3)
    assert d.shape == (1, 4)


def test_matcher_d3_aug_deltas():
    rng = np.random.default_rng(0)
    deltas = rng.standard_normal((3, 10)).astype(np.float32)
    m = PMHMatcher(nuisance="augmentation").fit(np.zeros((1, 10)), aug_deltas=deltas)
    assert m.artifact_.method == "D3"


def test_matcher_d6_sequences():
    rng = np.random.default_rng(1)
    seq = rng.standard_normal((20, 5, 8)).astype(np.float32)
    m = PMHMatcher(nuisance="temporal", rank=4)
    m.fit(seq)
    assert m.artifact_.method == "D6"


def test_trainer_d3_augmentations():
    torch.manual_seed(0)
    model = Enc()
    head = nn.Linear(4, 2)
    full = nn.Sequential(model, head)

    def jitter(x):
        return x + 0.05 * torch.randn_like(x)

    loader = DataLoader(TensorDataset(torch.randn(32, 6), torch.randint(0, 2, (32,))), batch_size=8)
    tr = PMHTrainer(
        full,
        hook=model,
        head=head,
        nuisance="augmentation",
        pmh_config=PMHConfig(weight=0.2, cap_ratio=0.3, warmup_epochs=0),
    )
    tr.estimate(source_batches=loader, augmentations=[jitter], max_batches=4)
    assert tr.artifact_.method == "D3"
