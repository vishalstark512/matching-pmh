"""PMHTrainer feature_diff / estimate_multilayer."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, PMHTrainer


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.enc(x))


def _loader(n=64, shift=0.0):
    x = torch.randn(n, 8) + shift
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def _forward_features(m: Tiny):
    def _fn(x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = m.enc[0](x)
        h = m.enc(x)
        return {"layer0": z, "layer2": h}

    return _fn


def test_estimate_multilayer_and_feature_diff_fit():
    torch.manual_seed(0)
    m = Tiny()
    ff = _forward_features(m)
    tr = PMHTrainer(
        m,
        hook=m.enc,
        head=m.head,
        nuisance="domain_shift",
        rank=3,
        pmh_config=PMHConfig(weight=0.2, cap_ratio=0.3, warmup_epochs=0),
        train_mode="feature_diff",
        forward_features=ff,
        layer_names=("layer0", "layer2"),
        head_layer="layer2",
    )
    sigmas = tr.estimate_multilayer(
        _loader(shift=0.0),
        _loader(shift=0.5),
        max_batches=2,
        save=False,
    )
    assert set(sigmas) == {"layer0", "layer2"}
    assert tr.artifact_.metadata.get("multilayer") is True
    stats = tr.fit(
        _loader(shift=0.1),
        source_batches=_loader(shift=0.0),
        target_batches=_loader(shift=0.5),
        epochs=1,
        max_steps_per_epoch=2,
        reestimate=False,
    )
    assert stats["n_steps"] == 2.0
    assert tr.layer_sigmas_ is not None
