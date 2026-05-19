#!/usr/bin/env python3
"""Minimal LightningModule using add_pmh_to_loss + PMHLightningCallback."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, estimate_from_config
from pmh.integrations.lightning import PMHLightningCallback, add_pmh_to_loss, _require_lightning


class SmallNet(nn.Module):
    def __init__(self, d_in: int = 20, d: int = 12, n_classes: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(d_in, d), nn.ReLU())
        self.head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def main() -> None:
    pl = _require_lightning()

    torch.manual_seed(0)
    x = torch.randn(200, 20)
    y = torch.randint(0, 3, (200,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

    net = SmallNet()
    artifact = estimate_from_config(
        SigmaTaskConfig.for_domain(rank=4),
        net.backbone(x[:100]).detach(),
        net.backbone(x[100:] + 0.5).detach(),
    )
    pmh_cfg = PMHConfig(weight=0.2, cap_ratio=0.3)

    class PMHLit(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.net = SmallNet()
            self.pmh_loss = PMHLoss(artifact, pmh_cfg)

        def training_step(self, batch, batch_idx: int = 0) -> torch.Tensor:
            xb, yb = batch
            logits = self.net(xb)
            task = F.cross_entropy(logits, yb)
            total, _ = add_pmh_to_loss(self.net, (xb,), task, self.pmh_loss, backbone_attr="backbone")
            return total

        def configure_optimizers(self):
            return torch.optim.Adam(self.net.parameters(), lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        callbacks=[PMHLightningCallback.from_artifact(artifact, pmh_cfg)],
    )
    trainer.fit(PMHLit(), loader)
    print("Lightning training with PMH completed.")


if __name__ == "__main__":
    main()
