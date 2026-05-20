#!/usr/bin/env python3
"""PyTorch Lightning + PMH (minimal recipe).

Run from repo root:
  pip install matching-pmh torch lightning
  python docs/cookbook/lightning_pmh.py
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L  # type: ignore[no-redef]

from pmh import PMHConfig, PMHTrainer
from pmh.integrations.torch import PMHCallback as LoopCallback
from pmh.pytorch_eval import pytorch_demo_loaders


class LitModel(L.LightningModule):
    def __init__(self, core: nn.Module, encoder: nn.Module, head: nn.Module, pmh_cb: LoopCallback) -> None:
        super().__init__()
        self.core = core
        self.encoder = encoder
        self.head = head
        self.pmh_cb = pmh_cb

    def training_step(self, batch, batch_idx):
        loss, step = self.pmh_cb.training_step(batch)
        self.log("train_loss", step.total_loss, prog_bar=True)
        self.log("pmh_ratio", step.pmh_task_ratio or 0.0, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.core.parameters(), lr=1e-3)

    def forward(self, x):
        return self.head(self.encoder(x))


def main() -> None:
    quick = os.environ.get("PMH_QUICK", "1").lower() in ("1", "true", "yes")
    bundle = pytorch_demo_loaders(n=120 if quick else 400, batch_size=32, seed=0)
    trainer_pmh = PMHTrainer(
        bundle.model,
        hook=bundle.encoder,
        head=bundle.head,
        nuisance="domain_shift",
        pmh_config=PMHConfig.golden_path(),
    )
    trainer_pmh.estimate(bundle.source_batches, bundle.target_batches, max_batches=5 if quick else 30)
    loop_cb = LoopCallback(trainer_pmh.pmh_loss_, trainer_pmh.encoder, head=trainer_pmh.head)

    lit = LitModel(bundle.model, bundle.encoder, bundle.head, loop_cb)
    lit_trainer = L.Trainer(max_epochs=1 if quick else 3, enable_checkpointing=False, logger=False)
    lit_trainer.fit(lit, bundle.train_loader)
    print("Done — check logs for pmh_ratio (target 5--30% of task).")


if __name__ == "__main__":
    main()
