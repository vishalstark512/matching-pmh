"""Domain shift with D4 — minimal PMHTrainer (Phase A + B)."""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, PMHTrainer

_QUICK = os.environ.get("PMH_QUICK", "").lower() in ("1", "true", "yes")


class Backbone(nn.Module):
    def __init__(self, d_in: int = 32, d: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _loader(n: int, batch: int, shift: float) -> DataLoader:
    x = torch.randn(n, 32) + shift
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True)


def main() -> None:
    torch.manual_seed(0)
    backbone = Backbone()
    head = nn.Linear(16, 2)
    model = nn.Sequential(backbone, head)

    n = 200 if _QUICK else 800
    batch = 32
    src_loader = _loader(n, batch, 0.0)
    tgt_loader = _loader(n, batch, 0.8)
    train_loader = _loader(n, batch, 0.2)

    trainer = PMHTrainer(
        model,
        hook=backbone,
        head=head,
        nuisance="domain_shift",
        rank=6,
        pmh_config=PMHConfig.balanced(),
        artifact_path="artifacts/demo_d4.pt",
    )

    stats = trainer.fit(
        train_loader,
        source_batches=src_loader,
        target_batches=tgt_loader,
        epochs=3 if _QUICK else 12,
        max_steps_per_epoch=10 if _QUICK else None,
    )
    print(f"done  task={stats['task_loss']:.4f}  pmh={stats['pmh_loss']:.4f}")
    print(f"preflight={trainer.artifact_.preflight}  method={trainer.artifact_.method}")


if __name__ == "__main__":
    main()
