#!/usr/bin/env python3
"""First run: baseline ERM vs PMH on a synthetic domain shift (readable metrics).

No paper background required. Prints target-domain accuracy before/after PMH.

  python scripts/demos/first_run_domain_shift.py
  PMH_QUICK=1 python scripts/demos/first_run_domain_shift.py
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, PMHTrainer
from pmh.adoption import RECIPE_ONE_LINER, STEP5_PYTORCH_HINT
from pmh.onboarding import preflight_plain_english

_QUICK = os.environ.get("PMH_QUICK", "").lower() in ("1", "true", "yes")


class Backbone(nn.Module):
    def __init__(self, d_in: int = 32, d: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _loader(n: int, batch: int, shift: float, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 32, generator=g) + shift
    y = torch.randint(0, 2, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True)


@torch.no_grad()
def target_accuracy(
    model: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = head(encoder(xb))
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(total, 1)


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    max_steps: int | None,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    step = 0
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                return


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = 200 if _QUICK else 800
    batch = 32
    epochs = 3 if _QUICK else 12
    max_steps = 15 if _QUICK else None

    src_loader = _loader(n, batch, shift=0.0, seed=1)
    tgt_loader = _loader(n, batch, shift=0.8, seed=2)
    train_loader = _loader(n, batch, shift=0.2, seed=3)
    tgt_eval = _loader(n // 2, batch, shift=0.8, seed=4)

    backbone = Backbone().to(device)
    head = nn.Linear(16, 2).to(device)

    # --- Baseline: standard training (no PMH) ---
    baseline = nn.Sequential(backbone, head).to(device)
    train_baseline(baseline, train_loader, epochs=epochs, lr=1e-2, device=device, max_steps=max_steps)
    acc_b0 = target_accuracy(baseline, backbone, head, tgt_eval, device)

    # --- PMH: same data, domain-shift regularizer ---
    backbone_pmh = Backbone().to(device)
    head_pmh = nn.Linear(16, 2).to(device)
    model_pmh = nn.Sequential(backbone_pmh, head_pmh).to(device)

    trainer = PMHTrainer(
        model_pmh,
        hook=backbone_pmh,
        head=head_pmh,
        nuisance="domain_shift",
        rank=6,
        pmh_config=PMHConfig.balanced(),
        artifact_path="artifacts/first_run_demo.pt",
        device=device,
    )
    trainer.fit(
        train_loader,
        source_batches=src_loader,
        target_batches=tgt_loader,
        epochs=epochs,
        max_steps_per_epoch=max_steps,
    )
    acc_pmh = target_accuracy(model_pmh, backbone_pmh, head_pmh, tgt_eval, device)

    print()
    print("matching-pmh first run (synthetic domain shift)")
    print(RECIPE_ONE_LINER)
    print("-" * 48)
    print(f"Target accuracy (baseline ERM):  {acc_b0:.3f}")
    print(f"Target accuracy (with PMH):      {acc_pmh:.3f}")
    pf = trainer.artifact_.preflight
    print(f"Preflight: {pf} — {preflight_plain_english(pf)}")
    print()
    print(STEP5_PYTORCH_HINT)
    print()
    print("Next: notebooks/tasks/t04a-vision-domain.ipynb  |  pmh-train route  |  pmh-train evaluate --demo")


if __name__ == "__main__":
    main()
