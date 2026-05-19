#!/usr/bin/env python3
"""Compare B0 vs matched vs wrong-W vs isotropic on YOUR model (template).

Replace Backbone / dataloaders with yours. Synthetic demo data only shows the API.

  python examples/20_compare_training_arms.py
  python examples/20_compare_training_arms.py --epochs 12 --out results/my_run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, SigmaTaskConfig, collect_features, estimate_from_config
from pmh.benchmark import run_benchmark_protocol, write_benchmark_report

_QUICK = os.environ.get("PMH_QUICK", "").lower() in ("1", "true", "yes")


class Backbone(nn.Module):
    def __init__(self, d_in: int = 32, d: int = 16, n_classes: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d))
        self.head = nn.Linear(d, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def _make_tensors(n: int, shift: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 32, generator=g)
    if shift:
        x = x + shift
    y = torch.randint(0, 2, (n,), generator=g)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3 if _QUICK else 12)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/compare_arms"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    # --- Phase A ---
    backbone = Backbone().to(device)
    backbone.eval()
    n_collect = 8 if _QUICK else 25
    h_src = collect_features(
        lambda x: backbone.encode(x),
        [_make_tensors(64, 0.0, i)[0] for i in range(n_collect)],
        max_batches=n_collect,
        device=device,
    )
    h_tgt = collect_features(
        lambda x: backbone.encode(x),
        [_make_tensors(64, 0.6, i + 100)[0] for i in range(n_collect)],
        max_batches=n_collect,
        device=device,
    )
    artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=args.rank), h_src, h_tgt)
    print(f"[estimate] method={artifact.method} preflight={artifact.preflight} eigengap={artifact.eigengap}")

    # --- Phase B: standard arms ---
    n_train, n_val = (256, 128) if _QUICK else (512, 256)
    x_tr, y_tr = _make_tensors(n_train, 0.0, 1)
    x_va, y_va = _make_tensors(n_val, 0.6, 2)  # val on target-style shift
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=64)

    def model_factory() -> Backbone:
        return Backbone().to(device)

    def setup(m: Backbone) -> tuple:
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        return m.encode, m.head, opt

    pmh_cfg = PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=1)
    result = run_benchmark_protocol(
        artifact,
        model_factory,
        setup,
        train_loader,
        val_loader,
        epochs=args.epochs,
        pmh_config=pmh_cfg,
        wrong_rank=args.rank,
        device=device,
        max_steps_per_epoch=20 if _QUICK else None,
        shared_init=True,
    )

    paths = write_benchmark_report(result, args.out)
    print(f"\nWrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")
    print("\n--- Summary (target-shift val accuracy) ---")
    for arm in ("b0", "matched", "wrong_w", "isotropic"):
        r = result.arms.get(arm)
        if r:
            print(f"  {arm:10s}  {r.val_metric:.4f}")


if __name__ == "__main__":
    main()
