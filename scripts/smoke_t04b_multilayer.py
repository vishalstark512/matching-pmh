#!/usr/bin/env python3
"""Smoke T4B multilayer vision demo (PMH_QUICK=1 default)."""

from __future__ import annotations

import os
import sys

import torch

from pmh import PMHConfig, PMHTrainer
from pmh.developer import RobustFitResult, check_applicability, evaluate_robust_fit
from pmh.pytorch_eval import pytorch_multilayer_vision_demo_loaders

QUICK = os.environ.get("PMH_QUICK", "1").lower() in ("1", "true", "yes")
N = 120 if QUICK else 300
EPOCHS = 1 if QUICK else 3
MAX_STEPS = 4 if QUICK else None


def main() -> int:
    torch.manual_seed(0)
    bundle = pytorch_multilayer_vision_demo_loaders(n=N, batch_size=16, seed=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = bundle.model.to(device)
    layer_names = ("conv1", "conv2")
    trainer = PMHTrainer(
        m,
        hook=bundle.encoder,
        head=m.head,
        nuisance="domain_shift",
        rank=8,
        pmh_config=PMHConfig(weight=0.2, cap_ratio=0.3, warmup_epochs=0),
        train_mode="feature_diff",
        forward_features=m.forward_features,
        layer_names=layer_names,
        head_layer="conv2",
        device=device,
    )
    sigmas = trainer.estimate_multilayer(
        bundle.source_batches,
        bundle.target_batches,
        max_batches=4 if QUICK else 15,
    )
    print("sigmas", {k: tuple(v.shape) for k, v in sigmas.items()})
    stats = trainer.fit(
        bundle.train_loader,
        source_batches=bundle.source_batches,
        target_batches=bundle.target_batches,
        epochs=EPOCHS,
        max_steps_per_epoch=MAX_STEPS,
        reestimate=False,
    )
    print("train", stats)
    report = evaluate_robust_fit(
        m,
        bundle.train_loader,
        bundle.val_loader,
        source_batches=bundle.source_batches,
        target_batches=bundle.target_batches,
        hook=bundle.encoder,
        head=m.head,
        epochs=1,
        pmh_result=RobustFitResult(
            trainer=trainer,
            stats=stats,
            applicability=check_applicability(has_target_domain=True),
            hook_used=bundle.encoder,
            preflight=trainer.artifact_.preflight if trainer.artifact_ else None,
        ),
        max_steps_per_epoch=MAX_STEPS,
    )
    print(report.deploy_summary())
    print(report.ship_verdict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
