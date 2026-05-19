#!/usr/bin/env python3
"""Minimal PMHTrainer (transformers) on a toy classification head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pmh import PMHConfig, SigmaTaskConfig, estimate_from_config


class ToyDataset(Dataset):
    def __init__(self, n: int = 64, d: int = 32, n_classes: int = 3) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(n, d)
        self.y = torch.randint(0, n_classes, (n,))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.x[i], "labels": self.y[i]}


class ToyModel(nn.Module):
    """Fake LM: input_ids are already [B, d] features."""

    def __init__(self, d: int = 32, n_classes: int = 3) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.lm_head = nn.Linear(d, n_classes)

    def forward(self, input_ids=None, labels=None, output_hidden_states=False, **kwargs):
        h = self.body(input_ids)
        logits = self.lm_head(h)
        out = type("Out", (), {"logits": logits, "hidden_states": (h,)})
        return out


def main() -> None:
    import os

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    from transformers import TrainingArguments

    from pmh.integrations.hf_trainer import get_pmh_trainer

    PMHTrainer = get_pmh_trainer()
    d = 32
    model = ToyModel(d)
    ds = ToyDataset(128, d)

    h0 = model.body(ds.x[:64])
    h1 = model.body(ds.x[64:] + 0.3)
    artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=4), h0.detach(), h1.detach())

    args = TrainingArguments(
        output_dir="./tmp_pmh_trainer",
        per_device_train_batch_size=16,
        num_train_epochs=2,
        logging_steps=5,
        report_to="none",
        save_strategy="no",
    )

    trainer = PMHTrainer.from_artifact(
        artifact,
        PMHConfig(weight=0.2, cap_ratio=0.3),
        model=model,
        args=args,
        train_dataset=ds,
        representation_fn=lambda m, inp: m.body(inp["input_ids"]),
    )
    trainer.train()
    print("PMHTrainer finished.")


if __name__ == "__main__":
    main()
