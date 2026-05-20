"""G3b — Hugging Face Trainer + PMH (you keep TrainingArguments / DPO / callbacks)."""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pmh import PMHConfig, SigmaTaskConfig, check_applicability, estimate_from_config
from pmh.integrations.hf_trainer import get_pmh_trainer


class ToyDS(Dataset):
    def __init__(self, n: int = 128, d: int = 32) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(n, d)
        self.y = torch.randint(0, 3, (n,))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.x[i], "labels": self.y[i]}


class ToyLM(nn.Module):
    def __init__(self, d: int = 32, n_classes: int = 3) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.lm_head = nn.Linear(d, n_classes)

    def forward(self, input_ids=None, labels=None, output_hidden_states=False, **kwargs):
        h = self.body(input_ids)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits, "hidden_states": (h,)})


def main() -> None:
    from transformers import TrainingArguments

    d = 32
    ds = ToyDS(128, d)
    model = ToyLM(d)

    print(check_applicability(stack="hf", n_source=64, n_target=64).summary())

    with torch.no_grad():
        h0 = model.body(ds.x[:64])
        h1 = model.body(ds.x[64:] + 0.3)
    artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=4), h0, h1)

    PMHTrainer = get_pmh_trainer()
    trainer = PMHTrainer.from_artifact(
        artifact,
        PMHConfig.balanced(),
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_g3b",
            per_device_train_batch_size=16,
            num_train_epochs=2,
            report_to="none",
            save_strategy="no",
        ),
        train_dataset=ds,
        representation_fn=lambda m, inp: m.body(inp["input_ids"]),
    )
    trainer.train()
    print("preflight:", artifact.preflight)


if __name__ == "__main__":
    main()
