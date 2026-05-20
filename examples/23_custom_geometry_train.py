"""Custom geometry: deltas → artifact → PMHTrainer.from_artifact (no paper scripts)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, artifact_from_deltas, estimate_custom
from pmh.data_adapters import batch_iterators, load_domain_arrays
from pmh.trainer import PMHTrainer


class Tiny(nn.Module):
    def __init__(self, d: int = 12, c: int = 3) -> None:
        super().__init__()
        self.enc = nn.Linear(d, 8)
        self.head = nn.Linear(8, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.enc(x)))


def main() -> None:
    torch.manual_seed(0)
    d, n = 12, 120
    x_src = torch.randn(n, d)
    x_tgt = x_src + 0.5 * torch.randn(n, d)
    y = torch.randint(0, 3, (n,))

    # Path 1: domain Gram (D4) from tensors
    art_d4 = estimate_custom(
        x_src=x_src.numpy(),
        x_tgt=x_tgt.numpy(),
        method="D4",
        rank=4,
    )
    print("D4 preflight:", art_d4.preflight)

    # Path 2: custom deltas (D7-style identification, same pipeline)
    deltas = (x_tgt - x_src).detach().numpy()
    art_custom = artifact_from_deltas(deltas, method="D7", rank=4)
    print("custom deltas method:", art_custom.method)

    train_loader = DataLoader(TensorDataset(x_src, y), batch_size=16, shuffle=True)
    src_it, tgt_it = batch_iterators(x_src.numpy(), x_tgt.numpy(), batch_size=16)

    model = Tiny(d)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sigma.pt"
        art_d4.save(path)

        trainer = PMHTrainer.from_artifact(
            model,
            path,
            hook="enc",
            head=model.head,
            pmh_config=PMHConfig.balanced(),
        )
        stats = trainer.fit(
            train_loader,
            source_batches=src_it,
            target_batches=tgt_it,
            epochs=3,
            max_steps_per_epoch=5,
        )
        print("train steps:", stats.get("n_steps"), "preflight:", trainer.artifact_.preflight)

    xs, _, xt, _ = load_domain_arrays(x_src.numpy(), x_tgt.numpy())
    print("loaded arrays", xs.shape, xt.shape)


if __name__ == "__main__":
    main()
