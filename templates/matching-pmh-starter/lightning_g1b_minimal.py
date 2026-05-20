"""G1b — PyTorch Lightning + PMH (domain shift on hook h)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, check_applicability, estimate_from_config
from pmh.integrations.lightning import PMHLightningCallback, add_pmh_to_loss, _require_lightning


class Net(nn.Module):
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

    print(check_applicability(stack="pytorch", n_source=100, n_target=100).summary())

    net = Net()
    with torch.no_grad():
        h_src = net.backbone(x[:100])
        h_tgt = net.backbone(x[100:] + 0.5)
    artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=4), h_src, h_tgt)
    pmh_cfg = PMHConfig.balanced()

    class Lit(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.net = Net()
            self.pmh_loss = PMHLoss(artifact, pmh_cfg)

        def training_step(self, batch, batch_idx: int = 0) -> torch.Tensor:
            xb, yb = batch
            task = F.cross_entropy(self.net(xb), yb)
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
    trainer.fit(Lit(), loader)
    print("preflight:", artifact.preflight)


if __name__ == "__main__":
    main()
