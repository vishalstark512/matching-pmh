"""Domain shift: estimate Sigma_task with D4 and run matched PMH."""

import os

import torch
import torch.nn as nn

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, collect_features, estimate_from_config

_QUICK = os.environ.get("PMH_QUICK", "").lower() in ("1", "true", "yes")


class Backbone(nn.Module):
    def __init__(self, d_in: int = 32, d: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def finite_batches(n_batches: int, batch: int, shift: float):
    for _ in range(n_batches):
        yield torch.randn(batch, 32) + shift


def main() -> None:
    torch.manual_seed(0)
    backbone = Backbone()
    head = nn.Linear(16, 2)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)

    n_collect = 5 if _QUICK else 20
    batch = 32

    backbone.eval()
    h_src = collect_features(
        backbone, finite_batches(n_collect, batch, 0.0), max_batches=n_collect
    )
    h_tgt = collect_features(
        backbone, finite_batches(n_collect, batch, 0.8), max_batches=n_collect
    )

    cfg = SigmaTaskConfig.for_domain(rank=6)
    artifact = estimate_from_config(cfg, h_src, h_tgt)
    print(f"method={artifact.method}  dim={artifact.dim}  preflight={artifact.preflight}")

    pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=1))
    backbone.train()
    n_epochs = 3 if _QUICK else 15
    for epoch in range(1, n_epochs + 1):
        pmh.set_epoch(epoch)
        x = torch.randn(64, 32)
        y = torch.randint(0, 2, (64,))
        opt.zero_grad()
        h = backbone(x)
        task = nn.functional.cross_entropy(head(h), y)
        total, raw = pmh.capped_total(task, h)
        total.backward()
        opt.step()
        if epoch == 1 or epoch == n_epochs or epoch % 5 == 0:
            print(f"epoch {epoch:2d}  task={task.item():.4f}  pmh={raw.item():.4f}")


if __name__ == "__main__":
    main()
