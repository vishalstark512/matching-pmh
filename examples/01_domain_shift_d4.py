"""Domain shift: estimate Sigma_task with D4 and run matched PMH."""

import torch
import torch.nn as nn

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, collect_features, estimate_from_config


class Backbone(nn.Module):
    def __init__(self, d_in: int = 32, d: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loader(n: int, shift: float, batch: int = 32):
    while True:
        yield torch.randn(batch, 32) + shift


def main() -> None:
    torch.manual_seed(0)
    backbone = Backbone()
    head = nn.Linear(16, 2)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)

    # --- Phase 1: estimate Sigma from frozen backbone features ---
    backbone.eval()
    src_batches = [b for i, b in enumerate(make_loader(512, 0.0)) if i < 20]
    tgt_batches = [b for i, b in enumerate(make_loader(512, 0.8)) if i < 20]
    h_src = collect_features(backbone, src_batches, max_batches=20)
    h_tgt = collect_features(backbone, tgt_batches, max_batches=20)

    cfg = SigmaTaskConfig.for_domain(rank=6)
    artifact = estimate_from_config(cfg, h_src, h_tgt)
    print(f"method={artifact.method}  dim={artifact.dim}  preflight={artifact.preflight}")

    # --- Phase 2: train with PMHLoss ---
    pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=1))
    backbone.train()
    for epoch in range(1, 31):
        pmh.set_epoch(epoch)
        x = torch.randn(64, 32)
        y = torch.randint(0, 2, (64,))
        opt.zero_grad()
        h = backbone(x)
        task = nn.functional.cross_entropy(head(h), y)
        total, raw = pmh.capped_total(task, h)
        total.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"epoch {epoch:2d}  task={task.item():.4f}  pmh={raw.item():.4f}")


if __name__ == "__main__":
    main()
