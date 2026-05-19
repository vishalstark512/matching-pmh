"""Minimal training loop (v0.2 API with configs)."""

import torch
import torch.nn as nn

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, estimate_from_config


class Backbone(nn.Module):
    def __init__(self, d_in: int = 32, d: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    torch.manual_seed(0)
    backbone = Backbone()
    head = nn.Linear(16, 2)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)

    with torch.no_grad():
        h_src, h_tgt = backbone(torch.randn(256, 32)), backbone(torch.randn(256, 32) + 0.5)
        artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=8), h_src, h_tgt)

    pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3))
    for step in range(50):
        x = torch.randn(64, 32)
        y = torch.randint(0, 2, (64,))
        opt.zero_grad()
        h = backbone(x)
        task = nn.functional.cross_entropy(head(h), y)
        total, _ = pmh.capped_total(task, h)
        total.backward()
        opt.step()


if __name__ == "__main__":
    main()
