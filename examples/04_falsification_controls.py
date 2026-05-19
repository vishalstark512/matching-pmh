"""Matched vs wrong-W vs isotropic PMH on the same batch."""

import torch
import torch.nn as nn

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, estimate_from_config


def main() -> None:
    torch.manual_seed(2)
    d = 16
    h = torch.randn(32, d, requires_grad=True)
    est = estimate_from_config(SigmaTaskConfig.for_isotropic(d, 0.12))

    cfg = PMHConfig(weight=1.0, warmup_epochs=0, cap_ratio=0)
    matched = PMHLoss(est, cfg, mode="matched")(h)
    wrong = PMHLoss(est, cfg, mode="wrong_w", wrong_rank=4)(h)
    iso = PMHLoss(est, cfg, mode="isotropic")(h)

    print(f"matched PMH:  {matched.item():.6f}")
    print(f"wrong-W PMH:  {wrong.item():.6f}")
    print(f"isotropic PMH: {iso.item():.6f}")
    print("(Expect wrong-W ≈ isotropic in expectation over many seeds; single seed may differ.)")


if __name__ == "__main__":
    main()
