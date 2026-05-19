"""Load estimator + training settings from JSON (no PyYAML dependency)."""

import json

import torch

from pmh import PMHConfig, SigmaTaskConfig, estimate_from_config


def main() -> None:
    job = {
        "estimator": {"method": "D4", "rank": 32, "shrinkage": 1e-5},
        "training": {"weight": 0.25, "cap_ratio": 0.3, "warmup_epochs": 2},
    }

    est_cfg = SigmaTaskConfig.from_dict(job["estimator"])
    pmh_cfg = PMHConfig.from_dict(job["training"])
    print("estimator:", est_cfg)
    print("training:", pmh_cfg)

    src = torch.randn(80, 24)
    tgt = src + 0.3 * torch.randn(80, 24)
    artifact = estimate_from_config(est_cfg, src, tgt)
    print("artifact method:", artifact.method, "preflight:", artifact.preflight)
    print(json.dumps(est_cfg.to_dict(), indent=2))


if __name__ == "__main__":
    main()
