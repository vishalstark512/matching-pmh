"""Compositional nuisance (D5): covariance on a coordinate block."""

import torch

from pmh import SigmaTaskConfig, estimate_from_config


def main() -> None:
    # 20 dims: first 5 are nuisance coordinates (e.g. atom indices)
    x = torch.randn(200, 20)
    nuisance_idx = list(range(5))
    artifact = estimate_from_config(
        SigmaTaskConfig.for_compositional(nuisance_idx),
        x,
    )
    print("Sigma shape:", tuple(artifact.sigma.shape))
    print("Non-zero block norm:", artifact.sigma[:5, :5].norm().item())
    print("Off-block norm (should be ~0):", artifact.sigma[5:, 5:].norm().item())


if __name__ == "__main__":
    main()
