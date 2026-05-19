"""Save / load a Sigma_task artifact and reuse in training."""

import tempfile
from pathlib import Path

import torch

from pmh import SigmaTaskConfig, SigmaTaskEstimate, estimate_from_config


def main() -> None:
    torch.manual_seed(1)
    src = torch.randn(100, 12)
    tgt = src + 0.4 * torch.randn(100, 12)

    artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=4), src, tgt)
    print("eigengap:", artifact.eigengap, " preflight:", artifact.preflight)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "office31_style"
        pt = artifact.save(path)
        print("saved:", pt)

        loaded = SigmaTaskEstimate.load(pt)
        assert loaded.method == artifact.method
        assert torch.allclose(loaded.sigma, artifact.sigma)
        print("loaded dim:", loaded.dim, "config:", loaded.config.to_dict())


if __name__ == "__main__":
    main()
