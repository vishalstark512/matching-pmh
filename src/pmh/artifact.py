"""Sigma_task estimates as first-class artifacts (save / load / metadata)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pmh.config import SigmaTaskConfig

ARTIFACT_VERSION = 1


@dataclass
class SigmaTaskEstimate:
    """Estimated Sigma_task plus diagnostics."""

    sigma: torch.Tensor
    method: str
    config: SigmaTaskConfig
    eigengap: float | None = None
    preflight: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.sigma.shape[0])

    def save(self, path: str | Path) -> Path:
        """Save ``.pt`` bundle and human-readable ``.json`` sidecar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stem = path.with_suffix("") if path.suffix == ".json" else path
        pt_path = stem.with_suffix(".pt")
        json_path = stem.with_suffix(".json")

        payload = {
            "version": ARTIFACT_VERSION,
            "sigma": self.sigma.detach().cpu(),
            "method": self.method,
            "config": self.config.to_dict(),
            "eigengap": self.eigengap,
            "preflight": self.preflight,
            "metadata": self.metadata,
        }
        torch.save(payload, pt_path)

        sidecar = {
            "version": ARTIFACT_VERSION,
            "method": self.method,
            "dim": self.dim,
            "eigengap": self.eigengap,
            "preflight": self.preflight,
            "config": self.config.to_dict(),
            "metadata": self.metadata,
            "tensor_file": pt_path.name,
        }
        json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        return pt_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        encoder: Any = None,
    ) -> SigmaTaskEstimate:
        """Load from ``.pt`` (or pass path stem / ``.json`` to resolve sibling ``.pt``)."""
        path = Path(path)
        if path.suffix == ".json":
            meta = json.loads(path.read_text(encoding="utf-8"))
            pt_path = path.with_name(meta.get("tensor_file", path.stem + ".pt"))
        elif path.suffix == ".pt":
            pt_path = path
        else:
            pt_path = path.with_suffix(".pt")
            if not pt_path.is_file():
                raise FileNotFoundError(f"No artifact at {path} or {pt_path}")

        payload = torch.load(pt_path, map_location=map_location, weights_only=False)
        if payload.get("version", 0) != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported artifact version: {payload.get('version')}")

        cfg = SigmaTaskConfig.from_dict(payload["config"])
        if encoder is not None:
            cfg.encoder = encoder

        return cls(
            sigma=payload["sigma"],
            method=payload["method"],
            config=cfg,
            eigengap=payload.get("eigengap"),
            preflight=payload.get("preflight"),
            metadata=dict(payload.get("metadata", {})),
        )
