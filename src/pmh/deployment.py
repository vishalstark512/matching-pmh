"""Export a reproducible PMH deployment bundle (artifact + manifest)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pmh.artifact import SigmaTaskEstimate
from pmh.config import PMHConfig


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("matching-pmh")
    except Exception:
        return "unknown"


_DEFAULT_README = """matching-pmh deployment bundle
=============================

Files:
  sigma_task.pt / sigma_task.json  — estimated Sigma_task (Phase A)
  manifest.json                    — version, method, preflight, library version
  pmh_config.json                  — training penalty settings (Phase B), if provided

Load in Python:
  from pmh import SigmaTaskEstimate, PMHTrainer, PMHConfig
  art = SigmaTaskEstimate.load("sigma_task.pt")
  trainer = PMHTrainer.from_artifact(model, art, hook=YOUR_HOOK, pmh_config=PMHConfig.balanced())

Docs: https://github.com/vishalstark512/matching-pmh/blob/main/docs/DEPLOYMENT.md
"""


@dataclass
class DeploymentBundle:
    """Paths written by :func:`export_deployment`."""

    root: Path
    sigma_pt: Path
    sigma_json: Path
    manifest: Path
    pmh_config: Path | None = None
    readme: Path | None = None

    def to_dict(self) -> dict[str, str]:
        out = {
            "root": str(self.root),
            "sigma_pt": str(self.sigma_pt),
            "sigma_json": str(self.sigma_json),
            "manifest": str(self.manifest),
        }
        if self.pmh_config:
            out["pmh_config"] = str(self.pmh_config)
        if self.readme:
            out["readme"] = str(self.readme)
        return out


def export_deployment(
    artifact: SigmaTaskEstimate,
    output_dir: str | Path,
    *,
    pmh_config: PMHConfig | None = None,
    hook: str | None = None,
    nuisance: str | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
    write_readme: bool = True,
) -> DeploymentBundle:
    """Write artifact + manifest for handoff (MLOps / another environment).

    Parameters
    ----------
    artifact
        Fitted :class:`~pmh.artifact.SigmaTaskEstimate`.
    output_dir
        Directory created if missing.
    pmh_config
        Optional Phase-B settings saved as ``pmh_config.json``.
    hook, nuisance
        Recorded in manifest for operators (not validated here).
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    sigma_pt = artifact.save(root / "sigma_task")
    sigma_json = sigma_pt.with_suffix(".json")

    manifest: dict[str, Any] = {
        "matching_pmh_version": _package_version(),
        "method": artifact.method,
        "dim": artifact.dim,
        "preflight": artifact.preflight,
        "eigengap": artifact.eigengap,
        "config": artifact.config.to_dict(),
        "hook": hook,
        "nuisance": nuisance,
        "notes": notes,
    }
    if extra:
        manifest["extra"] = extra

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    pmh_path: Path | None = None
    if pmh_config is not None:
        pmh_path = root / "pmh_config.json"
        pmh_path.write_text(json.dumps(pmh_config.to_dict(), indent=2), encoding="utf-8")

    readme_path: Path | None = None
    if write_readme:
        readme_path = root / "README.txt"
        readme_path.write_text(_DEFAULT_README, encoding="utf-8")

    return DeploymentBundle(
        root=root,
        sigma_pt=sigma_pt,
        sigma_json=sigma_json,
        manifest=manifest_path,
        pmh_config=pmh_path,
        readme=readme_path,
    )


def load_deployment_bundle(
    directory: str | Path,
    *,
    map_location: str = "cpu",
) -> tuple[SigmaTaskEstimate, dict[str, Any], PMHConfig | None]:
    """Load artifact + manifest (+ optional PMHConfig) from an export directory."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    pt = root / "sigma_task.pt"
    if not pt.is_file():
        pt = Path(manifest.get("tensor_file", "sigma_task.pt"))
        if not pt.is_absolute():
            pt = root / pt
    artifact = SigmaTaskEstimate.load(pt, map_location=map_location)
    pmh_cfg: PMHConfig | None = None
    pc = root / "pmh_config.json"
    if pc.is_file():
        pmh_cfg = PMHConfig.from_dict(json.loads(pc.read_text(encoding="utf-8")))
    return artifact, manifest, pmh_cfg
