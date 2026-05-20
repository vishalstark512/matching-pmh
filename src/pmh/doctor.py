"""Environment and setup diagnostics (``pmh-train doctor``)."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Literal

from pmh.adoption import RECIPE_ONE_LINER, format_newbie_checklist, format_recipe_banner
from pmh.onboarding import preflight_plain_english

Stack = Literal["pytorch", "sklearn", "hf"]

_SUGGESTED_DOC = {
    "pytorch": "docs/FIVE_STEP_RECIPE.md",
    "sklearn": "docs/GOLDEN_PATHS.md#g2",
    "hf": "docs/GOLDEN_PATHS.md#g3",
}


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("matching-pmh")
    except Exception:
        return "unknown"


@dataclass
class DoctorReport:
    """Human-readable environment check + newbie pipeline checklist."""

    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    suggested_path: str = "docs/FIVE_STEP_RECIPE.md"
    stack: Stack = "pytorch"
    artifact_preflight: str | None = None
    artifact_eigengap: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            format_recipe_banner(),
            "",
            "matching-pmh doctor",
            f"  package version: {_package_version()}",
            f"  stack: {self.stack}",
        ]
        for c in self.checks:
            lines.append(f"  ok: {c}")
        for w in self.warnings:
            lines.append(f"  warn: {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        if self.artifact_preflight is not None:
            lines.append(
                f"  artifact preflight: {self.artifact_preflight} — "
                f"{preflight_plain_english(self.artifact_preflight)}"
            )
            if self.artifact_eigengap is not None:
                lines.append(f"  artifact eigengap: {self.artifact_eigengap:.4f}")
        lines.extend(["", format_newbie_checklist(self.stack)])
        lines.append("")
        lines.append(f"  next doc: {self.suggested_path}")
        lines.append(f"  quick try: python examples/00_first_run_domain_shift.py")
        if self.stack == "sklearn":
            lines.append("  G2 demo: python -c \"from pmh import load_g2_demo_arrays; print(load_g2_demo_arrays()[0].shape)\"")
        return "\n".join(lines)


def _check_artifact(artifact_path: str, *, rank: int | None) -> tuple[str | None, float | None, list[str], list[str]]:
    """Load artifact and return preflight status, eigengap, warnings, errors."""
    warnings: list[str] = []
    errors: list[str] = []
    try:
        from pmh.artifact import SigmaTaskEstimate
        from pmh.diagnostics import eigengap_ratio
        from pmh.preflight import preflight_eigengap

        art = SigmaTaskEstimate.load(artifact_path)
        r = rank or art.config.rank or 16
        gamma = eigengap_ratio(art.sigma, r)
        status, _ = preflight_eigengap(art.sigma, r)
        pf = status.value if hasattr(status, "value") else str(status)
        if pf == "fail":
            errors.append(
                "artifact preflight FAIL — fix estimate (more data / rank) before Step 5 claims"
            )
        elif pf == "marginal":
            warnings.append(
                "artifact preflight MARGINAL — run falsification arms; matched may not beat CORAL"
            )
        return pf, gamma, warnings, errors
    except Exception as exc:
        errors.append(f"could not load artifact {artifact_path!r}: {exc}")
        return None, None, warnings, errors


def run_doctor(
    *,
    stack: Stack = "pytorch",
    artifact_path: str | None = None,
    rank: int | None = None,
) -> DoctorReport:
    """Check imports, optional extras, and print the newbie pipeline checklist."""
    rep = DoctorReport(stack=stack, suggested_path=_SUGGESTED_DOC.get(stack, "docs/FIVE_STEP_RECIPE.md"))
    rep.checks.append(f"Python {sys.version.split()[0]} on {platform.system()}")
    rep.checks.append(RECIPE_ONE_LINER)

    try:
        import torch

        rep.checks.append(f"torch {torch.__version__}")
    except ImportError:
        rep.errors.append("torch not installed (required for PyTorch path)")
        return rep

    try:
        import pmh  # noqa: F401

        rep.checks.append("import pmh")
    except ImportError as exc:
        rep.errors.append(f"import pmh failed: {exc}")
        return rep

    if stack == "sklearn":
        try:
            import sklearn

            rep.checks.append(f"sklearn {sklearn.__version__}")
        except ImportError:
            rep.errors.append('sklearn missing — pip install "matching-pmh[sklearn]"')
        else:
            try:
                from pmh import load_g2_demo_arrays

                x, y, xt, yt = load_g2_demo_arrays(n=40, seed=0)
                rep.checks.append(f"G2 demo arrays ok: source {x.shape}, target {xt.shape}")
            except Exception as exc:
                rep.warnings.append(f"G2 demo check failed: {exc}")
        if artifact_path:
            pf, gap, w, e = _check_artifact(artifact_path, rank=rank)
            rep.artifact_preflight, rep.artifact_eigengap = pf, gap
            rep.warnings.extend(w)
            rep.errors.extend(e)
        return rep

    if stack == "hf":
        try:
            import transformers  # noqa: F401

            rep.checks.append("transformers available")
        except ImportError:
            rep.errors.append('transformers missing — pip install "matching-pmh[hf]"')
        if artifact_path:
            pf, gap, w, e = _check_artifact(artifact_path, rank=rank)
            rep.artifact_preflight, rep.artifact_eigengap = pf, gap
            rep.warnings.extend(w)
            rep.errors.extend(e)
        return rep

    # pytorch default
    try:
        from pmh.integrations.lightning import lightning_available

        if lightning_available():
            rep.checks.append("lightning available (G1b)")
        else:
            rep.warnings.append('lightning not installed — optional G1b: pip install "matching-pmh[lightning]"')
    except Exception as exc:
        rep.warnings.append(f"lightning check skipped: {exc}")

    if artifact_path:
        pf, gap, w, e = _check_artifact(artifact_path, rank=rank)
        rep.artifact_preflight, rep.artifact_eigengap = pf, gap
        rep.warnings.extend(w)
        rep.errors.extend(e)

    return rep
