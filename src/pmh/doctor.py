"""Environment and setup diagnostics (``pmh-train doctor``)."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Literal

Stack = Literal["pytorch", "sklearn", "hf"]


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("matching-pmh")
    except Exception:
        return "unknown"



@dataclass
class DoctorReport:
    """Human-readable environment check."""

    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    suggested_path: str = "docs/index.md"

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = ["matching-pmh doctor", f"  package version: {_package_version()}"]
        for c in self.checks:
            lines.append(f"  ok: {c}")
        for w in self.warnings:
            lines.append(f"  warn: {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        lines.append(f"  suggested doc: {self.suggested_path}")
        return "\n".join(lines)


def run_doctor(*, stack: Stack = "pytorch") -> DoctorReport:
    """Check imports and optional extras for the chosen integration stack."""
    rep = DoctorReport()
    rep.checks.append(f"Python {sys.version.split()[0]} on {platform.system()}")

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
        rep.suggested_path = "docs/GOLDEN_PATHS.md#g2"
        return rep

    if stack == "hf":
        try:
            import transformers  # noqa: F401

            rep.checks.append("transformers available")
        except ImportError:
            rep.errors.append('transformers missing — pip install "matching-pmh[hf]"')
        rep.suggested_path = "docs/GOLDEN_PATHS.md#g3b"
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

    rep.suggested_path = "docs/START_HERE.md"
    return rep
