"""Layer 3 — apply matched PMH (Mode A Jacobian vs Mode B projection)."""

from __future__ import annotations

from typing import Literal

from pmh.matcher import PMHMatcher
from pmh.recipe import ApplicationMode, recommended_application_mode
from pmh.sklearn_match import MatchedSubspaceProjector, project_onto_complement
from pmh.train import PMHLoss, PMHTrainer, build_hybrid_trainer

ModeA = Literal["jacobian"]
ModeB = Literal["projection"]

__all__ = [
    "ApplicationMode",
    "ModeA",
    "ModeB",
    "PMHMatcher",
    "PMHTrainer",
    "PMHLoss",
    "build_hybrid_trainer",
    "MatchedSubspaceProjector",
    "project_onto_complement",
    "recommended_application_mode",
    "mode_a",
    "mode_b",
]


def mode_a() -> dict[str, type]:
    """Mode A exports (deep training + Jacobian penalty on hook h)."""
    return {"PMHTrainer": PMHTrainer, "PMHLoss": PMHLoss}


def mode_b() -> dict[str, type]:
    """Mode B exports (frozen features + projection / sklearn)."""
    return {"PMHMatcher": PMHMatcher, "MatchedSubspaceProjector": MatchedSubspaceProjector}
