"""Standard training arms for credible A/B comparison (B0, matched, controls)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ArmName = Literal["b0", "matched", "wrong_w", "isotropic", "coral"]

# Paper-style four-arm falsification suite (+ optional CORAL on features).
STANDARD_ARMS: tuple[ArmName, ...] = ("b0", "matched", "wrong_w", "isotropic")

# Task 7B naming (alias map for reports).
ARM_ALIASES: dict[str, str] = {
    "baseline": "b0",
    "matched": "matched",
    "pmh_matched": "matched",
    "pmh_aniso": "matched",
    "wrong_w": "wrong_w",
    "pmh_aniso_wrong_W": "wrong_w",
    "isotropic": "isotropic",
    "pmh_iso": "isotropic",
    "coral": "coral",
}


@dataclass(frozen=True)
class ArmSpec:
    name: ArmName
    label: str
    description: str
    uses_pmh: bool
    pmh_mode: str | None  # PMHLoss mode; None for b0 / coral


ARM_SPECS: dict[ArmName, ArmSpec] = {
    "b0": ArmSpec(
        "b0",
        "B0 (ERM)",
        "Task loss only; no PMH penalty.",
        uses_pmh=False,
        pmh_mode=None,
    ),
    "matched": ArmSpec(
        "matched",
        "Matched PMH",
        "PMH along estimated Sigma_task (deployment nuisance).",
        uses_pmh=True,
        pmh_mode="matched",
    ),
    "wrong_w": ArmSpec(
        "wrong_w",
        "Wrong-W",
        "Random rank-r subspace (Lemma C control; should resemble isotropic).",
        uses_pmh=True,
        pmh_mode="wrong_w",
    ),
    "isotropic": ArmSpec(
        "isotropic",
        "Isotropic PMH",
        "Uniform Sigma' proportional to I (generic VAT-like).",
        uses_pmh=True,
        pmh_mode="isotropic",
    ),
    "coral": ArmSpec(
        "coral",
        "CORAL",
        "Feature alignment baseline (Sun & Saenko); numpy/sklearn path only.",
        uses_pmh=False,
        pmh_mode=None,
    ),
}


def normalize_arm(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return ARM_ALIASES.get(key, key)


def resolve_arms(names: list[str] | None) -> tuple[ArmName, ...]:
    if not names:
        return STANDARD_ARMS
    out: list[ArmName] = []
    for n in names:
        a = normalize_arm(n)
        if a not in ARM_SPECS:
            raise ValueError(f"unknown arm {n!r}; choose from {list(ARM_SPECS)}")
        if a not in out:
            out.append(a)  # type: ignore[arg-type]
    return tuple(out)  # type: ignore[return-value]
