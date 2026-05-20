"""Nuisance subtype D1–D7 registry (product model, not paper blocks)."""

from __future__ import annotations

from dataclasses import dataclass

from pmh.config import Method
from pmh.suggest import NuisanceSuggestion, suggest_nuisance


@dataclass(frozen=True)
class SubtypeRecommendation:
    """Subtype + nuisance string + plain-language reason (developer API)."""

    method: Method
    nuisance: str
    reason: str
    title: str
    similar_structure: str
    doc_anchor: str
    calibrator_note: str = ""


@dataclass(frozen=True)
class SubtypeInfo:
    """One nuisance subtype (structural class of deploy shift)."""

    method: Method
    nuisance: str
    title: str
    similar_structure: str
    default_mode: str  # "jacobian" | "projection" | "either"
    doc_anchor: str
    paper_exemplars: tuple[str, ...]
    calibrator_note: str = ""


_SUBTYPES: dict[str, SubtypeInfo] = {
    "D1": SubtypeInfo(
        "D1",
        "subspace",
        "Cross-domain subspace",
        "Same classes; different domains shift h along a low-rank subspace.",
        "either",
        "NUISANCE_SUBTYPES.md#d1-cross-domain-subspace",
        ("T1",),
        "",
    ),
    "D2": SubtypeInfo(
        "D2",
        "isotropic",
        "Isotropic sensitivity",
        "Uniform σ²I nuisance; no learned directions.",
        "jacobian",
        "NUISANCE_SUBTYPES.md#d2-isotropic",
        ("T2A", "T2B"),
        "",
    ),
    "D3": SubtypeInfo(
        "D3",
        "augmentation",
        "Augmentation modes",
        "Finite known transforms; Σ from aug-induced deltas.",
        "jacobian",
        "NUISANCE_SUBTYPES.md#d3-augmentation-modes",
        ("T3A", "T3B"),
        "Gradient-SVD: pmh.calibrate.gradient_subspace_numpy",
    ),
    "D4": SubtypeInfo(
        "D4",
        "domain_shift",
        "Domain Gram",
        "Unlabeled domain difference; pooled source−target Gram.",
        "jacobian",
        "NUISANCE_SUBTYPES.md#d4-domain-gram",
        ("T4A", "T4B"),
        "",
    ),
    "D5": SubtypeInfo(
        "D5",
        "compositional",
        "Compositional coordinates",
        "Nuisance on named coordinates of h (positions, tokens, nodes).",
        "jacobian",
        "NUISANCE_SUBTYPES.md#d5-compositional",
        ("T5A", "T5B"),
        "",
    ),
    "D6": SubtypeInfo(
        "D6",
        "temporal",
        "Temporal / sequence",
        "Label-constant drift along time or sensor trajectories.",
        "jacobian",
        "NUISANCE_SUBTYPES.md#d6-temporal-sequence",
        ("T6A", "T6B"),
        "Content-residual: pmh.calibrate.content_residual_subspace",
    ),
    "D7": SubtypeInfo(
        "D7",
        "style",
        "Style / alignment",
        "Same content, different surface form (format, tone, PGD δ).",
        "jacobian",
        "NUISANCE_SUBTYPES.md#d7-style-alignment",
        ("T7A", "T7B"),
        "PGD deltas: pmh.calibrate.subspace_artifact_from_deltas",
    ),
}

_WIZARD_CHOICES: dict[str, dict[str, bool]] = {
    "1": {
        "has_target_domain": True,
        "has_source_labels": True,
        "has_target_labels": True,
    },
    "2": {"has_target_domain": True, "has_source_labels": True, "has_target_labels": False},
    "3": {"has_augmentation_modes": True},
    "4": {"has_temporal_sequences": True},
    "5": {"has_nuisance_indices": True},
    "6": {"has_style_pairs": True},
    "7": {"noise_level_known": True, "has_target_domain": False},
}


def list_subtypes() -> list[str]:
    """Return method ids D1..D7."""
    return [f"D{k}" for k in range(1, 8)]


def get_subtype(method: str) -> SubtypeInfo:
    key = method.strip().upper()
    if not key.startswith("D"):
        key = f"D{key}"
    if key not in _SUBTYPES:
        raise KeyError(f"unknown subtype {method!r}; choose from {list_subtypes()}")
    return _SUBTYPES[key]


def suggest_from_flags(**flags: bool) -> NuisanceSuggestion:
    """Same as :func:`pmh.suggest.suggest_nuisance` (explicit subtype entry point)."""
    return suggest_nuisance(**flags)


def suggest_subtype(**flags: bool) -> SubtypeRecommendation:
    """Recommend D1–D7 from data flags (plain English + nuisance string)."""
    sug = suggest_nuisance(**flags)
    info = get_subtype(sug.method)
    return SubtypeRecommendation(
        method=info.method,
        nuisance=sug.nuisance,
        reason=sug.reason,
        title=info.title,
        similar_structure=info.similar_structure,
        doc_anchor=info.doc_anchor,
        calibrator_note=info.calibrator_note,
    )


def print_subtype_guide(method: str | None = None) -> None:
    """Print one or all subtype summaries (CLI / REPL helper)."""
    if method is None:
        for m in list_subtypes():
            print(format_subtype_line(m))
        return
    print(format_subtype_line(method))


def format_subtype_line(method: str) -> str:
    """One-line summary for CLI / wizard."""
    info = get_subtype(method)
    ex = ", ".join(info.paper_exemplars)
    line = f"{info.method} {info.title}: {info.similar_structure} (exemplars: {ex})"
    if info.calibrator_note:
        line += f" | refinement: {info.calibrator_note}"
    return line


def apply_wizard_subtype_choice(choice: str) -> dict[str, bool]:
    """Map wizard menu key '1'..'7' to suggest_nuisance flags."""
    key = choice.strip()
    if key not in _WIZARD_CHOICES:
        raise ValueError(f"subtype choice must be 1-7, got {choice!r}")
    base = {
        "has_source_labels": True,
        "has_target_labels": False,
        "has_target_domain": True,
        "has_augmentation_modes": False,
        "has_style_pairs": False,
        "has_temporal_sequences": False,
        "has_nuisance_indices": False,
        "noise_level_known": False,
    }
    base.update(_WIZARD_CHOICES[key])
    return base
