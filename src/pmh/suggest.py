"""Suggest nuisance type (D1–D7) from data you have."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pmh.config import Method


@dataclass(frozen=True)
class NuisanceSuggestion:
    """Recommended shift type (``nuisance=`` API key) and Lemma method."""

    nuisance: str
    method: Method
    reason: str

    def plain_summary(self) -> str:
        """One block for CLI / notebooks (includes shift-type hint)."""
        from pmh.applications import explain_nuisance_key

        try:
            detail = explain_nuisance_key(self.nuisance)
        except Exception:
            detail = f"nuisance={self.nuisance!r} ({self.method})"
        return f"{self.reason}\n{detail}"


def suggest_nuisance(
    *,
    has_source_labels: bool = True,
    has_target_labels: bool = False,
    has_target_domain: bool = True,
    has_augmentation_modes: bool = False,
    has_style_pairs: bool = False,
    has_temporal_sequences: bool = False,
    has_nuisance_indices: bool = False,
    noise_level_known: bool = False,
) -> NuisanceSuggestion:
    """Rule-based **deployment shift type** (``nuisance=`` key) from data you have.

    Prefer ``pmh-train shifts`` or :func:`explain_nuisance_key` for plain English.
    """
    if has_style_pairs:
        return NuisanceSuggestion("style", "D7", "Style/content-fixed pairs -> D7 alignment Gram.")
    if has_augmentation_modes:
        return NuisanceSuggestion("augmentation", "D3", "Known augmentation modes -> D3.")
    if has_temporal_sequences:
        return NuisanceSuggestion("temporal", "D6", "Label-constant drift along time -> D6.")
    if has_nuisance_indices:
        return NuisanceSuggestion("compositional", "D5", "Nuisance on known coordinates -> D5.")
    if noise_level_known and not has_target_domain:
        return NuisanceSuggestion("isotropic", "D2", "Known noise level, no domain pair -> D2.")
    if has_source_labels and has_target_labels and has_target_domain:
        return NuisanceSuggestion(
            "subspace",
            "D1",
            "Paired domains with class labels -> D1 subspace (class-aligned).",
        )
    if has_target_domain:
        return NuisanceSuggestion(
            "domain_shift",
            "D4",
            "Source/target domains without requiring target labels -> D4 domain Gram.",
        )
    return NuisanceSuggestion(
        "domain_shift",
        "D4",
        "Default: cross-domain shift with frozen/paired features.",
    )


def resolve_nuisance_arg(nuisance: str, **data_flags: Any) -> str:
    """Expand ``nuisance='auto'`` via :func:`suggest_nuisance`."""
    if nuisance.strip().lower() not in ("auto", "automatic"):
        return nuisance
    sug = suggest_nuisance(**data_flags)
    return sug.nuisance
