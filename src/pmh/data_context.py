"""Describe your data for ``nuisance='auto'`` and docs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pmh.suggest import NuisanceSuggestion, suggest_nuisance


@dataclass
class DataContext:
    """Flags describing what data you have (for auto nuisance + validation)."""

    has_source_labels: bool = True
    has_target_labels: bool = False
    has_target_domain: bool = True
    has_augmentation_modes: bool = False
    has_style_pairs: bool = False
    has_temporal_sequences: bool = False
    has_nuisance_indices: bool = False
    noise_level_known: bool = False
    representation_dim: int | None = None
    notes: list[str] = field(default_factory=list)

    def suggest(self) -> NuisanceSuggestion:
        return suggest_nuisance(
            has_source_labels=self.has_source_labels,
            has_target_labels=self.has_target_labels,
            has_target_domain=self.has_target_domain,
            has_augmentation_modes=self.has_augmentation_modes,
            has_style_pairs=self.has_style_pairs,
            has_temporal_sequences=self.has_temporal_sequences,
            has_nuisance_indices=self.has_nuisance_indices,
            noise_level_known=self.noise_level_known,
        )

    def to_auto_kwargs(self) -> dict[str, Any]:
        return {
            "has_source_labels": self.has_source_labels,
            "has_target_labels": self.has_target_labels,
            "has_target_domain": self.has_target_domain,
            "has_augmentation_modes": self.has_augmentation_modes,
            "has_style_pairs": self.has_style_pairs,
        }

    @classmethod
    def from_batch_tuple(cls, batch: Any) -> DataContext:
        """Infer coarse flags from a single training batch."""
        ctx = cls()
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 2 and hasattr(batch[1], "shape"):
                ctx.has_source_labels = True
            if len(batch) >= 1:
                x = batch[0]
                if hasattr(x, "dim"):
                    if x.dim() == 3:
                        ctx.has_temporal_sequences = True
                        ctx.notes.append("3D input: consider temporal (D6) if label constant over time.")
        return ctx
