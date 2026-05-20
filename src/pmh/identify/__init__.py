"""Layer 1–2 entry — identify A_k and map to D_k / nuisance keys."""

from pmh.catalog import METHODS, MethodSpec
from pmh.nuisance import config_from_nuisance, list_nuisance_names, resolve_method
from pmh.recipe import ShiftIdentification, assumption_id, step_identify
from pmh.subtypes import (
    SubtypeInfo,
    SubtypeRecommendation,
    get_subtype,
    list_subtypes,
    suggest_subtype,
)
from pmh.suggest import NuisanceSuggestion, suggest_nuisance

__all__ = [
    "METHODS",
    "MethodSpec",
    "ShiftIdentification",
    "assumption_id",
    "step_identify",
    "config_from_nuisance",
    "list_nuisance_names",
    "resolve_method",
    "SubtypeInfo",
    "SubtypeRecommendation",
    "get_subtype",
    "list_subtypes",
    "suggest_subtype",
    "suggest_nuisance",
    "NuisanceSuggestion",
]
