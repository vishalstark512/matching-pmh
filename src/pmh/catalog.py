"""Catalog of Lemma D1--D7 estimators: inputs, factories, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pmh.config import Method, SigmaTaskConfig


@dataclass(frozen=True)
class MethodSpec:
    """What each nuisance type needs to estimate Sigma_task."""

    method: Method
    name: str
    assumption: str
    typical_tasks: str
    required_data: tuple[str, ...]
    optional_data: tuple[str, ...]
    config_fields: tuple[str, ...]


METHODS: dict[str, MethodSpec] = {
    "D1": MethodSpec(
        "D1",
        "Subspace (cross-domain SVD)",
        "A1: low-rank subspace W",
        "T1 digits, Office-31",
        ("source_features", "source_labels", "target_features", "target_labels"),
        ("source_npy", "target_npy"),
        ("rank", "shrinkage"),
    ),
    "D2": MethodSpec(
        "D2",
        "Isotropic noise",
        "A2: N(0, sigma^2 I)",
        "T2 sensor / acquisition",
        (),
        ("dim", "noise_level"),
        ("shrinkage",),
    ),
    "D3": MethodSpec(
        "D3",
        "Augmentation modes",
        "A3: finite aug coefficients",
        "T3 photometric",
        ("aug_deltas",),
        ("aug_npy",),
        ("shrinkage",),
    ),
    "D4": MethodSpec(
        "D4",
        "Domain Gram",
        "A4: paired domain shift",
        "T4 vision domain",
        ("source_features", "target_features"),
        ("source_npy", "target_npy"),
        ("rank", "shrinkage"),
    ),
    "D5": MethodSpec(
        "D5",
        "Compositional block",
        "A5: nuisance coordinates",
        "T5 atoms / tokens",
        ("features",),
        ("features_npy",),
        ("nuisance_indices", "shrinkage"),
    ),
    "D6": MethodSpec(
        "D6",
        "Temporal residual",
        "A6: label-constant drift",
        "T6 sensor sequences",
        ("sequences",),
        ("sequences_npy",),
        ("shrinkage",),
    ),
    "D7": MethodSpec(
        "D7",
        "Style / alignment Gram",
        "A7: style pairs or PGD deltas",
        "T7A LLM alignment",
        ("style_jsonl",),
        ("deltas_npy", "model_id"),
        ("rank", "shrinkage", "max_pairs", "batch_size"),
    ),
}


def list_methods() -> list[MethodSpec]:
    return [METHODS[k] for k in ("D1", "D2", "D3", "D4", "D5", "D6", "D7")]


def config_from_job(estimator: dict[str, Any]) -> SigmaTaskConfig:
    """Build :class:`SigmaTaskConfig` from a JSON job ``estimator`` block."""
    return SigmaTaskConfig.from_dict(estimator)


def validate_job_data(method: str, data: dict[str, Any]) -> list[str]:
    """Return list of missing required keys for a job data block."""
    spec = METHODS[method.upper()]
    missing: list[str] = []
    if method.upper() == "D2":
        if "dim" not in data and "representation_dim" not in data:
            missing.append("dim")
        return missing
    key_aliases = {
        "source_features": ("source_npy",),
        "target_features": ("target_npy",),
        "features": ("features_npy",),
        "sequences": ("sequences_npy",),
        "aug_deltas": ("aug_npy",),
        "style_jsonl": ("deltas_npy",),
    }
    for req in spec.required_data:
        if req in data:
            continue
        alts = key_aliases.get(req, ())
        if any(a in data for a in alts):
            continue
        missing.append(req)
    if method.upper() == "D5" and "nuisance_indices" not in data:
        missing.append("nuisance_indices")
    if method.upper() in ("D1",) and "rank" not in data:
        missing.append("rank")
    if method.upper() == "D7" and "deltas_npy" not in data and "model_id" not in data:
        missing.append("model_id (or precomputed deltas_npy)")
    return missing
