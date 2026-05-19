"""Human-readable nuisance names → Lemma D1–D7 methods."""

from __future__ import annotations

from typing import Any

from pmh.config import Method, SigmaTaskConfig

# Aliases users can pass to PMHMatcher(nuisance=...)
NUISANCE_ALIASES: dict[str, Method] = {
    "d1": "D1",
    "d2": "D2",
    "d3": "D3",
    "d4": "D4",
    "d5": "D5",
    "d6": "D6",
    "d7": "D7",
    "subspace": "D1",
    "paired_domains": "D1",
    "cross_domain": "D1",
    "isotropic": "D2",
    "noise": "D2",
    "augmentation": "D3",
    "aug": "D3",
    "domain_shift": "D4",
    "domain": "D4",
    "covariate_shift": "D4",
    "compositional": "D5",
    "coordinates": "D5",
    "temporal": "D6",
    "drift": "D6",
    "style": "D7",
    "alignment": "D7",
    "llm_style": "D7",
}


def resolve_method(nuisance: str) -> Method:
    """Map ``nuisance`` string to ``D1``–``D7``."""
    key = nuisance.strip().lower().replace("-", "_")
    if key in NUISANCE_ALIASES:
        return NUISANCE_ALIASES[key]
    upper = nuisance.strip().upper()
    if upper in ("D1", "D2", "D3", "D4", "D5", "D6", "D7"):
        return upper  # type: ignore[return-value]
    known = ", ".join(sorted(set(NUISANCE_ALIASES.keys())))
    raise ValueError(f"Unknown nuisance {nuisance!r}. Try one of: {known}, or D1–D7.")


def default_rank(*, dim: int, n_samples: int, requested: int | None) -> int:
    """Conservative default subspace rank from feature dim and sample size."""
    if requested is not None:
        return min(requested, dim)
    return min(32, max(1, dim // 4, min(n_samples, dim) // 10))


def config_from_nuisance(
    nuisance: str,
    *,
    rank: int | None = None,
    shrinkage: float = 1e-6,
    dim: int | None = None,
    noise_level: float = 0.1,
    nuisance_indices: list[int] | None = None,
    **kwargs: Any,
) -> SigmaTaskConfig:
    """Build :class:`SigmaTaskConfig` from a nuisance name."""
    method = resolve_method(nuisance)
    if method == "D1":
        if rank is None:
            raise ValueError("nuisance='subspace' (D1) requires rank= (or pass rank to PMHMatcher)")
        return SigmaTaskConfig.for_subspace(rank=rank, shrinkage=shrinkage, **kwargs)
    if method == "D2":
        if dim is None:
            raise ValueError("nuisance='isotropic' (D2) requires dim= (or fit on data to infer dim)")
        return SigmaTaskConfig.for_isotropic(dim=dim, noise_level=noise_level, shrinkage=shrinkage, **kwargs)
    if method == "D3":
        return SigmaTaskConfig.for_augmentation(shrinkage=shrinkage, **kwargs)
    if method == "D4":
        return SigmaTaskConfig.for_domain(rank=rank, shrinkage=shrinkage, **kwargs)
    if method == "D5":
        if nuisance_indices is None:
            raise ValueError("nuisance='compositional' (D5) requires nuisance_indices=")
        return SigmaTaskConfig.for_compositional(nuisance_indices, shrinkage=shrinkage, **kwargs)
    if method == "D6":
        return SigmaTaskConfig.for_temporal(shrinkage=shrinkage, **kwargs)
    return SigmaTaskConfig.for_alignment(rank=rank or 32, shrinkage=shrinkage, **kwargs)


def list_nuisance_names() -> list[str]:
    """Sorted alias names for documentation and CLI."""
    return sorted(NUISANCE_ALIASES.keys())
