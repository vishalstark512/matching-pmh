"""Public API tiers for matching-pmh (semver contract).

Tier 0 — stable adoption surface (``from pmh import ...`` + ``pmh.__all__``).
Tier 1 — integrator / training primitives (importable from ``pmh``; documented in ARCHITECTURE.md).
Tier 2 — research & lazy exports (benchmark protocol, numpy_api, vision multilayer, …).

Physical package moves (Phase 2) will add submodules; names in Tier 0/1 stay importable from ``pmh`` until 2.0.
"""

from __future__ import annotations

# Adoption: "train on A, deploy on B" without reading D1–D7 first.
TIER_0: tuple[str, ...] = (
    "__version__",
    "PMHMatcher",
    "PMHTrainer",
    "PMHConfig",
    "check_applicability",
    "robust_fit",
    "evaluate_baseline_vs_pmh",
    "evaluate_falsification_arms",
    "evaluate_robust_fit",
    "explain_task",
    "explain_nuisance_key",
    "format_shift_types",
    "get_task",
    "list_tasks",
)

# Wiring estimate → train → ship; golden-path code samples.
TIER_1: tuple[str, ...] = (
    "PMHLoss",
    "SigmaTaskConfig",
    "SigmaTaskEstimate",
    "estimate_sigma_task",
    "estimate_from_config",
    "suggest_hook",
    "suggest_nuisance",
    "suggest_subtype",
    "resolve_hook",
    "collect_features",
    "export_deployment",
    "load_deployment_bundle",
    "compare_arms",
    "compare_arms_sklearn",
    "load_g2_demo_arrays",
    "run_doctor",
    "preflight_eigengap",
    "robust_fit_text_domains",
    "make_pmh_pipeline",
    "build_hybrid_trainer",
    "format_five_step_guide",
    "plan_recipe",
    "step_identify",
    "step_scope",
    "ShiftIdentification",
    "RecipePlan",
)

# Lazy or research-only; prefer explicit submodule imports in new code.
TIER_2_LAZY: tuple[str, ...] = (
    "HFPMHTrainer",
    "get_pmh_trainer",
    "lightning_available",
    "PMHCallback",
    "train_epoch_with_pmh",
    "PMHStepResult",
    "MultiLayerPMHLoss",
    "gram_sample_noise",
    "run_benchmark_protocol",
    "train_one_arm",
    "write_benchmark_report",
    "benchmark_to_markdown",
    "run_sklearn_benchmark",
    "STANDARD_ARMS",
    "estimate_sigma_task_numpy",
    "gram_from_diff_numpy",
    "MatchedSubspaceProjector",
    "project_onto_complement",
)


def tier_of(name: str) -> int | None:
    """Return 0, 1, 2, or None if not a registered public name."""
    if name in TIER_0:
        return 0
    if name in TIER_1:
        return 1
    if name in TIER_2_LAZY:
        return 2
    return None


def list_public_names(*, tier: int | None = None) -> list[str]:
    """Names by tier (omit ``__version__`` when tier is None)."""
    groups: list[tuple[str, ...]] = []
    if tier is None or tier == 0:
        groups.append(TIER_0)
    if tier is None or tier == 1:
        groups.append(TIER_1)
    if tier is None or tier == 2:
        groups.append(TIER_2_LAZY)
    out: list[str] = []
    for g in groups:
        out.extend(g)
    if tier is not None:
        return sorted(set(out))
    return sorted({n for n in out if n != "__version__"})
