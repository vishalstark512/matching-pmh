"""Adoption routing and high-level developer helpers.

Prefer ``from pmh import explain_task, robust_fit`` (Tier 0). Submodule imports are stable but optional.
"""

from pmh.applications import (
    SHIFT_TYPES,
    ShiftTypePlain,
    explain_application,
    format_application_finder,
    format_search_results,
    format_shift_types,
    search_applications,
)
from pmh.developer import (
    ApplicabilityReport,
    DomainPair,
    EvaluationReport,
    HookSuggestion,
    RobustFitResult,
    check_applicability,
    evaluate_baseline_vs_pmh,
    evaluate_falsification_arms,
    evaluate_robust_fit,
    evaluate_trainer_on_loader,
    load_g2_demo_arrays,
    robust_fit,
    robust_fit_text_domains,
    suggest_hook,
)
from pmh.doctor import DoctorReport, run_doctor
from pmh.onboarding import (
    SetupRecommendation,
    format_setup_guide,
    preflight_plain_english,
    print_setup_guide,
    recommend_setup,
    run_wizard,
)
from pmh.subtypes import (
    SubtypeRecommendation,
    format_subtype_line,
    get_subtype,
    list_subtypes,
    print_subtype_guide,
    suggest_subtype,
)
from pmh.suggest import NuisanceSuggestion, suggest_nuisance
from pmh.task_router import (
    TaskRoute,
    explain_task,
    format_task_menu,
    get_task,
    list_tasks,
    route_from_wizard_choice,
    search_applications as search_tasks,
)

__all__ = [
    "SHIFT_TYPES",
    "ShiftTypePlain",
    "explain_application",
    "format_application_finder",
    "format_search_results",
    "format_shift_types",
    "search_applications",
    "ApplicabilityReport",
    "DomainPair",
    "EvaluationReport",
    "HookSuggestion",
    "RobustFitResult",
    "check_applicability",
    "evaluate_baseline_vs_pmh",
    "evaluate_robust_fit",
    "evaluate_trainer_on_loader",
    "robust_fit",
    "robust_fit_text_domains",
    "suggest_hook",
    "DoctorReport",
    "run_doctor",
    "SetupRecommendation",
    "format_setup_guide",
    "preflight_plain_english",
    "print_setup_guide",
    "recommend_setup",
    "run_wizard",
    "SubtypeRecommendation",
    "format_subtype_line",
    "get_subtype",
    "list_subtypes",
    "print_subtype_guide",
    "suggest_subtype",
    "NuisanceSuggestion",
    "suggest_nuisance",
    "TaskRoute",
    "explain_task",
    "format_task_menu",
    "get_task",
    "list_tasks",
    "route_from_wizard_choice",
    "search_tasks",
]
