"""matching-pmh: estimate Sigma_task and apply matched PMH penalties."""

from pmh.artifact import SigmaTaskEstimate
from pmh.compare import compare_arms, compare_arms_sklearn
from pmh.config import PMHConfig, PreflightConfig, SigmaTaskConfig
from pmh.controls import signal_W_projector, wrong_W_projector
from pmh.diagnostics import eigengap_ratio
from pmh.tdi import (
    TDIReport,
    directional_drift_numpy,
    geometry_report,
    tdi_cls,
    tdi_feature_isotropic,
    tdi_layout,
    trajectory_tdi_encoder,
    trajectory_tdi_layerwise,
)
from pmh.estimate import estimate_from_config, estimate_sigma_task
from pmh.features import (
    collect_augmentation_deltas,
    collect_features,
    collect_labeled_features,
    collect_sequence_features,
    paired_batches,
)
from pmh.hooks import (
    detect_model_family,
    encoder_gnn_mean_pool,
    encoder_hf_hidden_states,
    encoder_timm,
    encoder_torchvision_resnet,
    list_hook_families,
    register_hook_family,
    resolve_hook,
    validate_representation,
)
from pmh.matcher import PMHMatcher
from pmh.nuisance import config_from_nuisance, list_nuisance_names, resolve_method
from pmh.penalty import cap_pmh_term, pmh_penalty, pmh_penalty_feature_diff, pmh_penalty_on_rep
from pmh.preflight import PreflightStatus, preflight_eigengap
from pmh.developer import (
    ApplicabilityReport,
    DomainPair,
    EvaluationReport,
    HookSuggestion,
    RobustFitResult,
    check_applicability,
    evaluate_baseline_vs_pmh,
    evaluate_robust_fit,
    evaluate_trainer_on_loader,
    robust_fit,
    robust_fit_text_domains,
    suggest_hook,
)
from pmh.onboarding import (
    SetupRecommendation,
    format_setup_guide,
    preflight_plain_english,
    print_setup_guide,
    recommend_setup,
    run_wizard,
)
from pmh.suggest import NuisanceSuggestion, suggest_nuisance
from pmh.applications import (
    SHIFT_TYPES,
    ShiftTypePlain,
    explain_application,
    format_application_finder,
    format_search_results,
    format_shift_types,
    search_applications,
)
from pmh.task_router import (
    TaskRoute,
    explain_task,
    format_task_menu,
    get_task,
    list_tasks,
    search_applications as search_tasks,
)
from pmh.subtypes import (
    SubtypeRecommendation,
    format_subtype_line,
    get_subtype,
    list_subtypes,
    print_subtype_guide,
    suggest_subtype,
)
from pmh.custom import (
    artifact_from_deltas,
    artifact_from_w,
    estimate_custom,
    load_w_numpy,
)
from pmh.benchmark.validate import ValidationReport, validate_falsification
from pmh.benchmark.presets import get_subtype_preset
from pmh.data_adapters import (
    batch_iterators,
    batch_iterators_labeled,
    load_domain_arrays,
    load_domain_dirs,
    resolve_feature_npy,
)
from pmh.deployment import DeploymentBundle, export_deployment, load_deployment_bundle
from pmh.doctor import DoctorReport, run_doctor
from pmh.data_context import DataContext
from pmh.multi import MultiPMHLoss
from pmh.trainer import PMHTrainer, build_hybrid_trainer
from pmh.training import PMHLoss
from pmh.sklearn_pipeline import (
    default_pmh_param_grid,
    grid_search_pmh_pipeline,
    make_pmh_pipeline,
    tune_result_from_grid_search,
)
from pmh.tune import TuneResult, tune_pmh_config, tune_sklearn_matcher

__all__ = [
    "PMHMatcher",
    "PMHTrainer",
    "build_hybrid_trainer",
    "HFPMHTrainer",
    "MultiPMHLoss",
    "DataContext",
    "PMHConfig",
    "PMHLoss",
    "SigmaTaskConfig",
    "SigmaTaskEstimate",
    "PreflightConfig",
    "estimate_sigma_task",
    "estimate_from_config",
    "collect_features",
    "collect_labeled_features",
    "collect_augmentation_deltas",
    "collect_sequence_features",
    "paired_batches",
    "pmh_penalty",
    "pmh_penalty_on_rep",
    "pmh_penalty_feature_diff",
    "cap_pmh_term",
    "wrong_W_projector",
    "signal_W_projector",
    "eigengap_ratio",
    "preflight_eigengap",
    "tdi_cls",
    "tdi_layout",
    "tdi_feature_isotropic",
    "trajectory_tdi_layerwise",
    "trajectory_tdi_encoder",
    "directional_drift_numpy",
    "geometry_report",
    "TDIReport",
    "PreflightStatus",
    "config_from_nuisance",
    "list_nuisance_names",
    "resolve_method",
    "suggest_nuisance",
    "NuisanceSuggestion",
    "suggest_subtype",
    "SubtypeRecommendation",
    "TaskRoute",
    "ShiftTypePlain",
    "SHIFT_TYPES",
    "explain_task",
    "explain_application",
    "format_shift_types",
    "format_application_finder",
    "format_search_results",
    "search_applications",
    "search_tasks",
    "format_task_menu",
    "get_task",
    "list_tasks",
    "list_subtypes",
    "get_subtype",
    "format_subtype_line",
    "print_subtype_guide",
    "estimate_custom",
    "artifact_from_deltas",
    "artifact_from_w",
    "load_w_numpy",
    "load_domain_arrays",
    "load_domain_dirs",
    "resolve_feature_npy",
    "batch_iterators",
    "batch_iterators_labeled",
    "validate_falsification",
    "ValidationReport",
    "get_subtype_preset",
    "export_deployment",
    "load_deployment_bundle",
    "DeploymentBundle",
    "run_doctor",
    "DoctorReport",
    "recommend_setup",
    "print_setup_guide",
    "format_setup_guide",
    "SetupRecommendation",
    "run_wizard",
    "preflight_plain_english",
    "check_applicability",
    "ApplicabilityReport",
    "DomainPair",
    "robust_fit",
    "robust_fit_text_domains",
    "RobustFitResult",
    "suggest_hook",
    "HookSuggestion",
    "evaluate_baseline_vs_pmh",
    "evaluate_robust_fit",
    "evaluate_trainer_on_loader",
    "EvaluationReport",
    "resolve_hook",
    "validate_representation",
    "detect_model_family",
    "list_hook_families",
    "register_hook_family",
    "encoder_timm",
    "encoder_torchvision_resnet",
    "encoder_hf_hidden_states",
    "encoder_gnn_mean_pool",
    "compare_arms",
    "compare_arms_sklearn",
    "tune_sklearn_matcher",
    "tune_pmh_config",
    "TuneResult",
    "make_pmh_pipeline",
    "default_pmh_param_grid",
    "grid_search_pmh_pipeline",
    "tune_result_from_grid_search",
]

__version__ = "1.5.3"


def __getattr__(name: str):
    """Lazy subpackage exports."""
    if name == "HFPMHTrainer":
        from pmh.hf_trainer import HFPMHTrainer

        return HFPMHTrainer
    if name == "get_pmh_trainer":
        from pmh.integrations.hf_trainer import get_pmh_trainer

        return get_pmh_trainer
    if name == "lightning_available":
        from pmh.integrations.lightning import lightning_available

        return lightning_available
    if name in ("PMHCallback", "train_epoch_with_pmh", "PMHStepResult"):
        from pmh.integrations import PMHCallback, PMHStepResult, train_epoch_with_pmh

        return {"PMHCallback": PMHCallback, "train_epoch_with_pmh": train_epoch_with_pmh, "PMHStepResult": PMHStepResult}[name]
    if name in ("MultiLayerPMHLoss", "gram_sample_noise"):
        from pmh.vision import MultiLayerPMHLoss, gram_sample_noise

        return {"MultiLayerPMHLoss": MultiLayerPMHLoss, "gram_sample_noise": gram_sample_noise}[name]
    if name in (
        "run_benchmark_protocol",
        "train_one_arm",
        "write_benchmark_report",
        "benchmark_to_markdown",
        "run_sklearn_benchmark",
        "STANDARD_ARMS",
    ):
        from pmh import benchmark as _b

        return getattr(_b, name)
    if name in ("estimate_sigma_task_numpy", "gram_from_diff_numpy"):
        from pmh import numpy_api

        return getattr(numpy_api, name)
    if name in ("MatchedSubspaceProjector", "project_onto_complement"):
        from pmh import sklearn_match

        return getattr(sklearn_match, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
