"""matching-pmh: estimate Sigma_task and apply matched PMH penalties."""

from pmh.artifact import SigmaTaskEstimate
from pmh.config import PMHConfig, PreflightConfig, SigmaTaskConfig
from pmh.controls import signal_W_projector, wrong_W_projector
from pmh.diagnostics import eigengap_ratio
from pmh.estimate import estimate_from_config, estimate_sigma_task
from pmh.features import collect_features, paired_batches
from pmh.penalty import cap_pmh_term, pmh_penalty, pmh_penalty_feature_diff, pmh_penalty_on_rep
from pmh.preflight import PreflightStatus, preflight_eigengap
from pmh.training import PMHLoss

__all__ = [
    "SigmaTaskConfig",
    "PMHConfig",
    "PreflightConfig",
    "SigmaTaskEstimate",
    "estimate_sigma_task",
    "estimate_from_config",
    "collect_features",
    "paired_batches",
    "pmh_penalty",
    "pmh_penalty_on_rep",
    "pmh_penalty_feature_diff",
    "PMHLoss",
    "cap_pmh_term",
    "wrong_W_projector",
    "signal_W_projector",
    "eigengap_ratio",
    "preflight_eigengap",
    "PreflightStatus",
]

__version__ = "0.6.1"


def __getattr__(name: str):
    """Lazy subpackage exports."""
    if name in ("PMHCallback", "train_epoch_with_pmh", "PMHStepResult"):
        from pmh.integrations import PMHCallback, PMHStepResult, train_epoch_with_pmh

        return {"PMHCallback": PMHCallback, "train_epoch_with_pmh": train_epoch_with_pmh, "PMHStepResult": PMHStepResult}[name]
    if name in ("MultiLayerPMHLoss", "gram_sample_noise"):
        from pmh.vision import MultiLayerPMHLoss, gram_sample_noise

        return {"MultiLayerPMHLoss": MultiLayerPMHLoss, "gram_sample_noise": gram_sample_noise}[name]
    if name in ("estimate_sigma_task_numpy", "gram_from_diff_numpy"):
        from pmh import numpy_api

        return getattr(numpy_api, name)
    if name in ("MatchedSubspaceProjector", "project_onto_complement"):
        from pmh import sklearn_match

        return getattr(sklearn_match, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
