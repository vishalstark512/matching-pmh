"""Estimate → artifact → penalty (Phase A / B contract)."""

from pmh.artifact import SigmaTaskEstimate
from pmh.catalog import MethodSpec, METHODS
from pmh.config import PMHConfig, PreflightConfig, SigmaTaskConfig
from pmh.controls import signal_W_projector, wrong_W_projector
from pmh.diagnostics import eigengap_ratio
from pmh.estimate import estimate_from_config, estimate_sigma_task
from pmh.nuisance import config_from_nuisance, list_nuisance_names, resolve_method
from pmh.penalty import cap_pmh_term, pmh_penalty, pmh_penalty_feature_diff, pmh_penalty_on_rep
from pmh.preflight import PreflightStatus, preflight_eigengap

__all__ = [
    "SigmaTaskEstimate",
    "MethodSpec",
    "METHODS",
    "PMHConfig",
    "PreflightConfig",
    "SigmaTaskConfig",
    "signal_W_projector",
    "wrong_W_projector",
    "eigengap_ratio",
    "estimate_from_config",
    "estimate_sigma_task",
    "config_from_nuisance",
    "list_nuisance_names",
    "resolve_method",
    "cap_pmh_term",
    "pmh_penalty",
    "pmh_penalty_feature_diff",
    "pmh_penalty_on_rep",
    "PreflightStatus",
    "preflight_eigengap",
]
