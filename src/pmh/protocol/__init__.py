"""Layer 4 — training protocol (cap, warmup, hybrid compose, control modes)."""

from pmh.config import PMHConfig, PreflightConfig
from pmh.controls import signal_W_projector, wrong_W_projector
from pmh.multi import MultiPMHLoss
from pmh.penalty import cap_pmh_term
from pmh.recipe import default_protocol_config
from pmh.training import PMHLoss

__all__ = [
    "PMHConfig",
    "PreflightConfig",
    "PMHLoss",
    "MultiPMHLoss",
    "cap_pmh_term",
    "wrong_W_projector",
    "signal_W_projector",
    "default_protocol_config",
    "control_modes",
]


def control_modes() -> tuple[str, ...]:
    """PMHLoss ``mode`` values for falsification (paper §7 step 5)."""
    return ("matched", "wrong_w", "isotropic", "trace_iso", "signal_w")
