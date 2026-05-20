"""Five-step matching-principle recipe (paper §7, Fig. 4).

This module is the **product spine**: scope → identify A_k → estimate Σ̂_task → apply PMH → evidence.
Paper blocks T1–T7 are worked examples; use ``pmh.benchmark`` and ``docs/tasks/`` for replication.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np
import torch

from pmh.artifact import SigmaTaskEstimate
from pmh.config import PMHConfig, PreflightConfig, SigmaTaskConfig
from pmh.developer import ApplicabilityReport, DomainPair, check_applicability
from pmh.estimate import estimate_from_config
from pmh.preflight import PreflightStatus, preflight_eigengap
from pmh.subtypes import SubtypeRecommendation, get_subtype, suggest_subtype

ApplicationMode = Literal["jacobian", "projection"]

_ASSUMPTION_BY_METHOD = {
    "D1": "A1",
    "D2": "A2",
    "D3": "A3",
    "D4": "A4",
    "D5": "A5",
    "D6": "A6",
    "D7": "A7",
}


class RecipeStep(str, Enum):
    """Paper §7 numbered steps."""

    SCOPE = "scope"
    IDENTIFY = "identify"
    ESTIMATE = "estimate"
    APPLY = "apply"
    PROTOCOL = "protocol"
    EVIDENCE = "evidence"


@dataclass(frozen=True)
class ShiftIdentification:
    """Step 1 — nuisance family A_k and library mapping."""

    assumption: str
    method: str
    nuisance: str
    title: str
    reason: str
    application_mode: ApplicationMode | Literal["either"]
    doc_anchor: str
    calibrator_note: str = ""

    @classmethod
    def from_subtype(cls, rec: SubtypeRecommendation) -> ShiftIdentification:
        info = get_subtype(rec.method)
        mode = info.default_mode
        if mode == "either":
            mode_out: ApplicationMode | Literal["either"] = "either"
        else:
            mode_out = mode  # type: ignore[assignment]
        return cls(
            assumption=_ASSUMPTION_BY_METHOD.get(rec.method, rec.method),
            method=rec.method,
            nuisance=rec.nuisance,
            title=rec.title,
            reason=rec.reason,
            application_mode=mode_out,
            doc_anchor=rec.doc_anchor,
            calibrator_note=rec.calibrator_note,
        )


@dataclass
class EstimateResult:
    """Step 2 — Σ̂_task artifact + preflight."""

    artifact: SigmaTaskEstimate
    preflight_status: PreflightStatus
    eigengap: float
    preflight_message: str


@dataclass
class RecipePlan:
    """Scope + identification plan before estimate/train."""

    scope: ApplicabilityReport
    shift: ShiftIdentification
    recommended_mode: ApplicationMode
    steps: tuple[RecipeStep, ...]


def assumption_id(method: str) -> str:
    """Map lemma Dk → assumption Ak (e.g. ``'D4'`` → ``'A4'``)."""
    key = method.strip().upper()
    if not key.startswith("D"):
        key = f"D{key}"
    return _ASSUMPTION_BY_METHOD.get(key, key)


def recommended_application_mode(
    shift: ShiftIdentification,
    *,
    stack: Literal["pytorch", "sklearn", "hf"] | None = None,
) -> ApplicationMode:
    """Resolve Mode A (Jacobian) vs Mode B (projection)."""
    if stack == "sklearn":
        return "projection"
    if stack in ("pytorch", "hf"):
        return "jacobian"
    if shift.application_mode in ("jacobian", "projection"):
        return shift.application_mode
    return "jacobian"


def step_scope(
    *,
    stack: Literal["pytorch", "sklearn", "hf"] = "pytorch",
    n_source: int | None = None,
    n_target: int | None = None,
    has_target_domain: bool = True,
    has_target_labels: bool = False,
    has_style_pairs: bool = False,
    new_classes_at_deploy: bool = False,
) -> ApplicabilityReport:
    """Step 0 — is Σ_task defined for this deploy story?"""
    return check_applicability(
        stack=stack,
        n_source=n_source,
        n_target=n_target,
        has_target_domain=has_target_domain,
        has_target_labels=has_target_labels,
        has_style_pairs=has_style_pairs,
        new_classes_at_deploy=new_classes_at_deploy,
    )


def step_identify(**flags: bool) -> ShiftIdentification:
    """Step 1 — pick nuisance family (Table 4 → A_k → D_k)."""
    return ShiftIdentification.from_subtype(suggest_subtype(**flags))


def step_estimate(
    config: SigmaTaskConfig,
    *estimate_args: Any,
    preflight: PreflightConfig | None = None,
    **estimate_kwargs: Any,
) -> EstimateResult:
    """Step 2 — estimate Σ̂_task and run eigengap preflight."""
    artifact = estimate_from_config(config, *estimate_args, **estimate_kwargs)
    pf_cfg = preflight or PreflightConfig()
    rank = int(config.rank or 16)
    status, gamma = preflight_eigengap(artifact.sigma, rank, config=pf_cfg)
    msg = _preflight_plain(status, gamma)
    artifact.preflight = status.value
    artifact.eigengap = gamma
    return EstimateResult(
        artifact=artifact,
        preflight_status=status,
        eigengap=gamma,
        preflight_message=msg,
    )


def step_estimate_arrays(
    x_source: np.ndarray,
    x_target: np.ndarray,
    nuisance: str,
    *,
    y_source: np.ndarray | None = None,
    y_target: np.ndarray | None = None,
    rank: int = 16,
    **kwargs: Any,
) -> EstimateResult:
    """Step 2 shortcut for frozen features (builds config from nuisance key)."""
    from pmh.nuisance import config_from_nuisance

    pair = DomainPair.from_arrays(x_source, x_target, y_source, y_target)
    cfg = config_from_nuisance(nuisance, rank=rank, **kwargs)
    xs = torch.as_tensor(x_source, dtype=torch.float32)
    xt = torch.as_tensor(x_target, dtype=torch.float32)
    scope = step_scope(
        stack="sklearn",
        n_source=pair.n_source,
        n_target=pair.n_target,
    )
    if not scope.can_proceed:
        raise ValueError(scope.summary())
    if cfg.method == "D1" and y_source is not None and y_target is not None:
        return step_estimate(
            cfg,
            xs,
            torch.as_tensor(y_source),
            xt,
            torch.as_tensor(y_target),
        )
    return step_estimate(cfg, xs, xt)


def default_protocol_config(*, preset: str = "balanced") -> PMHConfig:
    """Step 4 — capped PMH (Prop. 3.5)."""
    factories = {
        "balanced": PMHConfig.balanced,
        "conservative": PMHConfig.conservative,
        "aggressive": PMHConfig.aggressive,
    }
    fn = factories.get(preset, PMHConfig.balanced)
    return fn()


def control_modes() -> tuple[str, ...]:
    """PMHLoss ``mode`` values for falsification (Step 5)."""
    return ("matched", "wrong_w", "isotropic", "trace_iso", "signal_w")


def plan_recipe(
    *,
    stack: Literal["pytorch", "sklearn", "hf"] = "pytorch",
    n_source: int | None = None,
    n_target: int | None = None,
    identify_flags: dict[str, bool] | None = None,
) -> RecipePlan:
    """Scope + identify without estimating (CLI / onboarding)."""
    scope = step_scope(stack=stack, n_source=n_source, n_target=n_target)
    flags = identify_flags or {
        "has_source_labels": True,
        "has_target_labels": False,
        "has_target_domain": True,
    }
    shift = step_identify(**flags)
    mode = recommended_application_mode(shift, stack=stack)
    steps = (
        RecipeStep.SCOPE,
        RecipeStep.IDENTIFY,
        RecipeStep.ESTIMATE,
        RecipeStep.APPLY,
        RecipeStep.PROTOCOL,
        RecipeStep.EVIDENCE,
    )
    return RecipePlan(scope=scope, shift=shift, recommended_mode=mode, steps=steps)


def format_five_step_guide(
    *,
    task_id: str | None = None,
    stack: Literal["pytorch", "sklearn", "hf"] | None = None,
) -> str:
    """Human-readable §7 recipe (docs / ``format_five_step_guide``)."""
    from pmh.task_router import explain_task

    from pmh.adoption import RECIPE_ONE_LINER, format_recipe_banner

    lines = [
        format_recipe_banner(),
        "",
        "Matching-principle recipe (5 steps + scope)",
        "=" * 50,
        "",
        RECIPE_ONE_LINER,
        "",
        "Step 0 — Scope",
        "  Label-preserving deploy shift? Same class semantics on A and B?",
        "  → check_applicability()",
        "",
        "Step 1 — Identify A_k (nuisance family)",
        "  Symptom → assumption A1–A7 → lemma D1–D7 → nuisance key",
        "  → suggest_subtype() / suggest_nuisance()",
        "",
        "Step 2 — Estimate Σ̂_task",
        "  One artifact type: SigmaTaskEstimate (+ eigengap preflight)",
        "  → estimate_from_config() / PMHTrainer.estimate() / pmh.recipe.step_estimate()",
        "",
        "Step 3 — Apply matched PMH",
        "  Mode A (Jacobian): PMHLoss, PMHTrainer, robust_fit — deep hook h",
        "  Mode B (projection): PMHMatcher — frozen features / sklearn",
        "  Mode A: PMHTrainer / robust_fit · Mode B: PMHMatcher (see docs/tasks/)",
        "",
        "Step 4 — Protocol",
        "  Cap PMH vs task loss (PMHConfig.cap_ratio); recipe.control_modes(); hybrid compose",
        "",
        "Step 5 — Evidence (required before you trust PMH)",
        "  On deploy holdout: matched > wrong-W and matched > isotropic",
        "  sklearn: evaluate_baseline_vs_pmh(..., include_falsification=True) — default",
        "  PyTorch: evaluate_robust_fit(..., include_falsification=True)",
        "  → compare_arms_sklearn / compare_arms / pmh.benchmark",
        "",
        "Paper blocks T1–T7 = docs/tasks/ + notebooks/tasks/, not a separate API layer.",
        "Task index: docs/tasks/index.md",
    ]
    if task_id:
        lines.extend(["", f"Task profile: {task_id}", "-" * 40, explain_task(task_id)])
    if stack:
        plan = plan_recipe(stack=stack)
        lines.extend(
            [
                "",
                f"Suggested for stack={stack!r}:",
                f"  nuisance={plan.shift.nuisance!r} ({plan.shift.assumption}/{plan.shift.method})",
                f"  mode={plan.recommended_mode!r}",
            ]
        )
    return "\n".join(lines)


def _preflight_plain(status: PreflightStatus, gamma: float) -> str:
    if status == PreflightStatus.PASS:
        return f"preflight PASS (eigengap γ_r={gamma:.3f}) — identification looks stable"
    if status == PreflightStatus.MARGINAL:
        return (
            f"preflight MARGINAL (γ_r={gamma:.3f}) — Office-31-style risk: matched PMH may lose to "
            "CORAL/second-moment baselines; report controls and do not over-claim"
        )
    return (
        f"preflight FAIL (γ_r={gamma:.3f}) — subspace unreliable; try isotropic (A2) or more "
        "deploy data before matched training"
    )
