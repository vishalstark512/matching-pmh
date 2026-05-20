"""Product copy and Step 5 (falsification) helpers — one voice for CLI, API, and docs."""

from __future__ import annotations

PRODUCT_TAGLINE = "Train on site A. Deploy on site B. Same labels."

# Perturbation Matching Hypothesis — five-step product spine (see main.pdf §7).
RECIPE_ONE_LINER = (
    "Scope -> estimate Sigma_task -> matched PMH on h -> "
    "Step 5 evidence (matched > wrong-W > isotropic on deploy holdout)."
)

STEP5_HEADING = "Step 5 (deploy holdout — falsification arms)"

STEP5_PASS = (
    "Falsification pattern holds: matched beats wrong-W and isotropic on deploy holdout."
)

STEP5_FAIL = (
    "Matched did not beat both controls on deploy holdout — "
    "see docs/WHEN_PMH_HELPS.md before claiming gains."
)

STEP5_PYTORCH_HINT = (
    "Step 5: evaluate_robust_fit(..., include_falsification=True) or compare_arms on deploy holdout."
)

NEWBIE_CHECKLIST_PYTORCH = [
    "0. Scope: same label semantics on site A (train) and site B (deploy)",
    "1. Data: train_loader + source_batches + target_batches (target labels optional for estimate)",
    "2. Train: robust_fit(..., hook='auto') or PMHTrainer.fit — read artifact preflight",
    "3. Step 5: evaluate_robust_fit(..., val_loader=deploy_holdout, include_falsification=True)",
    "4. Ship only if matched > wrong-W and matched > isotropic on that holdout",
]

NEWBIE_CHECKLIST_SKLEARN = [
    "0. Scope: frozen embeddings, same classes on source and target",
    "1. Hold out target rows for test — never pass them to PMHMatcher.fit",
    "2. Step 5: evaluate_baseline_vs_pmh(...) — falsification arms on by default",
    "3. Real data: python scripts/demos/office31_sklearn.py --office31-root PATH",
    "4. Demo arrays: from pmh import load_g2_demo_arrays",
]

NEWBIE_CHECKLIST_HF = [
    "0. Scope: same factual labels across two text corpora or style pairs",
    "1. Estimate: HFPMHTrainer.estimate or robust_fit_text_domains",
    "2. Step 5: task metric on deploy holdout + geometry report (see falsification walkthrough)",
]

FALSIFICATION_ARM_ORDER = ("b0", "matched", "wrong_w", "isotropic", "coral", "signal_w")

ARM_PLAIN_NAMES: dict[str, str] = {
    "b0": "ERM baseline (no PMH)",
    "matched": "shift-matched PMH",
    "wrong_w": "wrong-direction control",
    "isotropic": "generic isotropic control",
    "coral": "CORAL (linear adapt)",
    "signal_w": "signal-aligned control",
}

SHIP_VERDICT_SHIP = "PASS — matched beats both controls on deploy holdout (Step 5)."
SHIP_VERDICT_NO_SHIP = "FAIL — matched did not beat both controls on deploy holdout."
SHIP_VERDICT_INCONCLUSIVE = "INCONCLUSIVE — run Step 5 controls (need matched, wrong, isotropic)."


def falsification_step5_ok(arms: dict[str, float]) -> bool | None:
    """True when matched > wrong-W and matched > isotropic on the same holdout."""
    matched = arms.get("matched")
    wrong_w = arms.get("wrong_w")
    isotropic = arms.get("isotropic")
    if matched is None or wrong_w is None or isotropic is None:
        return None
    return matched > wrong_w and matched > isotropic


def format_recipe_banner(*, trailing: str = "") -> str:
    """Two-line banner for CLI, wizard, and reports."""
    lines = [PRODUCT_TAGLINE, RECIPE_ONE_LINER]
    if trailing:
        lines.append(trailing)
    return "\n".join(lines)


def newbie_checklist(stack: str = "pytorch") -> list[str]:
    """Plain pipeline steps for pmh-train doctor and onboarding."""
    if stack == "sklearn":
        return list(NEWBIE_CHECKLIST_SKLEARN)
    if stack == "hf":
        return list(NEWBIE_CHECKLIST_HF)
    return list(NEWBIE_CHECKLIST_PYTORCH)


def format_newbie_checklist(stack: str = "pytorch") -> str:
    lines = ["Your pipeline (copy this order):", *newbie_checklist(stack)]
    return "\n".join(lines)


def format_falsification_block(arms: dict[str, float], *, metric_name: str = "accuracy") -> list[str]:
    """Lines for EvaluationReport.summary() Step 5 section."""
    if not arms:
        return []
    lines = [STEP5_HEADING + f" ({metric_name}):"]
    for arm in FALSIFICATION_ARM_ORDER:
        if arm in arms:
            label = ARM_PLAIN_NAMES.get(arm, arm)
            lines.append(f"  {label}: {arms[arm]:.3f}")
    ok = falsification_step5_ok(arms)
    if ok is True:
        lines.append(f"  -> {SHIP_VERDICT_SHIP}")
    elif ok is False:
        lines.append(f"  -> {SHIP_VERDICT_NO_SHIP}")
    return lines


def ship_verdict_label(arms: dict[str, float]) -> str:
    """One-line ship / don't ship from falsification arms."""
    ok = falsification_step5_ok(arms)
    if ok is True:
        return SHIP_VERDICT_SHIP
    if ok is False:
        return SHIP_VERDICT_NO_SHIP
    return SHIP_VERDICT_INCONCLUSIVE
