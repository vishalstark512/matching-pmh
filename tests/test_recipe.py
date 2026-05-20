"""Five-step recipe API."""

from __future__ import annotations

import numpy as np
import pytest

from pmh.recipe import (
    RecipeStep,
    assumption_id,
    format_five_step_guide,
    plan_recipe,
    recommended_application_mode,
    step_identify,
    step_scope,
)


def test_assumption_id():
    assert assumption_id("D4") == "A4"
    assert assumption_id("4") == "A4"


def test_step_identify_domain_shift():
    shift = step_identify(
        has_target_domain=True,
        has_source_labels=True,
        has_target_labels=False,
    )
    assert shift.method in ("D1", "D4")
    assert shift.nuisance
    assert shift.assumption.startswith("A")


def test_recommended_mode_sklearn():
    shift = step_identify(has_target_domain=True, has_source_labels=True)
    assert recommended_application_mode(shift, stack="sklearn") == "projection"
    assert recommended_application_mode(shift, stack="pytorch") in ("jacobian", "projection")


def test_plan_recipe():
    plan = plan_recipe(stack="pytorch", n_source=100, n_target=80)
    assert plan.scope.can_proceed
    assert RecipeStep.EVIDENCE in plan.steps
    assert plan.recommended_mode == "jacobian"


def test_format_five_step_guide():
    text = format_five_step_guide(stack="sklearn")
    assert "Step 1" in text
    assert "Mode A" in text
    assert "pmh.evidence" in text


def test_step_estimate_arrays_smoke():
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((40, 8)).astype(np.float32)
    xt = xs + 0.5
    from pmh.recipe import step_estimate_arrays

    result = step_estimate_arrays(xs, xt, nuisance="domain_shift", rank=4)
    assert result.artifact.sigma.shape == (8, 8)
    assert result.eigengap > 0


def test_cli_recipe(capsys):
    from pmh.cli.main import main

    assert main(["recipe", "--stack", "pytorch"]) == 0
    out = capsys.readouterr().out
    assert "Step 2" in out
