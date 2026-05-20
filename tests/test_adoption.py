"""Product adoption copy, Step 5 helpers, and plain-language shift types."""

from __future__ import annotations

from pmh.adoption import (
    RECIPE_ONE_LINER,
    falsification_step5_ok,
    format_falsification_block,
    format_recipe_banner,
    format_newbie_checklist,
    newbie_checklist,
)
from pmh.developer import EvaluationReport


def test_falsification_step5_ok():
    assert falsification_step5_ok({"matched": 0.8, "wrong_w": 0.5, "isotropic": 0.55}) is True
    assert falsification_step5_ok({"matched": 0.5, "wrong_w": 0.8, "isotropic": 0.4}) is False
    assert falsification_step5_ok({"matched": 0.8}) is None


def test_evaluation_report_summary_includes_step5():
    rep = EvaluationReport(
        baseline_metric=0.6,
        pmh_metric=0.7,
        falsification_arms={"matched": 0.75, "wrong_w": 0.5, "isotropic": 0.52},
    )
    text = rep.summary()
    assert RECIPE_ONE_LINER in text
    assert "matched" in text
    assert rep.step5_ok() is True


def test_format_recipe_banner():
    assert "deploy holdout" in format_recipe_banner().lower()
    assert format_falsification_block({"matched": 0.9, "wrong_w": 0.4, "isotropic": 0.45})


def test_newbie_checklist():
    assert "Step 5" in " ".join(newbie_checklist("sklearn"))
    assert "evaluate_robust_fit" in format_newbie_checklist("pytorch")


def test_explain_nuisance_key_domain_shift():
    from pmh import explain_nuisance_key

    text = explain_nuisance_key("domain_shift")
    assert "domain_shift" in text
    assert "D4" in text
    assert any(w in text.lower() for w in ("site", "camera", "hospital"))


def test_format_shift_types_includes_isotropic():
    from pmh import format_shift_types

    assert "isotropic" in format_shift_types()


def test_format_shift_types_mentions_deploy():
    from pmh import format_shift_types

    assert "deploy shift" in format_shift_types().lower()
