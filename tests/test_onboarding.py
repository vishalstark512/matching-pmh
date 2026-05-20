"""Developer onboarding helpers."""

from __future__ import annotations

from pmh.onboarding import format_setup_guide, preflight_plain_english, recommend_setup


def test_recommend_pytorch_domain_shift():
    rec = recommend_setup(stack="pytorch", has_target_domain=True, has_target_labels=False)
    assert rec.stack == "pytorch"
    assert rec.lemma == "D4"
    assert rec.nuisance == "domain_shift"
    assert "subtype=" in format_setup_guide(rec)
    assert "PMHTrainer" in rec.snippet


def test_recommend_sklearn():
    rec = recommend_setup(stack="sklearn", has_frozen_features=True)
    assert rec.stack == "sklearn"
    assert "PMHMatcher" in rec.snippet


def test_preflight_plain_english_export():
    assert "weak" in preflight_plain_english("marginal").lower()


def test_format_setup_guide():
    rec = recommend_setup(stack="pytorch")
    text = format_setup_guide(rec)
    assert "Recommended:" in text
    assert rec.example_script in text
