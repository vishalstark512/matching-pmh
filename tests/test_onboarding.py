"""Onboarding, wizard, and setup recommendations."""

from __future__ import annotations

from pmh.onboarding import (
    format_setup_guide,
    preflight_plain_english,
    recommend_setup,
    run_wizard,
)


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


def test_preflight_plain_english():
    assert "usable" in preflight_plain_english("pass").lower()
    assert "weak" in preflight_plain_english("marginal").lower()
    assert "not reliable" in preflight_plain_english("fail").lower()


def test_format_setup_guide():
    rec = recommend_setup(stack="pytorch")
    text = format_setup_guide(rec)
    assert "Recommended:" in text
    assert rec.example_script in text


def test_run_wizard_non_interactive_pytorch(capsys):
    rec = run_wizard(stack="pytorch", interactive=False)
    assert rec.stack == "pytorch"
    assert rec.nuisance == "domain_shift"
    out = capsys.readouterr().out
    assert "PMHTrainer" in out
    assert "Colab:" in out


def test_run_wizard_non_interactive_sklearn(capsys):
    rec = run_wizard(stack="sklearn", interactive=False)
    assert rec.stack == "sklearn"
    assert "PMHMatcher" in rec.snippet
    assert "t01-classical.ipynb" in capsys.readouterr().out


