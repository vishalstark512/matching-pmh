"""pmh-train wizard and onboarding helpers."""

from __future__ import annotations

from pmh.onboarding import preflight_plain_english, run_wizard


def test_preflight_plain_english():
    assert "usable" in preflight_plain_english("pass").lower()
    assert "weak" in preflight_plain_english("marginal").lower()
    assert "not reliable" in preflight_plain_english("fail").lower()


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
    assert "sklearn_frozen_features" in capsys.readouterr().out


def test_cli_wizard_non_interactive(capsys):
    from pmh.cli.main import main

    assert main(["wizard", "--non-interactive", "--stack", "pytorch"]) == 0
    assert "PMHTrainer" in capsys.readouterr().out


def test_cli_wizard_requires_stack_when_non_interactive(capsys):
    from pmh.cli.main import main

    assert main(["wizard", "--non-interactive"]) == 2
