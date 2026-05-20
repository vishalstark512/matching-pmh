# Developer API

High-level helpers for **train on site A → deploy on site B** without paper vocabulary (D1–D7). See [Golden paths](../GOLDEN_PATHS.md) and [When PMH helps](../WHEN_PMH_HELPS.md).

## Applicability and setup

::: pmh.developer.check_applicability

::: pmh.developer.ApplicabilityReport
    options:
      members:
        - summary
        - verdict
        - can_proceed

::: pmh.developer.DomainPair
    options:
      members:
        - from_arrays
        - validate

::: pmh.developer.suggest_hook

::: pmh.developer.HookSuggestion

## Training and evaluation

::: pmh.developer.robust_fit

::: pmh.developer.RobustFitResult
    options:
      members:
        - summary
        - preflight_message
        - trainer
        - stats

::: pmh.developer.evaluate_baseline_vs_pmh

::: pmh.developer.evaluate_robust_fit

::: pmh.developer.evaluate_trainer_on_loader

::: pmh.developer.EvaluationReport
    options:
      members:
        - summary

## HF text corpora

::: pmh.developer.robust_fit_text_domains

## Onboarding (CLI wizard)

::: pmh.onboarding.recommend_setup

::: pmh.onboarding.preflight_plain_english

::: pmh.onboarding.run_wizard
