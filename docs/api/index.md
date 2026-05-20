# API reference

Lookup after [Five-step recipe](../FIVE_STEP_RECIPE.md) and [Integrate](../INTEGRATE.md).

---

## Tier 0 — `from pmh import …` (stable)

| Symbol | Role |
|--------|------|
| `robust_fit` | Mode A bundle (steps 0–3) |
| `PMHTrainer` | Estimate + train |
| `PMHMatcher` | Mode B adapt |
| `PMHConfig` | Cap / warmup |
| `check_applicability` | Scope gate |
| `evaluate_baseline_vs_pmh` | Mode B eval |
| `evaluate_robust_fit` | Mode A eval |
| `explain_task`, `get_task`, `list_tasks` | Routing |

---

## Meta-layer packages

| Package | Layer |
|---------|--------|
| `pmh.recipe` | Five-step spine |
| `pmh.scope` | Step 0 |
| `pmh.identify` | Step 1 |
| `pmh.apply` | Step 3 (A/B) |
| `pmh.protocol` | Step 4 |
| `pmh.evidence` | Step 5 |

---

## Generated detail (mkdocstrings)

::: pmh.matcher.PMHMatcher
    options:
      members: [__init__, fit, transform]
      show_root_heading: true

::: pmh.trainer.PMHTrainer
    options:
      members: [__init__, fit, estimate]
      show_root_heading: true

::: pmh.developer.robust_fit

::: pmh.recipe.format_five_step_guide

Paper contracts: [CORRECT_USAGE.md](../CORRECT_USAGE.md) (Evidence).
