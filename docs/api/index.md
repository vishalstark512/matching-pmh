# API reference

Start with [Quickstart](../QUICKSTART.md) and [13 tasks](../tasks/index.md). Import from **`pmh`** (flat public API in 2.0).

---

## Tier 0 — `from pmh import …`

| Symbol | Role |
|--------|------|
| `robust_fit` | Mode A bundle (estimate + train) |
| `PMHTrainer` | Estimate + train |
| `PMHMatcher` | Mode B adapt |
| `PMHConfig` | Cap / warmup |
| `check_applicability` | Scope gate |
| `evaluate_baseline_vs_pmh` | Mode B eval + falsification |
| `evaluate_robust_fit` | Mode A eval + falsification |
| `explain_task`, `get_task`, `list_tasks` | Application routing |

Recipe spine: `pmh.recipe` (`plan_recipe`, `control_modes`, `default_protocol_config`).

Benchmark / Step 5 arms: `pmh.benchmark`, `compare_arms_sklearn`.

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
