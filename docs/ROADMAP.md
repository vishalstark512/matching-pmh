# Roadmap

**Current release: v1.5.1** — [When PMH helps](WHEN_PMH_HELPS.md), `evaluate_robust_fit`, block recipe cards, API reference pages, notebook CI, Office-31 download walkthrough.

**v1.5.0** — developer API (`robust_fit`, `check_applicability`), golden paths G1–G3, wizard, Colab notebooks.

**Paper / benchmarks (1.4.x):** D1 fidelity, presets, CORRECT_USAGE — see [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md).

**Onboarding (shipped):** [DEVELOPER_ONBOARDING_PLAN.md](DEVELOPER_ONBOARDING_PLAN.md).

See [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) for task-by-task fidelity vs `Paper2/T1`–`T7`.

| Area | Status |
|------|--------|
| Adoption docs (GETTING_STARTED, CHOOSE, TROUBLESHOOTING) | Shipped |
| D1–D7 `PMHTrainer` / `PMHMatcher` | Shipped |
| Hybrid `MultiPMHLoss` | Shipped |
| Hook adapters + gallery | Shipped |
| vs CORAL guide | Shipped |

## In progress (sklearn-class)

| Item | Status |
|------|--------|
| `check_estimator` on `PMHMatcher` (isotropic path) | Shipped |
| `Pipeline.fit` via `X_target` in `__init__` or metadata routing | Shipped |
| `GridSearchCV` (`make_pmh_pipeline`, `grid_search_pmh_pipeline`) | Shipped |
| TDI + sklearn benchmark table (`pmh.tdi`, `examples/21_*`) | Shipped |
| mkdocstrings API reference + Pages | **Partial** — [developer](api/developer.md), [PMHTrainer](api/pmh-trainer.md) |

## Paper fidelity (priority)

| Item | Priority |
|------|----------|
| Unify D1: labeled cross-domain SVD in `estimate_from_config` | **Done** (v1.4.1+) |
| `PMHLoss` wrong-W orthogonal to matched \(W\) | **Done** (v1.4.1+) |
| Disambiguate “isotropic” (D2 vs arms) in API/docs | **Done** (`trace_iso`, `PAPER_ALIGNMENT.md`) |
| Benchmark presets per block (T2–T7 tuning + arms) | P1 |
| Optional calibrators (PGD, style Gram, gradient-W) | P2 |

## Ideas (backlog)

- Optional timm/HF smoke in CI extras job
- Expand mkdocstrings (compare_arms, estimators)
- sklearn-lite extra (numpy-only Mode B without torch import)
- MLOps `export_deployment()` bundle

See [CHANGELOG](https://github.com/vishalstark512/matching-pmh/blob/main/CHANGELOG.md).
