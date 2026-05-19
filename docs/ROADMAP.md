# Roadmap

**Current release: v1.3.0** — sklearn-class API, TDI benchmarks, GridSearchCV helpers.

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
| mkdocstrings API reference + Pages | Backlog |

## Ideas (backlog)

- Colab notebook linked from gallery
- Optional timm/HF smoke in CI extras job
- `pmh-train wizard` interactive CLI

See [CHANGELOG](https://github.com/vishalstark512/matching-pmh/blob/main/CHANGELOG.md).
