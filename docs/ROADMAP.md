# Roadmap

**Current release: v1.2.0** — adoption docs + full D1–D7 trainer paths + hybrid nuisances.

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
| `GridSearchCV` integration | Backlog |
| mkdocstrings API reference + Pages | Backlog |

## Ideas (backlog)

- Colab notebook linked from gallery
- Optional timm/HF smoke in CI extras job
- `pmh-train wizard` interactive CLI

See [CHANGELOG](https://github.com/vishalstark512/matching-pmh/blob/main/CHANGELOG.md).
