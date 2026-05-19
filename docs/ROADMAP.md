# Roadmap

**Current release: v1.4.0** — trajectory TDI, API docs, fixed Office-31 sklearn protocol.

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
| mkdocstrings API reference + Pages | Backlog |

## Paper fidelity (priority)

| Item | Priority |
|------|----------|
| Unify D1: labeled cross-domain SVD in `estimate_from_config` | **Done** (v1.4.1+) |
| `PMHLoss` wrong-W orthogonal to matched \(W\) | **Done** (v1.4.1+) |
| Disambiguate “isotropic” (D2 vs arms) in API/docs | **Done** (`trace_iso`, `PAPER_ALIGNMENT.md`) |
| Benchmark presets per block (T2–T7 tuning + arms) | P1 |
| Optional calibrators (PGD, style Gram, gradient-W) | P2 |

## Ideas (backlog)

- Colab notebook linked from gallery
- Optional timm/HF smoke in CI extras job
- `pmh-train wizard` interactive CLI

See [CHANGELOG](https://github.com/vishalstark512/matching-pmh/blob/main/CHANGELOG.md).
