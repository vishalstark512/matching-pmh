# Changelog

## Unreleased

## 1.6.1 (2026-05-20)

### Documentation

- Reframed the public docs around task-first production use cases: segmentation, pose, detection, speech, sensors, embeddings, LLM/text style, molecules, code, and adversarial-style robustness.
- Added [TASK_PATTERNS.md](docs/TASK_PATTERNS.md) to map the 13 paper tasks into reusable patterns for other datasets and architectures.
- Expanded [GOLDEN_PATHS.md](docs/GOLDEN_PATHS.md) into end-to-end runnable pipelines with demo, data replacement, training/scoring, and production-like evaluation steps.
- Simplified the main docs navigation and walkthrough index so paper/theory material is secondary to adoption.

## 1.6.0 (2026-05-20)

### Product spine (adoption-first)

- **Five-step recipe** — [FIVE_STEP_RECIPE.md](docs/FIVE_STEP_RECIPE.md), `pmh.recipe`, `pmh-train recipe`
- **Tier 0 API** — `robust_fit`, `evaluate_baseline_vs_pmh`, `evaluate_robust_fit`, `explain_task`, `explain_nuisance_key`, `format_shift_types` (`pmh._api` / `pmh.__all__`)
- **Plain language** — [WHAT_IS_DEPLOYMENT_SHIFT.md](docs/WHAT_IS_DEPLOYMENT_SHIFT.md), `pmh-train shifts` (deployment shift vs `nuisance=` API key)
- **Step 5 by default** — falsification arms on deploy holdout in `evaluate_*`; `include_falsification` on PyTorch path
- **CLI** — `pmh-train evaluate` (sklearn + `--stack pytorch`), `pmh-train doctor` checklist, `--artifact` preflight
- **G2 demo** — `load_g2_demo_arrays()`, `examples/02_g2_office31_style_demo.py`
- **Docs diet** — ~43 user-facing pages; redirects in `mkdocs.yml` only; Adopt / Reference / Evidence nav
- **Subpackages** — `pmh.scope`, `identify`, `apply`, `protocol`, `evidence`, `guide` (flat imports unchanged)

### Parameters & integration

- [PMH_PARAMETERS.md](docs/PMH_PARAMETERS.md), [PARAMETERS_CHEATSHEET.md](docs/PARAMETERS_CHEATSHEET.md)
- `PMHTrainer` default `PMHConfig.balanced()`; `pmh-train evaluate --pmh-preset`, `--weight`, `--cap-ratio`
- Starter templates + Colab notebooks updated (cheat sheet cells)

### Tests

- 217 tests; consolidated adoption/CLI tests; `test_public_api` Tier 0 contract

## 1.5.3 (2026-05-19)

### Documentation (adoption-first)

- **[APPLICATIONS.md](docs/APPLICATIONS.md)** — decision tree, finder table, full 7-step walkthroughs + snippets per application
- **[START_HERE.md](docs/START_HERE.md)** — gates + pointer to APPLICATIONS
- **[MAP.md](docs/MAP.md)** — adoption ladder + what **not** to read first
- **`format_shift_types()`** / **`explain_task()`** — terminal output: WHAT CHANGES + WALKTHROUGH
- **`pmh-train route --search KEYWORD`** — find apps by hospital, pose, blur, temporal, …
- **New application profiles:** augmentation (D3), temporal (D6), PyTorch Lightning (G1b)
- **mkdocs nav** — **Adopt** tab (5 pages); Integrate / Research / Reference separated
- **Stable anchors** — `GOLDEN_PATHS.md#g1` … `#g4` (fixes broken deep links)
- Redirect stubs → START_HERE; walkthrough index marked “not for first adoption”
- **GETTING_STARTED** — afternoon checklist (not a second onboarding path)

### Developer onboarding (CLI)

- **`pmh-train route`** / `explain_task()` / `list_tasks()` — task → golden path
- **`pmh-train wizard`** — task menu first
- Adoption banners on training, hooks, theory, integrations, WHEN_PMH_HELPS

## 1.5.2 (2026-05-19)

### Developer adaptability

- **Subtype product** — [NUISANCE_SUBTYPES.md](docs/NUISANCE_SUBTYPES.md), `suggest_subtype`, wizard subtype menu, [FIDELITY_BY_SUBTYPE.md](docs/FIDELITY_BY_SUBTYPE.md), `tests/test_subtype_fidelity.py`
- **Golden paths** — G1–G4; **G1b** (Lightning), **G3b** (HF `Trainer`); templates `lightning_g1b_minimal.py`, `hf_trainer_g3b_minimal.py`
- **Custom geometry** — `estimate_custom`, `artifact_from_deltas`, `PMHTrainer.from_artifact`, [CUSTOM_GEOMETRY.md](docs/CUSTOM_GEOMETRY.md)
- **Data** — `load_domain_dirs`, `pmh-train estimate --source-dir/--target-npy`, [DATA_LAYOUT.md](docs/DATA_LAYOUT.md)
- **CI / ops** — `pmh-train validate` (sklearn + `pytorch_smoke` protocol), `pmh-train doctor`, `export_deployment` / `load_deployment_bundle`
- **Presets** — `get_subtype_preset("D4")` → block preset; [examples/by_subtype/](examples/by_subtype/README.md)
- **Docs** — [index.md](docs/index.md) hub, slim mkdocs nav, redirect stubs, [DOCS_GUIDE.md](docs/DOCS_GUIDE.md), [PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md) subtype table
- **API docs** — [api/custom.md](docs/api/custom.md), [api/subtypes.md](docs/api/subtypes.md), [api/deployment.md](docs/api/deployment.md)

## 1.5.1 (2026-05-19)

### Documentation

- **[WHEN_PMH_HELPS.md](docs/WHEN_PMH_HELPS.md)** — honest expectations, Office-31 reference table, PMH vs CORAL/ERM, success checklist
- PyPI `Documentation` URL → GitHub Pages

### Developer API

- **`evaluate_robust_fit`** — PyTorch ERM vs PMH on a labeled target loader; same `EvaluationReport` as sklearn

### Research docs

- **Block recipe cards** — T1, T2A, T4, T7A under [docs/recipes/](docs/recipes/)
- **[Walkthrough 19](docs/walkthroughs/19-office31-real-data.md)** + `scripts/download_office31.py` (data stays outside git)
- **API reference** — [api/developer.md](docs/api/developer.md), [api/pmh-trainer.md](docs/api/pmh-trainer.md)

### CI

- **Notebook smoke** — `tests/test_notebooks_smoke.py` (nbconvert, `PMH_QUICK=1`)

## 1.5.0 (2026-05-19)

Developer-first release: plain-language docs, golden paths (G1–G3), high-level API, and interactive setup — without changing core estimator semantics from 1.4.1.

### Developer API

- **`check_applicability`**, **`robust_fit`**, **`suggest_hook`**, **`DomainPair`**
- **`evaluate_baseline_vs_pmh`** (sklearn target holdout; optional CORAL baseline)
- **`robust_fit_text_domains`** (HF two-corpora path)
- Export from `import pmh`; see [GOLDEN_PATHS.md](docs/GOLDEN_PATHS.md)

### Onboarding & docs

- [WHAT_IS_PMH.md](docs/WHAT_IS_PMH.md), [FIRST_HOUR.md](docs/FIRST_HOUR.md), [COLAB.md](docs/COLAB.md), [DEMO_OUTPUT.md](docs/DEMO_OUTPUT.md)
- [DEVELOPER_ONBOARDING_PLAN.md](docs/DEVELOPER_ONBOARDING_PLAN.md)
- README / mkdocs: developer links first; Research tab for paper benchmarks
- Gallery “You have → You do” intros; troubleshooting glossary + error snippet table

### CLI & tooling

- **`pmh-train wizard`** and `pmh.onboarding.run_wizard`
- **`pmh-train list-presets`** (from 1.4.x, documented in onboarding flow)
- **`preflight_plain_english()`** for human-readable diagnostics

### Examples & notebooks

- `examples/00_first_run_domain_shift.py`, `examples/22_developer_api_demo.py`
- Colab: PyTorch, sklearn frozen features, HF two corpora
- Starter template: `templates/matching-pmh-starter/`
- GitHub issue template: `minimal_repro.yml`

### Packaging

- PyPI short description and keywords aimed at domain-shift developers
- CI main job installs `[dev,sklearn]` so benchmark tests run on every PR

---

## 1.4.1

### Paper alignment (P0)

- **D1:** `estimate_from_config` / `estimate_d1` now require **labeled** `(x_src, y_src, x_tgt, y_tgt)` with class-mean shifts (T1 protocol). Unlabeled Gram → `method="D4"` or `estimate_d1_gram_unlabeled`. Artifact stores `metadata["w"]`.
- **`PMHLoss` wrong-W:** random subspace **orthogonal to matched** \(W\) (Lemma C); `wrong_seed` for reproducibility.
- **Training isotropic arm:** documented as `trace_iso` (alias `isotropic`); not the same as D2 nuisance or sklearn D4 control.
- **Docs:** [CORRECT_USAGE.md](docs/CORRECT_USAGE.md), [PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md), updated [d1.md](docs/estimators/d1.md).

### Paper alignment (P1)

- **`pmh.benchmark.presets`:** block presets (`t1_office31_sklearn`, `t4_domain_d4`, …) for `compare_arms` / `compare_arms_sklearn`.
- **Multi-seed sklearn:** `seeds=[...]` and `run_sklearn_benchmark_multi_seed`.
- **PyTorch geometry:** `compare_arms(..., include_geometry=True)` with TDI / \(D_N/D_S\) on val embeddings.
- **`pmh.calibrate`:** `subspace_artifact_from_deltas`, `style_gram_from_deltas`, `gradient_subspace_numpy`, `content_residual_subspace`.

## 1.4.0

### Fix: sklearn Office-31 benchmark protocol

- **`run_sklearn_benchmark`**: default **T1 paper protocol** (target pool for W, held-out test, no leakage).
- **D1**: class-mean shifts included in cross-domain SVD (was missing).
- **wrong-W**: orthogonalized against matched W (Lemma C).
- **isotropic arm**: D4 domain Gram directions (was target PCA, which destroyed features).
- Regenerate `docs/benchmarks/office31_amazon_to_dslr.md` after pull.

### Trajectory TDI (paper T2A)

- **`trajectory_tdi_layerwise`**, **`trajectory_tdi_encoder`**: isotropic input noise @ `sigma=0.01` (default).
- **`PMHTrainer.measure_trajectory_tdi`**: probe batches on the training hook encoder.
- **`TDIReport`**: optional `trajectory_tdi`, `tdi_per_layer`.

### Docs and benchmarks

- **mkdocstrings** API reference (`docs/api/index.md`) on GitHub Pages.
- **`scripts/generate_reference_benchmark.py`** → `docs/benchmarks/*.md` (metrics only; [DATA_POLICY](docs/DATA_POLICY.md)).
- [BENCHMARKS.md](docs/BENCHMARKS.md) updated for trajectory TDI and reference tables.

## 1.3.0

### sklearn-class API

- **`PMHMatcher`**: passes `sklearn.utils.estimator_checks.check_estimator` (isotropic path); `validate_data`, `n_features_in_`, standard `fit(X, y)` with `X_target` in `__init__` or kwargs.
- **`Pipeline.fit`**: target domain via `PMHMatcher(X_target=...)` or metadata routing on the matcher.
- **GridSearchCV**: `make_pmh_pipeline`, `default_pmh_param_grid`, `grid_search_pmh_pipeline`; `tune_sklearn_matcher(..., use_gridsearchcv=True)`.

### TDI and benchmarks

- **`pmh.tdi`**: `tdi_cls`, `tdi_feature_isotropic`, `directional_drift_numpy`, `geometry_report` (paper §6 layout / drift metrics).
- **`compare_arms_sklearn`**: reports target accuracy + **TDI_cls** + **D_N/D_S** per arm; markdown table in `benchmark.md`.
- **Example:** `examples/21_benchmark_sklearn_table.py`; docs: [BENCHMARKS.md](docs/BENCHMARKS.md).

### Documentation

- Sharpened README intro (value prop + 3-step table).
- [BENCHMARKS.md](docs/BENCHMARKS.md) — where TDI lives (package vs paper replication code).

## 1.2.0

### Documentation (adoption)

- **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** — main adoption guide with decision flow and checklists.
- **[CHOOSE_YOUR_SETUP.md](docs/CHOOSE_YOUR_SETUP.md)** — pick API by stack and data.
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — common errors and fixes.
- Restructured [ADAPT_YOUR_PIPELINE.md](docs/ADAPT_YOUR_PIPELINE.md), [QUICKSTART.md](docs/QUICKSTART.md), docs index, mkdocs nav.

### Features

- **D1–D7 in `PMHTrainer.estimate`**: labeled D1, D3 (`augmentations=` / `aug_deltas=`), D5 (`nuisance_indices=`), D6 (`sequences_batches=`), D7 (`style_jsonl` + HF model).
- **`MultiPMHLoss`** / `build_hybrid_trainer` for hybrid nuisances.
- **`HFPMHTrainer`**: HF hidden-state hook + `estimate_style` / `estimate_text_domains`.
- **`PMHMatcher`**: D3 (`aug_deltas=`), D6 (`[N,T,d]`).
- **`DataContext`**, `collect_labeled_features`, `collect_augmentation_deltas`, `collect_sequence_features`.
- Docs: [HYBRID_NUISANCE.md](docs/HYBRID_NUISANCE.md).

## 1.1.0

- **Hook adapters:** `encoder_timm`, `encoder_torchvision_resnet`, `encoder_hf_hidden_states`, `encoder_gnn_mean_pool`; expanded `HOOK_REGISTRY` and [docs/hooks.md](docs/hooks.md).
- **Benchmark protocol:** `run_benchmark_protocol`, falsification arms, markdown reports.
- **CLI:** `pmh-train estimate`, `preflight`, `run`, `list-methods`.

## 1.0.0

Initial public release: `SigmaTaskConfig`, `estimate_from_config`, `PMHLoss`, D1–D7 estimators, PyTorch integration.
