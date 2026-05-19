# Changelog

## 1.4.1 (unreleased)

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
- **Gallery:** [docs/gallery/](docs/gallery/) templates (vision, tabular, NLP).
- **Walkthrough 18:** PMHTrainer quickstart.
- **CI:** `sklearn-compat` job with `check_get_params` / `check_set_params`.
- **Fix:** `scripts/upload_pypi.ps1` version read on Windows (tomllib).

## 1.0.0

- **`PMHTrainer`**: Phase A estimate + Phase B `fit()` on your DataLoader; hook resolution via `pmh.hooks`.
- **`nuisance="auto"`**: `suggest_nuisance()` picks D1/D4/etc. from data flags.
- **`PMHConfig` presets**: `conservative()`, `balanced()`, `aggressive()`, `finetune_llm()`.
- **Top-level `compare_arms` / `compare_arms_sklearn`**: credible B0 vs matched vs controls.
- **`tune_sklearn_matcher` / `tune_pmh_config`**: small grid search helpers.
- Examples `01` and `06` rewritten as showcases.

## 0.8.0

- **`PMHMatcher`**: sklearn-style `fit` / `transform` / `get_params` on NumPy features (D1, D2, D4, D5).
- **`nuisance=` registry**: `domain_shift`, `subspace`, `isotropic`, … → D1–D7.
- Docs: [ROADMAP.md](docs/ROADMAP.md), updated [sklearn.md](docs/sklearn.md).

## 0.7.2

- **Pipeline-first docs:** [ADAPT_YOUR_PIPELINE.md](docs/ADAPT_YOUR_PIPELINE.md), walkthrough 17 (compare arms on your model).
- **`pmh.benchmark`:** `run_benchmark_protocol`, `run_sklearn_benchmark`, `write_benchmark_report`; CLI `pmh-train benchmark`.
- **Example:** `20_compare_training_arms.py` (B0 / matched / wrong-W / isotropic template).
- **Fix:** cap wrong-W rank to representation dimension `d`.

## 0.7.1

- README: PyPI-safe formatting (no LaTeX); clear intro and D1–D7 table with plain `P(y given x)` and full `SigmaTaskConfig` examples.

## 0.7.0

- **Lab-grade docs:** Quickstart, Philosophy, 16 walkthroughs (ViT, Whisper, QM9, CodeBERT, D3 aug), examples catalog.
- **New examples:** `14_vit_cls_d4.py`, `15_speech_encoder_d4.py`, `16_qm9_molecule_d5.py`, `17_code_tokens_d5.py`, `18_augmentation_d3.py`.
- **Community:** CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue templates.
- **CI:** smoke-test runnable examples.
- README: badges, integration map, citation block.

## 0.6.4

- Eleven walkthroughs under `docs/walkthroughs/` (PyTorch, ResNet, Office-31, multi-layer CNN, D5, LLM D7, HF Trainer, controls, CLI, Lightning, D6).
- New examples: `12_resnet_hook_d4.py`, `13_compositional_train_d5.py`.

## 0.6.3

- README: display math ($$ blocks) for Problem / Object / Repair / Unification.
- New [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md): two-phase workflow, where `h` hooks in, patterns for PyTorch, CNN/ViT, HF Trainer, D5, sklearn, D7.

## 0.6.2

- README and [docs/THEORY.md](docs/THEORY.md): matching principle for **any task / any architecture**; five-step recipe; falsification controls; explicit scope limits.

## 0.6.1

- **README / PyPI description:** plain-language problem statement, workflow, D1–D7 decision table, and per-domain use cases (vision, D5, D6, D7/LLM, CLI, falsification arms).

## 0.6.0

- **`pmh-train` CLI**: `list-methods`, `estimate --config job.json`, `preflight`, `run --config job.json`.
- **`pmh.catalog`**: D1–D7 input requirements and job validation.
- **Example 11**: Qwen/T7A JSONL (`style_pairs` + `preference_pairs`), optional LoRA + DPO+PMH demo.
- **Bundled samples**: `examples/data/*.jsonl`, `examples/configs/*.json`.
- **HF**: `load_preference_pairs_jsonl` for DPO schema.
- **Publishing**: TestPyPI workflow (`publish-testpypi.yml`), updated `PUBLISHING.md`.

## 0.5.0

- HF `PMHTrainer`, CORAL baseline, CI matrix, Office-31 example.

## 0.4.0

- Hugging Face D7, Lightning callback, Office-31 loader.

## 0.3.0

- Torch/sklearn/vision integrations, MkDocs.

## 0.2.0

- Artifacts, `PMHLoss`, configs.

## 0.1.0

- Core estimators D1–D7 and penalties.
