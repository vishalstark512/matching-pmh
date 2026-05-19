# Roadmap: sklearn-level usability

**matching-pmh** is pipeline-first (your data, your model). This roadmap turns v0.7.x into a library anyone can adopt without reading the research paper.

## Target (v1.0)

| Criterion | Target |
|-----------|--------|
| Happy path | `PMHMatcher.fit(...)` → train with one loss/callback |
| Any frozen features | sklearn `Pipeline` + `GridSearchCV` |
| Any PyTorch model | `PMHTrainer` + hook registry |
| Credible claims | `compare_arms()` one call |
| Tuning | Presets + optional search over `weight`, `cap_ratio`, `rank` |

Low-level API (`estimate_from_config`, `PMHLoss`) stays for power users.

---

## Releases

### v0.7.2 (shipped)

- [ADAPT_YOUR_PIPELINE.md](ADAPT_YOUR_PIPELINE.md)
- `pmh.benchmark`, `pmh-train benchmark`, example `20_compare_training_arms.py`

### v0.8 — sklearn layer (shipped)

- [x] `PMHMatcher` (`BaseEstimator` + `TransformerMixin` when sklearn installed)
- [x] `nuisance=` registry (`domain_shift` → D4, `subspace` → D1, …)
- [x] `get_params` / `set_params` for `GridSearchCV`
- [ ] Full `check_estimator` in CI (optional sklearn extra)
- [ ] Example: shorten `06_office31_sklearn.py`

### v0.9 — PyTorch facade

- [ ] `PMHTrainer` / unified `PMHCallback` (Phase A + B)
- [ ] Hook resolution + shape validation
- [ ] Rewrite `01_domain_shift_d4.py` as showcase

### v1.0 — tune & wizard

- [ ] `suggest_nuisance()` / `nuisance="auto"`
- [ ] `PMHConfig.conservative()` presets
- [ ] `compare_arms` as stable top-level export
- [ ] `pmh.tune` helper (optional)

### v1.1+ — adapters & gallery

- [ ] `pmh.hooks` registry (torchvision, timm, HF, GNN)
- [ ] Notebook gallery with user-data placeholders

---

## Principles (do not break)

- Two phases: estimate Σ once, train with frozen artifact.
- One hook tensor `h` in Phase A and B.
- Falsification: matched + wrong-W + isotropic.
- No paper task replication in the library.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to add an estimator or walkthrough.
