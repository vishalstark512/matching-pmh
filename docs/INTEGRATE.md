# Integrate your project

One page for install, CLI, stacks, data, and ship. **Prerequisites:** [Five-step recipe](FIVE_STEP_RECIPE.md) → [Applications](APPLICATIONS.md) → one [golden path](GOLDEN_PATHS.md).

**Product line:** estimate geometry once → capped PMH → falsification on deploy holdout (Step 5 is not optional for claims).

---

## Repo layout (what to open)

| Path | You use it for |
|------|----------------|
| `src/pmh/` | Library — start with `developer.py`, `recipe.py`, `trainer.py` |
| `examples/00_first_run_domain_shift.py` | 5-minute sanity check |
| `examples/G1–G4` in [GOLDEN_PATHS](GOLDEN_PATHS.md) | Copy-paste for your stack |
| `docs/FIVE_STEP_RECIPE.md` | Product spine |
| `docs/walkthroughs/` | Paper evidence (optional) |
| `research/` | Paper replication only — not the PyPI API |

---

## Install

```bash
pip install matching-pmh torch
python -c "import pmh; print(pmh.__version__)"
pmh-train doctor
```

| Extra | When |
|-------|------|
| `[sklearn]` | Frozen `.npy` / Mode B |
| `[hf]` | LLM / two corpora |
| `[lightning]` | Lightning (G1b) |
| `[vision]` | ResNet / timm examples |

**5-minute demo:** `python examples/00_first_run_domain_shift.py`  
**Colab:** [domain_shift notebook](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/domain_shift_first_run.ipynb)

---

## CLI

| Command | Purpose |
|---------|---------|
| `pmh-train shifts` | Plain English deploy shift types (`nuisance=` keys) |
| `pmh-train recipe` | Print §7 five-step recipe |
| `pmh-train route --task ID` | Full walkthrough for your task |
| `pmh-train route --search KEY` | Finder |
| `pmh-train wizard` | Interactive setup |
| `pmh-train doctor` | Install check + **newbie pipeline checklist** |
| `pmh-train evaluate --demo` | G2 Step 5 (sklearn, synthetic Office-31-style) |
| `pmh-train evaluate --demo --stack pytorch` | G1 Step 5 (synthetic domain shift + small MLP) |
| `pmh-train evaluate --stack pytorch --source-dir A/ --target-dir B/` | Features in folders → MLP + `evaluate_robust_fit` |
| `pmh-train evaluate --source-dir A/ --target-dir B/` | sklearn on `features.npy` folders |
| `pmh-train doctor --stack sklearn --artifact artifacts/run.pt` | Preflight on saved estimate before Step 5 |
| `pmh-train estimate` | Phase A from folders or `.npy` |
| `pmh-train preflight ARTIFACT` | Eigengap on saved Σ̂ |

Details: run `pmh-train -h` · configs in `examples/configs/`

---

## By stack (pick one golden path)

| Stack | Path | Mode |
|-------|------|------|
| PyTorch loop / `robust_fit` | [G1](GOLDEN_PATHS.md#g1) | A |
| Lightning | [G1b](GOLDEN_PATHS.md#g1b) | A |
| sklearn / `.npy` | [G2](GOLDEN_PATHS.md#g2) | B |
| HF two corpora | [G3](GOLDEN_PATHS.md#g3) | A |
| HF `Trainer` / DPO | [G3b](GOLDEN_PATHS.md#g3b) | A |
| Your own Σ̂ / W | [G4](GOLDEN_PATHS.md#g4) | A or B |

**Hooks (Mode A):** one layer `h` for estimate and train — backbone / pooler before the task head. `suggest_hook(model)` or `hook="auto"` in `robust_fit`.

**sklearn (Mode B):** `PMHMatcher` in a `Pipeline`; score on **target holdout**, not train-only accuracy.

**Low-level PyTorch:** `PMHTrainer` + `PMHLoss` — same two phases as G1.

**Lightning:** `PMHLightningCallback` + `add_pmh_to_loss` — see `examples/09_lightning_module.py`.

**HF Trainer:** `get_pmh_trainer()` — see `examples/10_hf_trainer.py`.

---

## Data layout

- Site A / B folders with `features.npy` (and optional labels for D1): `pmh-train estimate --source-dir A/ --target-dir B/`
- In-memory: pass tensors to `estimate_from_config` or `PMHTrainer.estimate`
- Custom deltas / W: [CUSTOM_GEOMETRY.md](CUSTOM_GEOMETRY.md) (advanced)

---

## Tunable parameters

Yes — you can change PMH settings. Summary:

| Knob | Object | Used in |
|------|--------|---------|
| Penalty strength / cap / warmup | `PMHConfig` | `PMHTrainer`, `robust_fit` |
| Subspace rank, nuisance family | `rank`, `nuisance` | Trainer, Matcher, `evaluate_*` |

Full table: **[PMH_PARAMETERS.md](PMH_PARAMETERS.md)** · one-page copy-paste: **[PARAMETERS_CHEATSHEET.md](PARAMETERS_CHEATSHEET.md)**.

```python
from pmh import PMHConfig, PMHTrainer

trainer = PMHTrainer(model, hook=backbone, rank=32, pmh_config=PMHConfig.conservative())
```

---

## Ship

```python
from pmh import export_deployment, PMHConfig
# after estimate: export_deployment(artifact, "deploy/bundle", pmh_config=PMHConfig.balanced())
```

Bundle reload: `load_deployment_bundle` — used for handoff / reproducibility.

---

## Before production (Step 5)

**sklearn / frozen features** — default report includes falsification arms:

```python
from pmh import evaluate_baseline_vs_pmh

report = evaluate_baseline_vs_pmh(x_source, y_source, x_target, y_target)
print(report.summary())  # matched, wrong_w, isotropic on deploy holdout
```

**PyTorch** — same holdout via `compare_arms` after `robust_fit` (see [falsification walkthrough](walkthroughs/08-falsification-controls.md)).

1. Deploy holdout metric (not source-only accuracy)  
2. `preflight` marginal → Office-31-style risk ([TROUBLESHOOTING](TROUBLESHOOTING.md))  
3. Only then ship `export_deployment`

**Compare to CORAL:** useful baseline for Mode B — [COMPARE_TO_CORAL.md](COMPARE_TO_CORAL.md) (optional read).

---

## API lookup

Generated reference: [api/index.md](api/index.md) · stable Tier 0: `robust_fit`, `PMHTrainer`, `PMHMatcher`, `explain_task`.
