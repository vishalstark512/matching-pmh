# Adopt matching-pmh (60 seconds)

**Train on site A. Deploy on site B. Same labels.**

This file is the **only** root doc you need to start integrating. Everything else is depth.

---

## Run this first

```bash
pip install matching-pmh torch
pmh-train doctor
pmh-train evaluate --demo              # sklearn Step 5 smoke (no data)
# pmh-train evaluate --demo --stack pytorch
```

**`nuisance=`** = **deployment shift type** (Hospital A vs B, same labels). Not “bad data.”  
[What is deployment shift?](docs/WHAT_IS_DEPLOYMENT_SHIFT.md)

---

## Five steps (your pipeline)

| Step | Action |
|------|--------|
| **0** | Same labels on train and deploy? → `check_applicability()` |
| **1** | Pick shift type → `pmh-train shifts` or `suggest_nuisance()` |
| **2** | Estimate geometry once (Phase A) |
| **3** | Train with capped PMH (Phase B) |
| **4** | Cap protocol → `PMHConfig.balanced()` |
| **5** | **Required:** falsification on deploy holdout → `evaluate_*` or [walkthrough 08](docs/walkthroughs/08-falsification-controls.md) |

---

## Pick your stack (one path)

| Daily AI work | Golden path | First command |
|---------------|-------------|---------------|
| Train/fine-tune PyTorch (CV, audio encoder, custom) | [G1](docs/GOLDEN_PATHS.md#g1) | `python examples/00_first_run_domain_shift.py` |
| Frozen embeddings / sklearn / `.npy` | [G2](docs/GOLDEN_PATHS.md#g2) | `pmh-train evaluate --demo` |
| LLM (two corpora or style pairs) | [G3 / G3b](docs/GOLDEN_PATHS.md#g3) | `pmh-train route --task llm_style_or_format` |
| Lightning | [G1b](docs/GOLDEN_PATHS.md#g1b) | `examples/09_lightning_module.py` |
| Custom Σ / W already | [G4](docs/GOLDEN_PATHS.md#g4) | `docs/CUSTOM_GEOMETRY.md` |

Task finder: `pmh-train route --task vision_classification` · `pmh-train route --search hospital`

---

## Docs ladder (read in order)

1. [WHAT_IS_DEPLOYMENT_SHIFT.md](docs/WHAT_IS_DEPLOYMENT_SHIFT.md)  
2. [FIVE_STEP_RECIPE.md](docs/FIVE_STEP_RECIPE.md)  
3. [APPLICATIONS.md](docs/APPLICATIONS.md)  
4. [GOLDEN_PATHS.md](docs/GOLDEN_PATHS.md) ← copy code from here  
5. [INTEGRATE.md](docs/INTEGRATE.md)  
6. [PARAMETERS_CHEATSHEET.md](docs/PARAMETERS_CHEATSHEET.md) (when tuning)  
7. [WHEN_PMH_HELPS.md](docs/WHEN_PMH_HELPS.md) · [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

**Evidence walkthroughs** (paper depth): [walkthroughs/DAILY_AI_USE.md](docs/walkthroughs/DAILY_AI_USE.md) — use **after** golden path works.

---

## Templates

Copy [templates/matching-pmh-starter/](templates/matching-pmh-starter/) into your repo.

[Full README](README.md) · [PyPI](https://pypi.org/project/matching-pmh/) · [Docs site](https://vishalstark512.github.io/matching-pmh/)
