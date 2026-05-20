# Your first hour with matching-pmh

**Read first:** [START_HERE](START_HERE.md) (5 min) → [Find your application](APPLICATIONS.md) (task + nuisance) → then this page (install + demo).

---

## 1. Install (2 min)

```bash
pip install matching-pmh torch
python -c "import pmh; print('matching-pmh', pmh.__version__)"
pmh-train doctor
```

---

## 2. Route your task (1 min)

```bash
pmh-train route --task pose_or_keypoints
# or: pmh-train route
```

---

## 3. Run the demo (2 min)

**Colab:** [PyTorch notebook](COLAB.md) · [sklearn notebook](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/sklearn_frozen_features_first_run.ipynb)

**Local:**

```bash
git clone https://github.com/vishalstark512/matching-pmh.git
cd matching-pmh
python examples/00_first_run_domain_shift.py
```

Expected: baseline vs PMH target accuracy + `Preflight: pass`. Numbers are synthetic — learn the **workflow**.

`PMH_QUICK=1 python examples/00_first_run_domain_shift.py` for CI-speed.

---

## 4. Pick one golden path (1 min)

| Stack | Section |
|-------|---------|
| PyTorch | [G1](GOLDEN_PATHS.md#g1) |
| Lightning | [G1b](GOLDEN_PATHS.md#g1b) |
| sklearn / `.npy` | [G2](GOLDEN_PATHS.md#g2) |
| HF corpora | [G3](GOLDEN_PATHS.md#g3) |
| HF `Trainer` | [G3b](GOLDEN_PATHS.md#g3b) |

```bash
pmh-train wizard
```

---

## 5. Copy into your project (10 min)

Use the snippet from your golden path. Pose / vision example:

```python
from pmh import robust_fit, suggest_hook

hook = suggest_hook(model).hook
out = robust_fit(
    model, train_loader,
    source_batches=loader_a, target_batches=loader_b,
    hook=hook, epochs=20,
)
```

Full afternoon checklist: [GETTING_STARTED](GETTING_STARTED.md).

---

## 6. Before production (later)

[Falsification controls](walkthroughs/08-falsification-controls.md) — after basic integration works.

---

## Stuck?

| Symptom | Doc |
|---------|-----|
| Hook errors | [hooks.md](hooks.md) |
| Preflight warnings | [TROUBLESHOOTING](TROUBLESHOOTING.md) |
| Doc overload | [MAP.md](MAP.md) |
