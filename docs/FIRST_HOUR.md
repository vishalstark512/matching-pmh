# Your first hour with matching-pmh

No paper. Default story: **train on domain A, deploy on domain B, same labels.**

---

## 1. Install (2 min)

```bash
pip install matching-pmh torch
# Frozen features + sklearn later:
# pip install "matching-pmh[sklearn]"
```

```bash
python -c "import pmh; print('matching-pmh', pmh.__version__)"
```

---

## 2. Run the demo (2 min)

**Colab (no clone):** [PyTorch notebook](COLAB.md) · [sklearn / frozen features notebook](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/sklearn_frozen_features_first_run.ipynb)

**Local:**

```bash
git clone https://github.com/vishalstark512/matching-pmh.git
cd matching-pmh
python examples/00_first_run_domain_shift.py
```

You should see something like:

```text
Target accuracy (baseline ERM):  0.xx
Target accuracy (with PMH):      0.xx
Preflight: pass
```

Synthetic data — the point is the **workflow**, not the exact numbers.

Quick CI mode: `PMH_QUICK=1 python examples/00_first_run_domain_shift.py`

---

## 3. Which API is yours? (1 min)

| Stack | Golden path |
|-------|-------------|
| PyTorch loop | **G1** |
| PyTorch Lightning | **G1b** |
| sklearn / `.npy` | **G2** |
| HF two corpora | **G3** |
| HF `Trainer` (DPO, LoRA, …) | **G3b** |

**Interactive:**

```bash
pmh-train wizard
```

**Or in Python:**

```python
from pmh.onboarding import run_wizard

run_wizard(interactive=True)   # questionnaire
# run_wizard(stack="pytorch", interactive=False)  # scripted
```

---

## 4. Copy the default integration (10 min)

### PyTorch (most common)

You need:

- `model` + `hook` where `h` has shape `[batch, d]`
- `train_loader` (labeled task data)
- `source_batches` and `target_batches` (target labels optional)

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    model,
    hook=backbone,                    # nn.Module or "layer_name"
    head=classifier,                  # optional if logits inside model
    nuisance="domain_shift",
    pmh_config=PMHConfig.balanced(),
    artifact_path="artifacts/run1.pt",
)

trainer.fit(
    train_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    epochs=20,
)
```

Full guide: [Walkthrough 1 — PyTorch domain shift](walkthroughs/01-pytorch-domain-d4.md).

### sklearn (frozen features)

You need `x_source`, `x_target` (same feature dim), labels on source for training downstream.

```python
from pmh import PMHMatcher
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("adapt", PMHMatcher(nuisance="domain_shift").fit(x_source, x_target)),
    ("clf", LogisticRegression(max_iter=500)),
])
pipe.fit(x_source, y_source)
```

Evaluate on **target** holdout. Guide: [Walkthrough 3](walkthroughs/03-office31-sklearn-d1.md).

---

## 5. Sanity-check before you ship (later)

When you want evidence the gain is real (not generic regularization), run falsification arms — **after** the basic integration works:

- [Walkthrough 8 — Controls](walkthroughs/08-falsification-controls.md)
- PyTorch: `compare_arms(...)`
- sklearn: `compare_arms_sklearn(...)`

Researchers benchmarking against the paper: [CORRECT_USAGE.md](CORRECT_USAGE.md) (not required for first integration).

---

## 6. Stuck?

| Symptom | Doc |
|---------|-----|
| Hook shape errors | [hooks.md](hooks.md) |
| `preflight` warnings | [Troubleshooting glossary](TROUBLESHOOTING.md#plain-language-glossary) |
| CORAL vs PMH | [COMPARE_TO_CORAL.md](COMPARE_TO_CORAL.md) |
| Pick shift type D1–D7 | [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) |

---

## After the first hour

- [Integrate your project](GETTING_STARTED.md) — afternoon checklist  
- [Golden paths](GOLDEN_PATHS.md) — G1 / G1b / G2 / G3 / G3b / G4  
- [Gallery](gallery/README.md) — examples by domain
