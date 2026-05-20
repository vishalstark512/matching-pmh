# Walkthrough 17: Compare arms on **your** pipeline — full guide


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden paths](../GOLDEN_PATHS.md) · **Step 5:** examples/20_compare_training_arms.py
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**At a glance**

| | |
|---|---|
| **Purpose** | B0 / matched / wrong-W / isotropic on **your** model + data |
| **Output** | `benchmark.md` + `benchmark.json` |
| **Script** | `examples/20_compare_training_arms.py` |
| **Required** | After PMH integrates; before publication claims |

[Walkthrough 8](08-falsification-controls.md) · [BENCHMARKS.md](../PAPER_ALIGNMENT.md)

---

## Who this is for

You already train with `PMHTrainer` or `PMHLoss` and need a **fair ablation table** without rewriting training loops four times by hand.

Also use when:

- Writing a report section “matched vs controls”.
- Debugging “PMH helped” vs “any penalty helped”.

---

## Your deployment shift sentence

*Before we ship, matched PMH must beat B0, wrong-W, and isotropic on our deploy split.* -> required evidence.

---

## What you implement (two callbacks)

The example script expects **your** training code behind these hooks:

```python
def model_factory():
    """Return a fresh nn.Module (same architecture each arm)."""
    return YOUR_BUILD_MODEL()

def setup_model(model):
  """Return dict with encoder, head, optimizer, etc."""
    return {
        "encoder": model.backbone,
        "head": model.head,
        "optimizer": torch.optim.Adam(model.parameters(), lr=YOUR_LR),
        # ... whatever your loop needs
    }
```

Open `examples/20_compare_training_arms.py` and replace the toy `ToyModel` with imports from **your** codebase.

---

## Step-by-step

### 1. Fix artifact (Phase A done once)

```python
from pmh import PMHTrainer

trainer = PMHTrainer(...)
trainer.estimate(source_batches=..., target_batches=...)
artifact = trainer.artifact_
```

Or load: `SigmaTaskEstimate.load("artifacts/sigma.pt")`.

### 2. Point val loader at deployment

```python
val_loader = YOUR_TARGET_DOMAIN_LOADER   # not train-source only
```

### 3. Run comparison

```bash
python examples/20_compare_training_arms.py --out results/YOUR_EXPERIMENT
```

Or API:

```python
from pmh import compare_arms

compare_arms(
    artifact,
    model_factory=YOUR_MODEL_FACTORY,
    setup_model=YOUR_SETUP,
    train_loader=YOUR_TRAIN_LOADER,
    val_loader=YOUR_VAL_LOADER,
    epochs=YOUR_EPOCHS,
    report_dir="results/YOUR_EXPERIMENT",
)
```

### 4. Read report

Open `results/YOUR_EXPERIMENT/benchmark.md`.

---

## sklearn-only path

No end-to-end training — frozen features:

```bash
python examples/21_benchmark_sklearn_table.py --report results/sklearn_arms
```

```python
from pmh import compare_arms_sklearn
compare_arms_sklearn(x_src, y_src, x_tgt, y_tgt, report_dir="results/run1")
```

---

## Interpretation table

| Pattern | Meaning |
|---------|---------|
| matched > b0, wrong_w ≈ isotropic | Strong matched-geometry story |
| matched > b0, wrong_w also > b0 | Generic regularization — lower `weight`, check hook |
| matched ≈ b0 | Weak ID — check `preflight`, shift type (`nuisance=`), Dk |
| All similar | Val metric may not reflect deployment shift |

---

## Adaptation worksheet

| Template | Your value |
|----------|------------|
| `model_factory` | |
| `setup_model` | |
| `val_loader` | |
| `epochs` per arm | |
| `report_dir` | |

---

## Verify success

- [ ] Four arms in `benchmark.md`
- [ ] Same `rank` for matched / wrong_w / isotropic
- [ ] Metric name documents target vs source

---

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Shared weights across arms | New model per arm |
| Source validation | Target validation |
| Skipping isotropic | Required control |

---

## Next steps

- [8 — Controls theory](08-falsification-controls.md)
- [1 — Training setup](01-pytorch-domain-d4.md)
