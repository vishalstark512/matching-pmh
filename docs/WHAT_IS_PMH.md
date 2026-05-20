# What is matching-pmh? (no paper required)

You train a model on **one environment** and deploy on **another** — same task, same label meaning, different look or sensor or formatting.

**matching-pmh** adds a regularizer so your model is less sensitive to the ways inputs change between those environments, without replacing your architecture or task loss.

---

## The situation it solves

| You have | Example |
|----------|---------|
| Training data from site A | Hospital 1, camera A, country A corpus |
| Deployment on site B | Hospital 2, new phone camera, new customer segment |
| Labels mean the same thing | “Positive” still means the same disease / intent / defect |

**Good fit.** PMH estimates “what varies between A and B but should not change the label” and penalizes sensitivity along those directions in your representation `h`.

---

## When **not** to use it

| Situation | Use instead |
|-----------|-------------|
| New classes appear at test time | Proper label-shift / open-set methods |
| You only want generic robustness to any noise | Standard augmentation, adversarial training |
| You have no signal from environment B | Collect a small unlabeled target sample (typical domain-adaptation setup) |
| Labels are not comparable across domains | Fix labeling / merge classes first |

**Full guide (expectations, benchmarks, controls):** [When PMH helps](WHEN_PMH_HELPS.md).

---

## How it compares to tools you may know

| Approach | What it does |
|----------|----------------|
| **Train on source only (ERM)** | Fast baseline; often fails on target |
| **Fine-tune on target** | Strong if you have many target labels |
| **CORAL / moment matching** | Aligns feature statistics between domains |
| **matching-pmh (this library)** | Keeps your task loss; adds a penalty on representation sensitivity along **deployment-specific** directions (estimated from source + target data) |

PMH is closest to **domain adaptation with a named shift**, not to “drop in a new loss and forget about domains.”

---

## What you actually run (default)

**Check first:**

```python
from pmh import check_applicability
print(check_applicability(stack="pytorch", n_source=500, n_target=400).summary())
```

**PyTorch** — two data streams (source + target), one hook layer:

```python
from pmh import robust_fit

out = robust_fit(
    model,
    train_loader,
    source_batches=src_loader,
    target_batches=tgt_loader,
    hook="auto",
    epochs=20,
)
print(out.preflight_message)
```

**Frozen features + sklearn** — NumPy matrices for source and target:

```python
from pmh import PMHMatcher

matcher = PMHMatcher(nuisance="domain_shift").fit(x_source, x_target)
x_train_robust = matcher.transform(x_source)
```

You do **not** need to read about D1–D7 to start. Those are advanced estimator choices for specific shift types (style, augmentations, sequences, …).

---

## Two steps under the hood (optional detail)

1. **Estimate once** — compare batches from A and B at a fixed layer `h`; build a geometry summary (saved as an artifact).
2. **Train as usual** — your classification / regression loss + an extra term that discourages large changes in `h` along that geometry.

Same layer `h` in both steps.

---

## Next steps

| Step | Doc |
|------|-----|
| Run a 2-minute demo (Colab or local) | [First hour](FIRST_HOUR.md) · [Colab](COLAB.md) |
| Copy into your project | [First hour](FIRST_HOUR.md) → [Integrate](GETTING_STARTED.md) |
| Pick vision / tabular / NLP template | [Gallery](gallery/README.md) |
| Before trusting a production claim | [Falsification controls](walkthroughs/08-falsification-controls.md) |
| Paper / benchmark fidelity | [Research → Paper alignment](PAPER_ALIGNMENT.md) |

**Not sure which API?**

```python
from pmh.onboarding import recommend_setup, print_setup_guide

print_setup_guide(stack="pytorch", has_target_domain=True, has_target_labels=False)
```

Or: `pmh-train wizard` from the shell after install.
