# Walkthrough 11: Temporal sequences (D6) — full guide


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden path G1–G4](../GOLDEN_PATHS.md#g1) · **Route:** `pmh-train route --task temporal_drift` · **Step 5:** compare_arms on sequence holdout
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**At a glance**

| | |
|---|---|
| **Estimator** | D6 — temporal residual scatter |
| **Input** | Sequences `[N, T, d]` per window |
| **API** | `PMHTrainer(sequences_batches=...)` or `PMHMatcher.fit(X)` with 3D `X` |

[Walkthrough 1](01-pytorch-domain-d4.md)

---

## Who this is for

Label **constant within a window**, but sensor / market / physiology **drifts over time** (HAR, finance, ICU waveforms).

---

## Your deployment shift sentence

*“Activity class fixed in this window; sensor bias drifts across timesteps.”*

---

## Step-by-step

### 1. Build `[T, d]` per example

```python
sequences = []
for window in YOUR_DATASET:
    with torch.no_grad():
        h_t = YOUR_ENCODER(window)   # [T, d]
    sequences.append(h_t.cpu())
```

### 2. Estimate

```python
from pmh import PMHTrainer, SigmaTaskConfig, estimate_from_config

# Option A — Trainer
trainer = PMHTrainer(model, hook=..., nuisance="temporal")
trainer.estimate(sequences_batches=YOUR_SEQ_LOADER)

# Option B — numpy 3D array
import numpy as np
X = np.stack(sequences)   # [N, T, d]
PMHMatcher(nuisance="temporal").fit(X)
```

### 3. Train

Same `h` semantics as estimate — pooled or per-step; document your choice.

```python
pmh.capped_total(task_loss, h.reshape(-1, d) if h.dim() == 3 else h)
```

---

## Adaptation worksheet

| Template | Your setup |
|----------|------------|
| Window length T | |
| Pooling rule | |

---

## Verify & controls

- [ ] `artifact.method == "D6"`
- [ ] Falsification arms — [walkthrough 08](08-falsification-controls.md)

---

## Next steps

- Pooled windows only → often [D4 Walkthrough 1](01-pytorch-domain-d4.md)
- Known coord channels → [5 — D5](05-compositional-d5.md)
