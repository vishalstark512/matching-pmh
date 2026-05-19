# Walkthrough 11: Temporal sequences (D6)

**Goal:** Label constant along a **time axis**, but sensors / markets drift within the window (HAR, finance, physiological signals).

**Estimator:** D6 (`SigmaTaskConfig.for_temporal()`).  
**Script:** No dedicated example yet—API walkthrough below.

---

## Prerequisites

```bash
pip install matching-pmh torch
```

---

## Step 1 — Name nuisance

*“Within this window, the activity label is fixed, but sensor bias drifts over time.”* → **D6**.

---

## Step 2 — Build sequences of $h_t$

Run your encoder on windows:

```python
# sequences: list of [T_i, d] tensors, one per example
sequences = []
for window in dataset:
    with torch.no_grad():
        h_t = encoder(window)   # [T, d] per timestep or per segment
    sequences.append(h_t)
```

---

## Step 3 — Estimate

```python
from pmh import SigmaTaskConfig, estimate_from_config

cfg = SigmaTaskConfig.for_temporal()
artifact = estimate_from_config(cfg, sequences=sequences)
print(artifact.preflight, artifact.method)
artifact.save("artifacts/d6_temporal")
```

`estimate_from_config` routes `method=="D6"` to the temporal estimator (residual scatter along time).

---

## Step 4 — Train

Use the same `PMHLoss` on the **pooled** or **last** $h$ you used to define the sequence semantics—or on per-step $h_t$ if your training loop backprops through time:

```python
h = encoder(sequence)          # [B, T, d] or [B, d]
task_loss = criterion(h, y)
total, _ = pmh.capped_total(task_loss, h.reshape(-1, d) if h.dim() == 3 else h)
```

Keep representation definition **consistent** between Phase A and B.

---

## Step 5 — CLI

```bash
pmh-train list-methods   # shows D6
```

For batch jobs, pass precomputed sequence arrays via the CLI data schema in [nuisance_types.md](../nuisance_types.md) (D6 section).

---

## Related walkthroughs

- Global domain shift on pooled windows → often **D4** ([Walkthrough 1](01-pytorch-domain-d4.md))
- Coordinate-specific sensor channels → **D5** ([Walkthrough 5](05-compositional-d5.md))
