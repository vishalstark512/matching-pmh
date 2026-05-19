# Walkthrough 5: Compositional nuisance (D5)

**Goal:** Nuisance lives on **known coordinates** of your representation (atom indices, token positions, sensor channels)—not a full-domain rotation.

**Estimator:** D5 (block covariance on `nuisance_indices`).  
**Scripts:** `examples/03_compositional_d5.py` (estimate only), `examples/13_compositional_train_d5.py` (estimate + train)

---

## Prerequisites

```bash
pip install matching-pmh torch
```

---

## Step 1 — Name nuisance

Examples:

- *“Atom positions jitter but molecular label unchanged.”* → nuisance indices = atom coordinate slots in $h$.
- *“Only whitespace / comment tokens may change formatting.”* → nuisance indices = those token embedding dims.

---

## Step 2 — List `nuisance_indices`

```python
nuisance_idx = [0, 1, 2, 3, 4]  # length k; representation dim d >= k
cfg = SigmaTaskConfig.for_compositional(nuisance_idx)
```

---

## Step 3 — Phase A: estimate

```python
# h: [N, d] from your encoder on deployment-style data
artifact = estimate_from_config(cfg, h)
print(artifact.sigma[:k, :k].norm())   # active block
print(artifact.sigma[k:, k:].norm())   # should be ~0 off-block
```

```bash
python examples/03_compositional_d5.py
```

---

## Step 4 — Phase B: train

```python
h = gnn_readout(batch)   # [B, d]
task_loss = your_loss(h, y)
total, _ = pmh.capped_total(task_loss, h)
```

Full loop:

```bash
python examples/13_compositional_train_d5.py
```

---

## Step 5 — GNN-specific notes

| Piece | Guidance |
|-------|----------|
| $h$ | Graph-level readout or pooled node states |
| Indices | Align with the coordinate system used when building $h$ |
| Estimation data | Graphs with label-preserving perturbations on nuisance coords |

If nuisance is **global** (whole-graph style), prefer D4 or D7 instead.

---

## Controls

Matched D5 vs wrong-W on the same $h$; off-block $\Sigma$ should stay near zero for valid D5 specs.
