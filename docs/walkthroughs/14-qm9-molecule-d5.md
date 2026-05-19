# Walkthrough 14: QM9 / molecular GNN + D5

**Paper block:** T5A (QM9) — atom **positions** are label-preserving nuisance at deployment noise levels; geometry can still be signal.

**Goal:** Apply D5 on **coordinate channels** of your molecular readout; train regression with PMH.

**Script:** `examples/16_qm9_molecule_d5.py`

---

## Hook & indices

```python
nuisance_idx = [0, 1, 2]  # dims corresponding to x,y,z coordinate block in h
h = gnn.encode_graph(node_features, adj)  # [B, d]
```

Map indices to the partition used when you built $h$ (per-atom pooling, invariant readout, etc.).

---

## Phase A

```python
artifact = estimate_from_config(
    SigmaTaskConfig.for_compositional(nuisance_idx),
    h_collection,  # graphs with position noise in deployment regime
)
```

---

## Phase B

```python
task = mse_loss(head(h), property_target)
total, _ = pmh.capped_total(task, h)
```

---

## Paper lessons (integrate honestly)

- Position-noise PMH can **help** robustness at $\sigma > 0$.
- **Signal-aligned** PMH on coordinates at large $\sigma$ **hurts** MAE (Corollary E) — run wrong partition as negative control.
- Full per-atom gradient $W_c$ spec in paper; library D5 is the coordinate-block estimator.

---

## Run

```bash
python examples/16_qm9_molecule_d5.py
```

Also: [Compositional D5](05-compositional-d5.md), `examples/13_compositional_train_d5.py`.
