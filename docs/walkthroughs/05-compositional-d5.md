# Walkthrough 5: Compositional coordinates (D5) — full guide

**At a glance**

| | |
|---|---|
| **Estimator** | D5 — block coordinates in input or feature vector |
| **Scripts** | `examples/03_compositional_d5.py`, `13_compositional_train_d5.py` |
| **API** | `nuisance_indices=` on `PMHTrainer` / `PMHMatcher` |

[NUISANCE_SUBTYPES.md](../NUISANCE_SUBTYPES.md#d5-compositional)

---

## Who this is for

Input has **known nuisance coordinates** (e.g. sensor axes, token groups, molecular descriptors) separate from task coordinates.

---

## Your nuisance sentence

*“Channels 0–2 are site metadata; channels 3–end are biology — label depends only on biology.”*

---

## Step-by-step

1. List indices: `nuisance_indices = [0, 1, 2]`.
2. Phase A:

```python
trainer = PMHTrainer(model, hook=..., nuisance="compositional", nuisance_indices=[0,1,2])
trainer.estimate(source_batches=...)
```

3. Phase B: `trainer.fit(...)`.

```bash
python examples/03_compositional_d5.py
python examples/13_compositional_train_d5.py
```

---

## Adaptation worksheet

| Example | Your project |
|---------|--------------|
| Synthetic 8-d split | Your feature partition |
| QM9 / CodeBERT | [14](14-qm9-molecule-d5.md), [15](15-codebert-tokens-d5.md) |

---

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Wrong index split | Task coords must not include pure nuisance |
| Using D4 when coords known | D5 is more precise |

---

## Next steps

- [14 — QM9](14-qm9-molecule-d5.md)
- [15 — Code tokens](15-codebert-tokens-d5.md)
