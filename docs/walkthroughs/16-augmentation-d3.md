# Walkthrough 16: Known augmentations (D3) — full guide

**At a glance**

| | |
|---|---|
| **Estimator** | D3 — augmentation delta Gram |
| **Script** | `examples/18_augmentation_d3.py` |
| **When** | You **define** the deployment shift as aug modes |

[Walkthrough 4](04-multilayer-convnet.md)

---

## Who this is for

You know deployment shift equals **specific transforms** (blur, JPEG, color jitter) — not an unlabeled second domain.

---

## Your nuisance sentence

*“Deploy sees stronger blur and compression; label unchanged.”*

---

## Step-by-step

### 1. Implement augmentations

```python
def YOUR_AUG(batch_x):
    # return list of augmented tensors per mode
    ...
```

### 2. Collect deltas

```python
from pmh import collect_augmentation_deltas
deltas = collect_augmentation_deltas(encoder, batch, YOUR_AUG)
```

### 3. Train

```python
trainer = PMHTrainer(model, hook=..., nuisance="augmentation")
trainer.estimate(train_loader, augmentations=YOUR_AUG)
trainer.fit(train_loader, epochs=20)
```

```bash
python examples/18_augmentation_d3.py
```

---

## Adaptation worksheet

| Example augs | Your deploy shift |
|--------------|-------------------|
| blur/jpeg | Your pipeline |

---

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Train-time random aug only | D3 modes must **name** deploy shift |
| Unlabeled second domain available | Consider D4 instead |

---

## Next steps

- [4 — Multi-layer](04-multilayer-convnet.md)
- [1 — D4 domain](01-pytorch-domain-d4.md)
