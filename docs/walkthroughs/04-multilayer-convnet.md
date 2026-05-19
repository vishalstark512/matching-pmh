# Walkthrough 4: Multi-layer ConvNet (D3/D4)

**Goal:** Nuisance shows up in **mid-level** feature maps (texture, local color), not only the final pool vector.

**Estimator:** per-layer domain Gram (D4-style) + `MultiLayerPMHLoss`.  
**Script:** `examples/07_vision_multilayer.py`

---

## Prerequisites

```bash
pip install matching-pmh torch
```

---

## Step 1 — Name nuisance

*“Deployment changes local appearance (sensor color, blur) while global semantics stay.”*  
May need penalties at `layer1` and `layer2`, not only `pool`.

---

## Step 2 — Expose layer dict

```python
def forward_features(self, x):
    l1 = F.relu(self.conv1(x))
    l2 = F.relu(self.conv2(l1))
    return {"layer1": l1, "layer2": l2, "pool": self.pool(l2).flatten(1)}
```

Choose `pmh_layers = ("layer1", "layer2")`.

---

## Step 3 — Estimate per-layer $\Sigma$

For each layer name, flatten spatial dims and build a Gram from source vs target feature differences (see script). Alternatively estimate separate artifacts with `SigmaTaskConfig.for_domain` per layer.

---

## Step 4 — Train with `MultiLayerPMHLoss`

```python
from pmh.vision import MultiLayerPMHLoss, gram_sample_noise

pmh_mod = MultiLayerPMHLoss(pmh_layers, PMHConfig(weight=1.0, cap_ratio=0.3))

feats_clean = model.forward_features(x)
feats_noisy = {k: feats_clean[k] + noise_fn[k](feats_clean[k]) for k in pmh_layers}
task = F.cross_entropy(model.fc(feats_clean["pool"]), y)
total, pmh_term = pmh_mod.capped_total(task, feats_clean, feats_noisy)
```

Noise functions sample along the estimated Gram directions (`gram_sample_noise`).

---

## Run

```bash
python examples/07_vision_multilayer.py
```

---

## When to use single-layer vs multi-layer

| Use single `h` (Walkthrough 1–2) | Use multi-layer |
|----------------------------------|-----------------|
| Domain shift in embedding space | Nuisance is spatial / early visual |
| Simpler falsification | More hyperparameters (layers, ranks) |

---

## Adapt

- Replace `TinyCNN` with your U-Net / SegFormer encoder dict
- Keep task loss on the **task head** output; PMH on intermediate maps only
