# Walkthrough 16: Augmentation modes + D3

**Paper block:** T2 photometric / finite augmentation family — nuisance is a **span of known modes**, not a full domain shift.

**Goal:** Estimate $\Sigma_{\mathrm{task}}$ from feature-space augmentation deltas; train with matched PMH.

**Script:** `examples/18_augmentation_d3.py`

---

## Phase A — stack mode deltas

For each augmentation mode $k$ (color jitter, blur, …):

```python
with torch.no_grad():
    h0 = encoder(x)
    delta_k = encoder(aug_k(x)) - h0   # per sample or batch mean
```

Stack to `[K, d]` or `[K, N, d]`:

```python
artifact = estimate_from_config(
    SigmaTaskConfig.for_augmentation(),
    aug_deltas=stack,
)
```

---

## Phase B

Standard `PMHLoss` on $h = encoder(x)$ during training (augmentations may still be in the task pipeline).

---

## D3 vs D4

| Use D3 | Use D4 |
|--------|--------|
| Finite known aug set | Unlabeled target domain / site |
| $\Sigma$ from mode Gram | $\Sigma$ from source–target feature Gram |

Hybrid: estimate both and **add two PMH terms** (paper §5).

---

## Run

```bash
python examples/18_augmentation_d3.py
```

Vision: see also [Multi-layer CNN](04-multilayer-convnet.md) for spatial feature maps.
