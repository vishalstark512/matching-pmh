# Walkthrough 1: PyTorch domain shift (D4)

**Goal:** Add matched PMH to an existing PyTorch training loop when deployment shifts look like a **different site/camera/corpus** but labels are still meaningful ($P(y\mid x)$ roughly stable).

**Estimator:** D4 (domain Gram on unlabeled source vs target features).  
**Script:** `examples/01_domain_shift_d4.py`

---

## Prerequisites

```bash
pip install matching-pmh torch
```

You need:

- `backbone(x) -> h` with shape `[B, d]`
- Batches from a **source** domain and a **target** domain (labels optional for D4 estimation)

---

## Step 1 — Name the nuisance

Example: *“At deployment, inputs are brighter / from another camera, but the class label is unchanged.”*

That is a **domain shift** story → **D4**.

---

## Step 2 — Choose the hook layer

Pick one representation $h=\phi(x)$:

- Penultimate layer before the classifier (most common)
- Same layer you would use for linear probing

Record `d = h.shape[-1]`.

---

## Step 3 — Phase A: estimate $\hat\Sigma_{\mathrm{task}}$

Use a **frozen or warm** backbone (no PMH yet):

```python
from pmh import SigmaTaskConfig, collect_features, estimate_from_config

backbone.eval()
h_src = collect_features(backbone, source_batches, max_batches=50)
h_tgt = collect_features(backbone, target_batches, max_batches=50)

artifact = estimate_from_config(
    SigmaTaskConfig.for_domain(rank=32),  # tune rank from eigengap
    h_src,
    h_tgt,
)
print(artifact.preflight, artifact.eigengap)
artifact.save("artifacts/d4_sigma")
```

**Preflight:** prefer `pass`. `marginal` means weak identification (see Office-31 walkthrough).

---

## Step 4 — Phase B: train with `PMHLoss`

```python
from pmh import PMHLoss, PMHConfig

pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2))

backbone.train()
for epoch in range(num_epochs):
    pmh.set_epoch(epoch)
    for x, y in train_loader:
        opt.zero_grad()
        h = backbone(x)
        task_loss = criterion(head(h), y)
        total, pmh_term = pmh.capped_total(task_loss, h)
        total.backward()
        opt.step()
```

`cap_ratio=0.3` keeps PMH from overpowering the task loss.

---

## Step 5 — Controls

Train three runs (or swap `mode=`):

| Run | Code |
|-----|------|
| Matched | `PMHLoss(artifact, cfg)` |
| Wrong-W | `PMHLoss(artifact, cfg, mode="wrong_w")` |
| Isotropic | `PMHLoss(artifact, cfg, mode="isotropic")` |

See [Walkthrough 8](08-falsification-controls.md).

---

## Run the example

```bash
python examples/01_domain_shift_d4.py
```

Expected: prints `preflight`, then epoch logs with `task` and `pmh` decreasing or stabilizing.

---

## Adapt to your project

| Paper toy | Your project |
|-----------|----------------|
| `Backbone` MLP | Your `nn.Module` / ResNet / ViT |
| `make_loader` synthetic | `DataLoader` for source vs target |
| `rank=6` | Increase until eigengap is acceptable |

**Next:** [Walkthrough 2 — ResNet hook](02-resnet-vision-d4.md) for torchvision models.
