# Walkthrough 2: ResNet-18 hook + D4

**Goal:** Wire PMH into a **standard torchvision classifier** without rewriting your training framework.

**Estimator:** D4 on penultimate ResNet features.  
**Script:** `examples/12_resnet_hook_d4.py`

---

## Prerequisites

```bash
pip install "matching-pmh[vision]"
```

---

## Architecture hook

```python
from torchvision.models import resnet18

backbone = resnet18(weights=None)
backbone.fc = nn.Identity()   # h in R^512

def encode(x):
    return backbone(x)        # [B, 512]
```

Your classifier head stays separate: `logits = head(encode(x))`.

---

## Step 1 — Name nuisance

Example: *“Target hospital images are darker; diagnoses (labels) unchanged.”* → D4.

---

## Step 2 — Collect source vs target features

```python
from pmh import collect_features, SigmaTaskConfig, estimate_from_config

backbone.eval()
h_src = collect_features(encode, source_loader, max_batches=50, device=device)
h_tgt = collect_features(encode, target_loader, max_batches=50, device=device)

artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=24), h_src, h_tgt)
artifact.save("artifacts/resnet_d4")
```

**Tip:** target loader should reflect **deployment** appearance (site B), not augmented training-only noise unless that is your deployment story.

---

## Step 3 — Train

Same as Walkthrough 1, with `h = encode(x)` during `backbone.train()`.

The example script uses **synthetic** images (random tensors + brightness shift) so you can run without downloading data:

```bash
python examples/12_resnet_hook_d4.py
python examples/12_resnet_hook_d4.py --pretrained --device cuda
```

If torchvision cannot load (torch ABI mismatch), the script prints a warning and uses a small ConvNet fallback—the **estimate → train** steps are unchanged.

---

## Step 4 — Real image folders

Replace synthetic iterators with:

```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

src_loader = DataLoader(ImageFolder("data/source", transform=train_tf), batch_size=32, shuffle=True)
tgt_loader = DataLoader(ImageFolder("data/target", transform=train_tf), batch_size=32, shuffle=True)
```

Use the **same** `transform` for estimation unless nuisance is explicitly in the transform (then D3 may apply).

---

## Step 5 — Controls + evaluation

- Evaluate **target-domain** accuracy / calibration, not only source.
- Run wrong-W and isotropic arms ([Walkthrough 8](08-falsification-controls.md)).

---

## Related

- Multi-layer PMH: [Walkthrough 4](04-multilayer-convnet.md)
- Office-31 + sklearn: [Walkthrough 3](03-office31-sklearn-d1.md)
