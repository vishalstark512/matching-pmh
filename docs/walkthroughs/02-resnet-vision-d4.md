# Walkthrough 2: ResNet / torchvision + D4 — full guide


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden path G1–G4](../GOLDEN_PATHS.md#g1) · **Route:** `pmh-train route --task vision_classification` · **Step 5:** evaluate_robust_fit
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**At a glance**

| | |
|---|---|
| **Estimator** | D4 on ResNet penultimate features (512-d for ResNet-18) |
| **Stack** | torchvision + PyTorch |
| **Script** | `examples/12_resnet_hook_d4.py` |
| **Related** | [hooks.md](../INTEGRATE.md) · [Walkthrough 1](01-pytorch-domain-d4.md) |

---

## Who this is for

You have a **torchvision** classifier (ResNet, etc.) and domain shift between **image folders** or datasets — same labels, different visual appearance.

Use [Walkthrough 12](12-vit-cls-d4.md) for ViT/timm instead.

---

## Prerequisites

```bash
pip install "matching-pmh[vision]"
```

---

## Your deployment shift sentence

*“Target-site photos are darker / different camera; object class unchanged.”* → **D4**.

---

## Step 1 — Wire the hook (512-d for ResNet-18)

```python
import torch.nn as nn
from torchvision.models import resnet18

backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # or None
backbone.fc = nn.Identity()

def encode(x):
    return backbone(x)   # YOUR: [B, 512]
head = nn.Linear(512, YOUR_NUM_CLASSES)
```

---

## Step 2 — Data loaders (your folders)

```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

YOUR_TRANSFORM = ...  # same for estimate unless nuisance IS the transform

src_loader = DataLoader(ImageFolder("YOUR_PATH/source", transform=YOUR_TRANSFORM), ...)
tgt_loader = DataLoader(ImageFolder("YOUR_PATH/target", transform=YOUR_TRANSFORM), ...)
train_loader = DataLoader(ImageFolder("YOUR_PATH/train", ...), ...)
```

**Do not commit** image folders — [DATA_POLICY.md](../DOCS_GUIDE.md).

---

## Step 3 — `PMHTrainer` (recommended)

```python
from pmh import PMHTrainer, PMHConfig

model = nn.Sequential(backbone, head)  # or your module tree
trainer = PMHTrainer(
    model,
    hook=backbone,                    # or hook="avgpool"
    head=head,
    nuisance="domain_shift",
    pmh_config=PMHConfig.balanced(),
    artifact_path="artifacts/resnet_d4.pt",
)
trainer.fit(train_loader, source_batches=src_loader, target_batches=tgt_loader, epochs=20)
```

---

## Step 4 — Manual estimate (optional)

```python
from pmh import collect_features, SigmaTaskConfig, estimate_from_config

backbone.eval()
h_src = collect_features(encode, src_loader, max_batches=50, device=YOUR_DEVICE)
h_tgt = collect_features(encode, tgt_loader, max_batches=50, device=YOUR_DEVICE)
artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=24), h_src, h_tgt)
```

---

## Run the example (synthetic images)

```bash
python examples/12_resnet_hook_d4.py
python examples/12_resnet_hook_d4.py --pretrained --device cuda
```

Runs without your data (random tensors + brightness shift). Use it to verify install, then swap loaders.

---

## Adaptation worksheet

| Example | Your project |
|---------|--------------|
| Synthetic brightness | Real site A / site B folders |
| `resnet18` | `resnet50`, EfficientNet, … |
| `rank=24` | Tune; check `preflight` |
| `12_resnet_hook_d4.py` | Copy structure into your training repo |

---

## Verify success

- [ ] `preflight` pass/marginal
- [ ] Target-domain val accuracy tracked
- [ ] Falsification arms run — [walkthrough 08](08-falsification-controls.md)

---

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Different transforms on src vs tgt | Same `transform` for D4 estimate unless shift is augmentation (then D3) |
| Eval on source only | Target val set |
| `fc` not removed | `h` must be 512-d penultimate, not logits |

---

## Next steps

- [4 — Multi-layer CNN](04-multilayer-convnet.md)
- [3 — Frozen features + sklearn](03-office31-sklearn-d1.md)
- [8 — Controls](08-falsification-controls.md)
