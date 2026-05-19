# Walkthrough 12: ViT / CLS token + D4

**Paper block:** T2 (ViT-B/16) — patch embedding is the input projection; deployment nuisance often acts on appearance while semantics stay.

**Goal:** Hook PMH on the **CLS token** (or pooled patch tokens) with domain-shift estimation D4.

**Script:** `examples/14_vit_cls_d4.py`

---

## Hook point

```python
h = model.encode(images)  # CLS vector [B, d]
```

For `timm` ViT:

```python
import timm
vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
h = vit.forward_features(x)  # often [B, d] CLS
```

Use the **same** function in Phase A and B.

---

## Steps

1. **Nuisance:** deployment lighting / site (D4).
2. **Phase A:** `collect_features(encode, source_loader)` vs target loader.
3. **Preflight:** check `artifact.preflight`.
4. **Phase B:** `PMHLoss` on CLS during fine-tune.
5. **Controls:** wrong-W, isotropic ([Walkthrough 8](08-falsification-controls.md)).

---

## Run

```bash
python examples/14_vit_cls_d4.py
```

---

## Adapt

| Example | Production |
|---------|------------|
| `PatchViTEncoder` | `timm` / HF ViT |
| Brightness synthetic shift | ImageNet-C / site B folder |
| `rank=16` | Tune via eigengap |

**Related:** [Multi-layer CNN](04-multilayer-convnet.md) if nuisance is mid-level texture.
