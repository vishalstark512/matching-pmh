# Walkthrough 4: Multi-layer CNN features — full guide


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden path G1–G4](../GOLDEN_PATHS.md#g1) · **Route:** `pmh-train route --task vision_classification` · **Step 5:** compare_arms — paper multilayer
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**At a glance**

| | |
|---|---|
| **Estimators** | D3 (augmentation Gram) and/or D4 (domain) on feature maps |
| **Script** | `examples/07_vision_multilayer.py` |
| **API** | `MultiLayerPMHLoss`, `MultiPMHLoss` |

[Walkthrough 2](02-resnet-vision-d4.md) · [HYBRID_NUISANCE.md](../HYBRID_NUISANCE.md)

---

## Who this is for

CNNs where deployment shift lives in **early layers** (texture) and task signal in **late layers** — you want PMH on multiple feature maps, not only final `h`.

---

## Your deployment shift sentence

*“Deployment changes low-level texture; class semantics in late features unchanged.”*

---

## Step-by-step

1. Register hooks on layers `L1…Lk` → tensors `[B, C, H, W]` or pooled `[B, d]`.
2. Phase A: estimate per-layer or hybrid artifacts ([HYBRID_NUISANCE.md](../HYBRID_NUISANCE.md)).
3. Phase B: `MultiLayerPMHLoss` sums matched penalties per layer.

```bash
python examples/07_vision_multilayer.py
```

---

## Adaptation worksheet

| Example | Your project |
|---------|--------------|
| Toy CNN in script | Your U-Net / ResNet stages |
| Layer names | Your `named_modules()` paths |

---

## Verify & controls

- [ ] Each layer’s `d` matches artifact
- [ ] Falsification arms — [walkthrough 08](08-falsification-controls.md)

---

## Next steps

- [16 — D3 augmentations](16-augmentation-d3.md)
- [2 — ResNet single-layer](02-resnet-vision-d4.md)
