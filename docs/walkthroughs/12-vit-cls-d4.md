# Walkthrough 12: ViT / CLS token + D4 — full guide


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden path G1–G4](../GOLDEN_PATHS.md#g1) · **Route:** `pmh-train route --task vision_classification` · **Step 5:** compare_arms
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**At a glance**

| | |
|---|---|
| **Estimator** | D4 on CLS or pooled patch tokens |
| **Script** | `examples/14_vit_cls_d4.py` |
| **Hooks** | `encoder_timm`, `encoder_hf_hidden_states` |

[hooks.md](../INTEGRATE.md) · [Walkthrough 2](02-resnet-vision-d4.md) · [Paper presets](paper-presets-by-block.md)

**Paper preset (T2A isotropic):** `t2a_vit_isotropic` — D2, σ=0.10, Jacobian mode, arms `b0` + `matched` only. **Recipe:** [T2A one-pager](../PAPER_ALIGNMENT.md).

---

## Who this is for

Vision Transformers (timm, HF ViT) with domain shift — hook at **CLS** or mean patch token.

---

## Your deployment shift sentence

*“New camera / site; class label unchanged.”*

---

## Step-by-step

```python
from pmh.hooks import encoder_timm
from pmh import PMHTrainer
from pmh.benchmark.presets import get_preset

p = get_preset("t2a_vit_isotropic")
hook = encoder_timm(YOUR_VIT, layer="blocks", pool="cls")
trainer = PMHTrainer(
    model,
    hook=hook,
    nuisance=p.nuisance,       # isotropic / D2
    pmh_config=p.pmh_config,
)
trainer.fit(train_loader, source_batches=src, target_batches=tgt, epochs=20)
```

For **domain** shift (D4) instead of isotropic noise, use preset `t4_domain_d4` — [Walkthrough 1](01-pytorch-domain-d4.md).

```bash
python examples/14_vit_cls_d4.py
```

---

## Adaptation worksheet

| Example | Your project |
|---------|--------------|
| timm ViT-S | Your HF ViT checkpoint |
| `pool="cls"` | `mean` patch pool |

---

## Next steps

- [2 — ResNet](02-resnet-vision-d4.md)
- [16 — D3 aug](16-augmentation-d3.md)
