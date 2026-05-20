# Walkthrough 12: ViT / CLS token + D4 — full guide

**At a glance**

| | |
|---|---|
| **Estimator** | D4 on CLS or pooled patch tokens |
| **Script** | `examples/14_vit_cls_d4.py` |
| **Hooks** | `encoder_timm`, `encoder_hf_hidden_states` |

[hooks.md](../hooks.md) · [Walkthrough 2](02-resnet-vision-d4.md) · [Paper presets](paper-presets-by-block.md)

**Paper preset (T2A isotropic):** `t2a_vit_isotropic` — D2, σ=0.10, Jacobian mode, arms `b0` + `matched` only.

---

## Who this is for

Vision Transformers (timm, HF ViT) with domain shift — hook at **CLS** or mean patch token.

---

## Your nuisance sentence

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
