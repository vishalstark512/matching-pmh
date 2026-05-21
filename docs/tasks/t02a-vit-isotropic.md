# T2A — ViT / image classifier — isotropic sensor noise

**Paper evidence:** [main.pdf](../../main.pdf) · [Block findings](../findings.html)

**Lemma:** D2 · **Stack:** pytorch
**Nuisance key:** `isotropic`

**Production change:** Small **sensor / embedding noise**; class semantics unchanged.

**Notebook (Run All, built-in demo):** [t02a-vit-isotropic.ipynb](../../notebooks/tasks/t02a-vit-isotropic.ipynb)

```bash
pip install matching-pmh torch
# Open the notebook and Run All
```

## What this task achieved (headline)

> Isotropic PMH on ViT-B/16: +4.29 pp mean ImageNet-C, TDI −58% at σ=0.10.



**Paper preset:** `t2a_vit_isotropic` · `from pmh.benchmark.presets import get_preset`

## Subtasks (paper)

<a id="t2a-vit-imagenet"></a>

### ImageNet ViT-B/16 + isotropic PMH (Type 2A)

PMH ≈ ERM clean; +4.29 pp mean ImageNet-C; TDI −58% at σ=0.10.

Preset: `t2a_vit_isotropic`

<a id="t2a-geometry-tdi"></a>

### TDI / Jacobian probes (label-free)

Layer-averaged CLS displacement under input Gaussian.

<a id="t2a-imagenet-c"></a>

### ImageNet-C transfer (15 corruptions, severity 3)

Train on Gaussian only; largest gains on noise/frost/blur.

## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="isotropic"
```

## Do not use PMH when

Large domain shift without D2 setup — use T4 instead.

## Replace demo data with yours

Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, and deploy holdout. Hook the backbone before your task head.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t02a-vit-isotropic"></a>
