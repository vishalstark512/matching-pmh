# T2B — Medical imaging — hospital / scanner embedding shift

**Paper evidence:** [main.pdf](../../main.pdf) · [Block findings](../findings.html)

**Lemma:** D2 · **Stack:** pytorch
**Nuisance key:** `isotropic`

**Production change:** Hospital or scanner changes **appearance**; disease label unchanged.

**Notebook (Run All, built-in demo):** [t02b-chexpert-isotropic.ipynb](../../notebooks/tasks/t02b-chexpert-isotropic.ipynb)

```bash
pip install matching-pmh torch
# Open the notebook and Run All
```

## What this task achieved (headline)

> Chest X-ray E1: best saliency (0.723) and ~9× lower embedding drift vs B0.



**Paper preset:** `t2b_chexpert_isotropic` · `from pmh.benchmark.presets import get_preset`

## Subtasks (paper)

<a id="t2b-pneumonia-clean"></a>

### Pneumonia chest X-ray — clean test (B0 vs E1 arms)

B0 best clean (91.7%); E1_no_pmh / E1 trade clean for shift robustness.

Preset: `t2b_chexpert_isotropic`

<a id="t2b-robust-shift"></a>

### Robust eval — Gaussian + acquisition shifts

E1_no_pmh best worst-shift; E1 best saliency + embedding drift.

Preset: `t2b_chexpert_isotropic`

<a id="t2b-saliency"></a>

### Saliency stability under noise

E1 (PMH) 0.723 vs B0 0.560 cosine stability.

## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="isotropic"
```

## Do not use PMH when

New disease labels at deploy only.

## Replace demo data with yours

Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, and deploy holdout. Hook the backbone before your task head.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t02b-chexpert-isotropic"></a>
