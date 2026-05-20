# T4B — Vision domain shift (multilayer FPN / U-Net)

**Source of truth:** `paper_code/T4/Task4B/FINAL.md`

**Lemma:** D4 · **Stack:** pytorch
**Nuisance key:** `domain_shift`

**Production change:** Texture **and** scene style shift together; same label map.

**Notebook (Run All, built-in demo):** [t04b-multilayer-vision.ipynb](../../notebooks/tasks/t04b-multilayer-vision.ipynb)

```bash
pip install matching-pmh torch
# Open the notebook and Run All
```

## What this task achieved (headline)

> E1_multiscale rare-5 Cityscapes mIoU **30.75%** (+11.1 pp vs B0 19.68%).

| B0 | E1 | E1_multiscale |
|----|-----|---------------|
| 19.68% mIoU | 19.99% | **30.75%** |

**Paper preset:** `t4_domain_d4` · `from pmh.benchmark.presets import get_preset`

**Note:** Notebook runs `PMHTrainer(train_mode='feature_diff')` + `estimate_multilayer` on demo loaders. Paper GTA5→Cityscapes: `paper_code/T4/Task4B/`.

## Subtasks (paper_code)

<a id="t4b-rare5"></a>

### Cityscapes rare-5 mIoU

E1_multiscale +11.1 pp mIoU.

```bash
python paper_code/T4/Task4B/run_rare5_all.py
```

Preset: `t4_domain_d4`

<a id="t4b-subset"></a>

### Build rare-5 training subset



```bash
python paper_code/T4/Task4B/build_subset.py
```

<a id="t4b-tdi"></a>

### Pixel-aligned per-layer TDI



```bash
python paper_code/T4/Task4B/tdi.py
```

Preset: `t4_domain_d4`

## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="domain_shift"
```

## Do not use PMH when

Single-layer hook enough — try T4A first.

## Replace demo data with yours

Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, and deploy holdout. Hook the backbone before your task head.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t04b-multilayer-vision"></a>
