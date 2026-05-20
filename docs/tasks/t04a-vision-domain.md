# T4A — Vision domain shift (single-layer / ResNet)

**Source of truth:** `paper_code/T4/Task4A/FINAL.md`

**Lemma:** D4 · **Stack:** pytorch
**Nuisance key:** `domain_shift`

**Production change:** New camera, site, or geography; **same classes**.

**Notebook (Run All, built-in demo):** [t04a-vision-domain.ipynb](../../notebooks/tasks/t04a-vision-domain.ipynb)

```bash
pip install matching-pmh torch
# Open the notebook and Run All
```

## What this task achieved (headline)

> E1_multiscale Gram PMH on DomainNet real→sketch: **42.15%** test acc (+3.31 pp vs B0 38.84%).

| B0 | E1 | E1_multiscale |
|----|-----|---------------|
| 38.84% | 39.34% | **42.15%** |

**Paper preset:** `t4_domain_d4` · `from pmh.benchmark.presets import get_preset`

**Note:** Notebook = single-hook D4 with **class-aligned** Gram when loaders are `(x,y)`. Paper multiscale DomainNet: `paper_code/T4/Task4A/`.

## Subtasks (paper_code)

<a id="t4a-domainnet"></a>

### DomainNet real→sketch

E1_multiscale +3.31 pp test acc.

```bash
python paper_code/T4/Task4A/run_pipeline.py
```

Preset: `t4_domain_d4`

<a id="t4a-tdi"></a>

### Per-layer TDI geometry

Domain Gram on hook features.

```bash
python paper_code/T4/Task4A/tdi.py
```

Preset: `t4_domain_d4`

<a id="t4a-train"></a>

### B0 / E1 / E1_multiscale training



```bash
python paper_code/T4/Task4A/train.py
```

Preset: `t4_domain_d4`

## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="domain_shift"
```

## Do not use PMH when

New classes at deploy without relabeling.

## Replace demo data with yours

Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, and deploy holdout. Hook the backbone before your task head.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t04a-vision-domain"></a>
