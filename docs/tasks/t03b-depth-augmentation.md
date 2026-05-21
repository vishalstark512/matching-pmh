# T3B — Depth estimation — photometric shift

**Paper evidence:** [main.pdf](../../main.pdf) · [Block findings](../findings.html)

**Lemma:** D3 · **Stack:** pytorch
**Nuisance key:** `augmentation`

**Production change:** Lighting/texture shift; **depth target meaning** unchanged.

**Notebook (Run All, built-in demo):** [t03b-depth-augmentation.ipynb](../../notebooks/tasks/t03b-depth-augmentation.ipynb)

```bash
pip install matching-pmh torch
# Open the notebook and Run All
```

## What this task achieved (headline)

> E1_aniso beats E1 on hard photometric depth stress (combined_hard AbsRel **0.2152** vs 0.2191).

| baseline | E1 | E1_aniso |
|----------|-----|----------|
| 0.2033 AbsRel | 0.1951 | **0.2152** (photo hard) |

**Paper preset:** `t3b_depth_d3` · `from pmh.benchmark.presets import get_preset`

## Subtasks (paper)

<a id="t3b-pipeline"></a>

### NYU depth pipeline (train + calibrate)

E1_aniso wins photometric hard stress.

Preset: `t3b_depth_d3`

<a id="t3b-calibrate"></a>

### Photometric subspace calibration

Aug-delta Gram rank 32.

Preset: `t3b_depth_d3`

<a id="t3b-wrong-w"></a>

### E1_wrong control (random W)

Strengthening / falsification arm.

## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="augmentation"
```

## Do not use PMH when

Different depth semantics or scale definition at deploy.

## Replace demo data with yours

Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, and deploy holdout. Hook the backbone before your task head.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t03b-depth-augmentation"></a>
