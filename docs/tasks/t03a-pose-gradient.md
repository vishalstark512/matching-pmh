# T3A — Pose / keypoints — camera & studio shift

**Paper evidence:** [main.pdf](../../main.pdf) · [Block findings](../findings.html)

**Lemma:** D3 · **Stack:** pytorch
**Nuisance key:** `augmentation`

**Production change:** Camera, lighting, viewpoint change; **same keypoint indices**.

**Notebook (Run All, built-in demo):** [t03a-pose-gradient.ipynb](../../notebooks/tasks/t03a-pose-gradient.ipynb)

```bash
pip install matching-pmh torch
# Open the notebook and Run All
```

## What this task achieved (headline)

> E1_aniso subspace PMH on COCO pose: **54.49%** clean PCK@0.05 (+22.4 pp vs baseline 32.07%).

| baseline | E1_aniso |
|----------|----------|
| 32.07% PCK | **54.49%** PCK |
| — | 35.21% @ occ 0.40 |

## Subtasks (paper)

<a id="t3a-calibrate-w"></a>

### Calibrate occlusion subspace W

Gradient-SVD on COCO pose features.

<a id="t3a-train-arms"></a>

### Train baseline / E1 / E1_aniso / VAT

E1_aniso +22.4 pp PCK vs baseline.

<a id="t3a-eval-robust"></a>

### Robustness + embedding eval

Occlusion stress + drift metrics.

## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="augmentation"
```

## Do not use PMH when

Different skeleton or keypoint definitions at deploy.

## Replace demo data with yours

Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, and deploy holdout. Hook the backbone before your task head.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t03a-pose-gradient"></a>
