# T7B — Adversarial / PGD perturbations

**Source of truth:** `paper_code/T7/task7B/FINAL.md`

**Lemma:** D7 · **Stack:** pytorch
**Nuisance key:** `style`

**Production change:** Small **input perturbations** are the production threat.

**Notebook (Run All, built-in demo):** [t07b-adversarial-pgd.ipynb](../../notebooks/tasks/t07b-adversarial-pgd.ipynb)

```bash
pip install matching-pmh torch
# Open the notebook and Run All
```

## What this task achieved (headline)

> pmh_aniso (PGD-W): TDI **0.878** (−19% vs baseline 1.090); clean **80.9%**.

| baseline TDI | pmh_aniso TDI | clean acc |
|--------------|---------------|----------|
| 1.090 | **0.878** | **80.9%** |

**Paper preset:** `t7b_pgd_d7` · `from pmh.benchmark.presets import get_preset`

## Subtasks (paper_code)

<a id="t7b-train"></a>

### CIFAR ViT PGD arms (seed 7)

pmh_aniso TDI 0.878.

```bash
python paper_code/T7/task7B/run_task7b.py
```

Preset: `t7b_pgd_d7`

<a id="t7b-eval"></a>

### Adversarial + geometry eval

Correct W +8.6 pp PGD@4 vs wrong_W.

```bash
python paper_code/T7/task7B/run_task7b.py
```

Preset: `t7b_pgd_d7`

<a id="t7b-summary"></a>

### Bootstrap CI summary



```bash
python paper_code/T7/task7B/run_task7b.py
```

## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="style"
```

## Do not use PMH when

Unbounded arbitrary shift with no perturbation model.

## Replace demo data with yours

Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, and deploy holdout. Hook the backbone before your task head.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t07b-adversarial-pgd"></a>
