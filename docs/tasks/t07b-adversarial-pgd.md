# T7B — Adversarial / PGD perturbations

**Paper evidence:** [main.pdf](../../main.pdf) · [Block findings](../findings.html)

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

## Subtasks (paper)

<a id="t7b-train"></a>

### CIFAR ViT PGD arms (seed 7)

pmh_aniso TDI 0.878.

Preset: `t7b_pgd_d7`

<a id="t7b-eval"></a>

### Adversarial + geometry eval

Correct W +8.6 pp PGD@4 vs wrong_W.

Preset: `t7b_pgd_d7`

<a id="t7b-summary"></a>

### Bootstrap CI summary



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
