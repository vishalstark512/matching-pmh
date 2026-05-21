# T5B — Code models — token-group shift

**Paper evidence:** [main.pdf](../../main.pdf) · [Block findings](../findings.html)

**Lemma:** D5 · **Stack:** pytorch
**Nuisance key:** `compositional`

**Production change:** Imports/comments/identifiers change; **downstream label** fixed.

**Notebook (Run All, built-in demo):** [t05b-code-tokens.ipynb](../../notebooks/tasks/t05b-code-tokens.ipynb)

```bash
pip install "matching-pmh[hf]"
# Open the notebook and Run All
```

## What this task achieved (headline)

> E1 identifier PMH: rename_bacc_ratio **0.9383** vs B0 **0.8297**.

| B0 | E1 | E1S (wrong blocks) |
|----|-----|------------------|
| 0.8297 | **0.9383** | 0.7379 (fails) |

**Paper preset:** `t5_compositional_d5` · `from pmh.benchmark.presets import get_preset`

## Subtasks (paper)

<a id="t5b-pipeline"></a>

### Clone detection train + eval

E1 rename_bacc 0.9383.

Preset: `t5_compositional_d5`

<a id="t5b-train"></a>

### CodeBERT clone training



Preset: `t5_compositional_d5`

<a id="t5b-eval"></a>

### Rename / reformat eval suites



Preset: `t5_compositional_d5`

## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="compositional"
```

## Do not use PMH when

New task or label at deploy.

## Replace demo data with yours

Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, and deploy holdout. Hook the backbone before your task head.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t05b-code-tokens"></a>
