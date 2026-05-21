# T5A — Molecules / graphs (QM9-style)

**Paper evidence:** [main.pdf](../../main.pdf) · [Block findings](../findings.html)

**Lemma:** D5 · **Stack:** pytorch
**Nuisance key:** `compositional`

**Production change:** Conformer / position blocks move; **property label** fixed.

**Notebook (Run All, built-in demo):** [t05a-qm9-molecule.ipynb](../../notebooks/tasks/t05a-qm9-molecule.ipynb)

```bash
pip install matching-pmh torch
# Open the notebook and Run All
```

## What this task achieved (headline)

> E1 matched position PMH: clean MAE **24.921** (−0.155 vs B0); σ=0.2 Å noise MAE **47.415** vs B0.

| B0 MAE | E1 MAE | @ σ=0.2 |
|--------|--------|--------|
| 25.076 | **24.921** | **47.415** |

**Paper preset:** `t5_compositional_d5` · `from pmh.benchmark.presets import get_preset`

## Subtasks (paper)

<a id="t5a-pipeline"></a>

### QM9 train + noise + embedding eval

E1 clean MAE 24.921.

Preset: `t5_compositional_d5`

<a id="t5a-train"></a>

### MolGCN training



Preset: `t5_compositional_d5`

<a id="t5a-noise"></a>

### Position-noise eval sweep



Preset: `t5_compositional_d5`

## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="compositional"
```

## Do not use PMH when

Property definition changes at deploy.

## Replace demo data with yours

Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, and deploy holdout. Hook the backbone before your task head.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t05a-qm9-molecule"></a>
