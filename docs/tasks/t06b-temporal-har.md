# T6B — Time-series / HAR — sensor drift

**Paper evidence:** [main.pdf](../../main.pdf) · [Block findings](../findings.html)

**Lemma:** D6 · **Stack:** pytorch
**Nuisance key:** `temporal`

**Production change:** Sensor aging, device, session drift; **activity label** fixed.

**Notebook (Run All, built-in demo):** [t06b-temporal-har.ipynb](../../notebooks/tasks/t06b-temporal-har.ipynb)

```bash
pip install matching-pmh torch
# Open the notebook and Run All
```

## What this task achieved (headline)

> Matched PMH wins HAR stress 3.0: bal. acc **0.4099** vs baseline **0.2794** (3 seeds).

| baseline | PMH | wrong_W |
|----------|-----|--------|
| 0.2794 @ stress 3 | **0.4099** | fails geometry |

**Paper preset:** `t6_temporal_d6` · `from pmh.benchmark.presets import get_preset`

## Subtasks (paper)

<a id="t6b-multi-seed"></a>

### HAR multi-seed paper runs

PMH 0.4099 vs 0.2794 @ stress 3.

Preset: `t6_temporal_d6`

<a id="t6b-collect-w"></a>

### Collect W from baseline



Preset: `t6_temporal_d6`

<a id="t6b-stress"></a>

### Stress robustness eval



Preset: `t6_temporal_d6`

## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="temporal"
```

## Do not use PMH when

New activities only at deploy.

## Replace demo data with yours

Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, and deploy holdout. Hook the backbone before your task head.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t06b-temporal-har"></a>
