# T7A — LLM — format / tone / template

**Paper evidence:** [main.pdf](../../main.pdf) · [Block findings](../findings.html)

**Lemma:** D7 · **Stack:** hf
**Nuisance key:** `style`

**Production change:** Same facts, different **surface form** (JSON, bullets, tone).

**Notebook (Run All, built-in demo):** [t07a-llm-style.ipynb](../../notebooks/tasks/t07a-llm-style.ipynb)

```bash
pip install "matching-pmh[hf]"
# Open the notebook and Run All
```

## What this task achieved (headline)

> Matched $\Sigma_{\text{style}}$ RM: sycophancy **38.5%→13.5%**, style gap **2.199→0.803**; margin_pmh DPO Style TDI **1.836**.

| matched MC1 | sycophancy | style gap |
|-------------|------------|----------|
| 0.548 | **13.5%** | **0.803** |

**Paper preset:** `t7a_style_d7` · `from pmh.benchmark.presets import get_preset`

## Subtasks (paper)

<a id="t7a-rm-eval"></a>

### RM behavioral eval (TQA n=500)

Matched sycophancy 13.5%.

Preset: `t7a_style_d7`

<a id="t7a-dpo"></a>

### Geometric DPO + style geometry

margin_pmh Style TDI 1.836.

Preset: `t7a_style_d7`

<a id="t7a-pipeline"></a>

### Synthetic alignment pipeline



Preset: `t7a_style_d7`

## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="style"
```

## Do not use PMH when

Factual drift or new knowledge at deploy.

## Replace demo data with yours

Style-pair JSONL (same content, two surfaces) → `estimate_style_sigma` / D7 trainer.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t07a-llm-style"></a>
