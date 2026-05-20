# Recipe: D7 — Style / alignment (exemplar T7A, LLM)

**Preset:** `t7a_style_d7` · **Lemma:** D7 · **Mode:** A (Jacobian on hidden states)

---

## Use this when

- Deployment shift is **formatting / tone / template**, not factual content.
- You can build **style pairs**: same semantic answer, multiple surface forms (bullets, JSON, verbose).
- You train with **Hugging Face** causal LM or encoder + `HFPMHTrainer` / `PMHLoss`.

**Not for:** new facts in answers, raw preference-only data without fixed `content_fixed`, or pure domain shift between corpora (use [Golden path G3](../GOLDEN_PATHS.md) / `robust_fit_text_domains`).

---

## Data contract

| Input | Format |
|-------|--------|
| Style JSONL | One object per line: `prompt`, `content_fixed`, `style_variants` map |
| Pooling | **Same** rule in estimate and train (e.g. mean of last hidden layer) |
| Sample in repo | `examples/data/style_pairs_sample.jsonl` (tiny — OK in git) |
| Your production file | **Outside repo** — see [DATA_POLICY.md](../DATA_POLICY.md) |

Example line:

```json
{
  "id": "ex1",
  "prompt": "Summarize the policy.",
  "content_fixed": "The refund window is 30 days.",
  "style_variants": {
    "bullets": "- 30-day refund window",
    "verbose": "Customers may return items within a thirty-day refund window."
  }
}
```

---

## Preset defaults

| Field | Value |
|-------|--------|
| `sigma_method` | D7 |
| `default_rank` | **128** |
| `estimate_kwargs` | `shrinkage` **0.1** |
| `pmh_config` | weight **0.7**, cap **0.3**, warmup **5** |
| `wrong_rank` | 128 |
| `arms` | `b0`, `matched`, `wrong_w`, `isotropic` |
| `application_mode` | `jacobian` |

!!! note "Wrong-W in paper"
    Paper wrong arm uses **content/semantic** Σ, not random QR — document which arm you run.

---

## Minimal code (estimate + train)

```bash
python examples/08_hf_style_d7.py
python examples/08_hf_style_d7.py --model-id YOUR_MODEL --jsonl /path/outside/repo/styles.jsonl
```

```python
from pmh import PMHLoss
from pmh.benchmark.presets import get_preset

p = get_preset("t7a_style_d7")
# Phase A: trainer.estimate_style(style_jsonl=path) or HFPMHTrainer.estimate_style
pmh = PMHLoss(artifact, p.pmh_config)
# training_step: total, parts = pmh.capped_total(task_loss, h)
```

Full HF `Trainer` integration: [Walkthrough 7](../walkthroughs/07-hf-trainer-d7-dpo.md).

---

## Developer / product paths

| Path | API |
|------|-----|
| Style JSONL (paper T7A) | `HFPMHTrainer.estimate_style` · preset `t7a_style_d7` |
| Two text corpora (same labels) | `robust_fit_text_domains` · [G3](../GOLDEN_PATHS.md) |
| Colab | [hf_two_corpora_first_run.ipynb](https://github.com/vishalstark512/matching-pmh/blob/main/notebooks/hf_two_corpora_first_run.ipynb) |

---

## Falsification arms

| Arm | Meaning |
|-----|---------|
| `b0` | No PMH |
| `matched` | Style-pair Gram |
| `wrong_w` | Content Σ (paper) — not random ⊥W |
| `isotropic` | Training trace_iso control |

---

## CPU wiring check

```bash
PMH_QUICK=1 python examples/08_hf_style_d7.py
```

Uses `HashEncoder` when no GPU — validates JSONL + loss path only.

---

## Related

| Doc | Purpose |
|-----|---------|
| [Walkthrough 6](../walkthroughs/06-llm-style-d7.md) | Full D7 guide |
| [gallery/nlp.md](../gallery/nlp.md) | NLP integration |
| [integrations-hf-trainer.md](../integrations-hf-trainer.md) | DPO / Trainer hook |

**Paper:** `Paper2/T7/task7A/`
