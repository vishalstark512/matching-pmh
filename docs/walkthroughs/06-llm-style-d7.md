# Walkthrough 6: LLM style / format (D7) — full guide

**At a glance**

| | |
|---|---|
| **Estimator** | D7 — style Gram from JSONL pairs |
| **Script** | `examples/08_hf_style_d7.py` |
| **Sample data** | `examples/data/style_pairs_sample.jsonl` (tiny, OK in git) |

[Walkthrough 7](07-hf-trainer-d7-dpo.md) · [gallery/nlp.md](../gallery/nlp.md) · [Paper presets](paper-presets-by-block.md)

**Paper preset:** `t7a_style_d7` — D7 rank 128, shrinkage 0.1, `PMHConfig(weight=0.7, cap_ratio=0.3, warmup_epochs=5)`. **Recipe:** [T7A one-pager](../recipes/t7a-style-d7.md).

---

## Who this is for

LLM deployment shift is **formatting / tone / template**, not factual content — you can write **style variants of the same answer**.

---

## Your nuisance sentence

*“Bullets vs paragraphs vs JSON wrapper; semantic answer unchanged.”*

---

## Step 1 — Prepare JSONL (your file, not in repo)

Each line:

```json
{
  "id": "ex1",
  "prompt": "YOUR_PROMPT",
  "content_fixed": "THE SAME FACTUAL ANSWER",
  "style_variants": {
    "bullets": "- point one\n- point two",
    "verbose": "Longer phrasing of the same facts..."
  }
}
```

Copy structure from `examples/data/style_pairs_sample.jsonl`.

---

## Step 2 — Same pooling for estimate and train

```python
out = model(..., output_hidden_states=True)
h = out.hidden_states[-1].mean(dim=1)   # YOUR: fixed rule
```

---

## Step 3 — Estimate

```bash
python examples/08_hf_style_d7.py
python examples/08_hf_style_d7.py --model-id YOUR_MODEL --jsonl YOUR_FILE.jsonl
```

Or `pmh-train estimate --config examples/configs/d7_style_estimate.json` (edit paths).

---

## Step 4 — Train with PMH

```python
from pmh import PMHLoss
from pmh.benchmark.presets import get_preset

p = get_preset("t7a_style_d7")
pmh = PMHLoss(artifact, p.pmh_config)
# in step: total, _ = pmh.capped_total(task_loss, h)
```

Full Trainer: [Walkthrough 7](07-hf-trainer-d7-dpo.md).

---

## What D7 is / is not

| D7 is for | D7 is not for |
|-----------|----------------|
| Format, tone, bullets | Changing facts in `content_fixed` |
| Style pairs with fixed semantics | Raw preference-only data without style structure |

---

## Run toy (no GPU)

```bash
python examples/08_hf_style_d7.py
```

Uses `HashEncoder` when no GPU — wiring check only.

---

## Adaptation worksheet

| Example | Your project |
|---------|--------------|
| `style_pairs_sample.jsonl` | Your production JSONL export |
| Mean pool last layer | Your chosen `h` |

---

## Verify & controls

- [ ] Same tokenizer + pooling in Phase A/B
- [ ] Falsification arms — [walkthrough 08](08-falsification-controls.md)

---

## Next steps

- [7 — HF Trainer + DPO](07-hf-trainer-d7-dpo.md)
- [hooks.md](../hooks.md) — `encoder_hf_hidden_states`
