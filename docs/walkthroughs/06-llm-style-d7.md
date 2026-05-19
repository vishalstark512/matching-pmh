# Walkthrough 6: LLM style Gram (D7)

**Goal:** Penalize sensitivity to **style / format** while keeping **semantic content** fixed (alignment, formatting robustness).

**Estimator:** D7 (`for_alignment`) from style-pair JSONL.  
**Script:** `examples/08_hf_style_d7.py`

---

## Prerequisites

```bash
pip install "matching-pmh[hf]"   # real LM
# CPU toy encoder works without GPU
```

---

## Step 1 — Prepare JSONL

Each line:

```json
{
  "id": "ex1",
  "prompt": "Summarize the paper.",
  "content_fixed": "The method estimates Sigma_task and adds matched PMH.",
  "style_variants": {
    "bulleted": "- Estimates Sigma_task\n- Adds PMH",
    "verbose": "In detail, the method estimates deployment nuisance covariance..."
  }
}
```

Bundled sample: `examples/data/style_pairs_sample.jsonl`.

Optional `semantic_control` field for negative controls in research scripts.

---

## Step 2 — Choose representation $h$

Standard choice: **mean-pooled last hidden state** `[B, d]` from your causal LM.

```python
out = model(input_ids=..., attention_mask=..., output_hidden_states=True)
h = out.hidden_states[-1].mean(dim=1)
```

Use the **same** pooling in estimation and training.

---

## Step 3 — Estimate (Phase A)

```bash
python examples/08_hf_style_d7.py
python examples/08_hf_style_d7.py --model-id Qwen/Qwen2.5-0.5B-Instruct --jsonl path/to/style_pairs.jsonl
```

Or CLI:

```bash
pmh-train estimate --config examples/configs/d7_style_estimate.json
```

Edit the config to point at your JSONL and model id.

---

## Step 4 — Phase B: add `PMHLoss` or `PMHTrainer`

After `artifact.save(...)`:

```python
pmh = PMHLoss(artifact, PMHConfig(weight=0.2, cap_ratio=0.3, warmup_epochs=1))
# inside training step on h from last hidden state
```

Full Trainer path: [Walkthrough 7](07-hf-trainer-d7-dpo.md).

---

## Step 5 — What D7 is / is not

| D7 is for | D7 is not for |
|-----------|----------------|
| Format, tone, bullet vs prose | Changing factual content |
| Style pairs with fixed semantics | General preference-only data without style structure |

Preference pairs (`chosen`/`rejected`) go with DPO training; style JSONL drives $\Sigma_{\mathrm{task}}$.

---

## Run toy (no GPU)

```bash
python examples/08_hf_style_d7.py
```

Uses `HashEncoder` for CI-friendly output.
