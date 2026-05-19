# Walkthrough 7: Hugging Face Trainer + DPO-style PMH

**Goal:** Keep `transformers.Trainer` (or TRL-style preference training) and add PMH on hidden states during fine-tune / LoRA.

**Estimator:** D7 artifact from style JSONL; optional preference JSONL for DPO loss.  
**Scripts:** `examples/10_hf_trainer.py` (minimal), `examples/11_dpo_lora_style_pmh.py` (full story)

---

## Prerequisites

```bash
pip install "matching-pmh[hf-lora]"
```

---

## Step 1 — Estimate style $\Sigma$ first

```bash
pmh-train estimate --config examples/configs/d7_style_estimate.json
```

Produces an artifact under `artifacts/` (path in config).

---

## Step 2 — Minimal `PMHTrainer` (toy)

```bash
python examples/10_hf_trainer.py
```

Key API:

```python
from pmh.integrations.hf_trainer import get_pmh_trainer

PMHTrainer = get_pmh_trainer()

trainer = PMHTrainer.from_artifact(
    artifact,
    PMHConfig(weight=0.2, cap_ratio=0.3),
    model=model,
    args=TrainingArguments(...),
    train_dataset=dataset,
    representation_fn=lambda m, batch: m.model(
        input_ids=batch["input_ids"],
        output_hidden_states=True,
    ).hidden_states[-1].mean(dim=1),
)
trainer.train()
```

`representation_fn` must match the layer used in Phase A.

---

## Step 3 — Qwen + LoRA + JSONL (paper Task 7A style)

```bash
# CPU / CI smoke (hash encoder)
python examples/11_dpo_lora_style_pmh.py

# GPU + real model
python examples/11_dpo_lora_style_pmh.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --train --lora
```

Data:

| File | Fields |
|------|--------|
| `style_pairs.jsonl` | `prompt`, `content_fixed`, `style_variants` |
| `preference_pairs.jsonl` | `prompt`, `chosen`, `rejected`, optional `style_variants` |

Defaults: `examples/data/*.jsonl`.

CLI mirror:

```bash
pmh-train run --config examples/configs/dpo_train_job.json
```

---

## Step 4 — Training objective stack

```
L_total = L_preference (e.g. DPO) + L_task (SFT) + capped PMH(h)
```

PMH does not replace preference learning; it **removes sensitivity along estimated style directions**.

---

## Step 5 — Evaluation

- Style robustness probes (rephrase, format shift)
- **Wrong-W** / **isotropic** Trainer runs with `mode=` on `PMHLoss` inside a custom loop, or separate jobs
- Do not claim alignment gains without control arms

---

## Adapt to your LM

| Example | Your model |
|---------|------------|
| `Qwen2.5-0.5B-Instruct` | Any `AutoModelForCausalLM` |
| LoRA targets | Your `peft` config |
| Tokenizer chat template | Set on tokenizer before collate |

See also [integrations-hf-trainer.md](../integrations-hf-trainer.md).
