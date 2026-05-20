# Walkthrough 7: Hugging Face Trainer + DPO (D7) — full guide


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden path G1–G4](../GOLDEN_PATHS.md#g3b) · **Route:** `pmh-train route --task llm_style_or_format` · **Step 5:** HF eval holdout + walkthrough 08
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**At a glance**

| | |
|---|---|
| **Stack** | `transformers` Trainer, LoRA, DPO-style preference data |
| **Scripts** | `examples/10_hf_trainer.py`, `11_dpo_lora_style_pmh.py` |
| **Estimator** | D7 style geometry + task loss |

[Walkthrough 6](06-llm-style-d7.md)

---

## Who this is for

You fine-tune LMs with **HF Trainer** (or similar) and already have **preference pairs** or **style JSONL** from Walkthrough 6.

---

## Your deployment shift sentence

*Same task, different writing style or template at deploy.* -> **D7** + HF Trainer.

---

## Prerequisites

```bash
pip install "matching-pmh[hf,hf-lora]"
```

---

## Step-by-step

1. Estimate style Σ with [Walkthrough 6](06-llm-style-d7.md) → `artifact.pt`.
2. Integrate `PMHCallback` or `HFPMHTrainer` (see `examples/10_hf_trainer.py`).
3. For DPO + style PMH jointly: `examples/11_dpo_lora_style_pmh.py`.

```bash
python examples/10_hf_trainer.py
python examples/11_dpo_lora_style_pmh.py
```

Use bundled `examples/data/preference_pairs_sample.jsonl` for wiring — replace with **your** JSONL (do not commit large files).

---

## Adaptation worksheet

| Example | Your project |
|---------|--------------|
| Sample JSONL | Your preference export |
| LoRA rank | Your PEFT config |
| `HFPMHTrainer` | Your `Trainer` subclass |

---

## Verify & controls

- [ ] Style artifact matches hidden-state pooling
- [ ] Falsification arms — [walkthrough 08](08-falsification-controls.md) on LM metric

---

## Next steps

- [integrations-hf-trainer.md](../GOLDEN_PATHS.md#g3b)
- [6 — Style JSONL](06-llm-style-d7.md)
