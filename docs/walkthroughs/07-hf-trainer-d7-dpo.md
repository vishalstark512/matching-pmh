# Walkthrough 7: Hugging Face Trainer + DPO (D7) — full guide

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
- [ ] [Walkthrough 8](08-falsification-controls.md) on LM metric

---

## Next steps

- [integrations-hf-trainer.md](../integrations-hf-trainer.md)
- [6 — Style JSONL](06-llm-style-d7.md)
