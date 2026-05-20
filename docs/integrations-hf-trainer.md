# Hugging Face `Trainer` (golden path G3b)

Use this when you **already** train with `transformers.Trainer` (PEFT, DPO, custom callbacks) and only need PMH in the loss.

| Path | Doc |
|------|-----|
| **G3b (this page)** | Subclass `Trainer` via `get_pmh_trainer()` |
| **G3** | [Golden paths — HFPMHTrainer](GOLDEN_PATHS.md#g3) · `robust_fit_text_domains` |
| **D7 style** | [Walkthrough 6](walkthroughs/06-llm-style-d7.md) |

Install: `pip install "matching-pmh[hf]"`

---

## Minimal pattern

```python
from transformers import TrainingArguments
from pmh import PMHConfig, SigmaTaskConfig, estimate_from_config
from pmh.integrations.hf_trainer import get_pmh_trainer

# Phase A — same representation as training
with torch.no_grad():
    h_src = model.body(batch_a["input_ids"])   # your hook → [B, d]
    h_tgt = model.body(batch_b["input_ids"])
artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=32), h_src, h_tgt)

PMHTrainer = get_pmh_trainer()
trainer = PMHTrainer.from_artifact(
    artifact,
    PMHConfig.balanced(),
    model=model,
    args=training_args,
    train_dataset=dataset,
    representation_fn=lambda m, inp: m.body(inp["input_ids"]),
)
trainer.train()
```

Template: [`hf_trainer_g3b_minimal.py`](https://github.com/vishalstark512/matching-pmh/blob/main/templates/matching-pmh-starter/hf_trainer_g3b_minimal.py)  
Example: `examples/10_hf_trainer.py`

---

## Custom representation

```python
from pmh.integrations.hf_trainer import default_representation_fn

def my_rep(model, inputs):
    out = model(**{**inputs, "output_hidden_states": True})
    return default_representation_fn(model, inputs, hidden_state_index=-1, pool="last")

trainer = PMHTrainer.from_artifact(..., representation_fn=my_rep)
```

---

## Without subclassing `Trainer`

```python
from pmh.integrations.hf_trainer import compute_pmh_training_loss
from pmh.training import PMHLoss

pmh_loss = PMHLoss(artifact, PMHConfig.balanced())
total, task, pmh = compute_pmh_training_loss(model, inputs, pmh_loss, representation_fn=my_rep)
```

---

## Style pairs (D7)

For same-content / different-format JSONL, estimate with `HFPMHTrainer.estimate_style` ([G3](GOLDEN_PATHS.md#g3)) or build artifact via [CUSTOM_GEOMETRY.md](CUSTOM_GEOMETRY.md), then pass to `PMHTrainer.from_artifact`.
