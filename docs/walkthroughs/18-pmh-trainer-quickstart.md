# Walkthrough 18: PMHTrainer quickstart

**Goal:** One object for Phase A + Phase B on your PyTorch model.

## 1. Install

```bash
pip install matching-pmh torch
```

## 2. Minimal script

See `examples/01_domain_shift_d4.py`:

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    model,
    hook=backbone,
    head=head,
    nuisance="domain_shift",  # or "auto"
    pmh_config=PMHConfig.balanced(),
    artifact_path="artifacts/sigma.pt",
)
trainer.fit(
    train_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    epochs=20,
)
print(trainer.artifact_.preflight)
```

## 3. Hooks

| Model | `hook=` |
|-------|---------|
| Sub-module | `hook=backbone` (module or path) |
| ResNet | `"avgpool"` |
| timm ViT | `encoder_timm(model)` |
| HF LM | `encoder_hf_hidden_states(model)` |

Details: [hooks.md](../hooks.md)

## 4. Credible comparison

```python
from pmh import compare_arms

compare_arms(
    trainer.artifact_,
    model_factory,
    setup_model,
    train_loader,
    val_loader,
    report_dir="results/arms",
)
```

Or sklearn features: [17-compare-arms-your-pipeline.md](17-compare-arms-your-pipeline.md)
