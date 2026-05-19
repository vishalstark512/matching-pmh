# Walkthrough 18: PMHTrainer quickstart — full guide

**At a glance**

| | |
|---|---|
| **API** | `PMHTrainer` — Phase A `estimate` + Phase B `fit` in one object |
| **Estimator** | Any D1–D7 via `nuisance=` and `estimate()` kwargs |
| **Script** | `examples/01_domain_shift_d4.py` (simplest) |
| **Best for** | “Just show me the default PyTorch integration.” |

Start here if [Walkthrough 1](01-pytorch-domain-d4.md) feels long — this page is the **shortest** complete path; WT1 has more D4 detail.

[Adaptation workbook](../ADAPTATION_WORKBOOK.md)

---

## Who this is for

- You have a **PyTorch** model and dataloaders.
- You want **one object** instead of manual `collect_features` + `PMHLoss`.
- You will run [Walkthrough 8](08-falsification-controls.md) before claiming results.

---

## Prerequisites

```bash
pip install matching-pmh torch
```

---

## Step 1 — Copy this skeleton

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    YOUR_MODEL,                      # nn.Module
    hook=YOUR_BACKBONE,                # submodule, str path, or encoder fn
    head=YOUR_HEAD,                    # optional
    nuisance="domain_shift",           # or "auto"
    rank=32,
    pmh_config=PMHConfig.balanced(),
    artifact_path="artifacts/YOUR_RUN/sigma.pt",
)

stats = trainer.fit(
    YOUR_TRAIN_LOADER,
    source_batches=YOUR_SOURCE_LOADER,
    target_batches=YOUR_TARGET_LOADER,
    epochs=20,
)
print(stats)
print("preflight:", trainer.artifact_.preflight, "method:", trainer.artifact_.method)
```

Replace `YOUR_*` — see [Walkthrough 1 worksheet](01-pytorch-domain-d4.md#adaptation-worksheet).

---

## Step 2 — `nuisance="auto"`

```python
trainer = PMHTrainer(
    model,
    hook=backbone,
    nuisance="auto",
    has_target_domain=True,
    has_target_labels=False,
    has_augmentation_modes=False,
)
```

Or call `suggest_nuisance()` first ([nuisance_types.md](../nuisance_types.md)).

---

## Step 3 — Other estimators (same trainer)

| Story | `fit()` kwargs |
|-------|----------------|
| D4 domain | `source_batches=`, `target_batches=` |
| D1 subspace | labeled source + target loaders |
| D3 augment | `augmentations=your_aug_fn` |
| D5 compositional | `nuisance_indices=[...]` |
| D6 temporal | `sequences_batches=` |
| D7 style | `style_jsonl="pairs.jsonl"` + HF model |

Full table: [ADAPT_YOUR_PIPELINE.md](../ADAPT_YOUR_PIPELINE.md)

---

## Step 4 — Hooks cheat sheet

| Model | `hook=` |
|-------|---------|
| Sub-module | `hook=model.backbone` |
| ResNet | `"avgpool"` or `model.avgpool` |
| timm ViT | `encoder_timm(model, layer="blocks")` |
| HF LM | `encoder_hf_hidden_states(model)` |

Details: [hooks.md](../hooks.md) · [Walkthrough 2](02-resnet-vision-d4.md)

---

## Step 5 — Run the bundled example

```bash
python examples/01_domain_shift_d4.py
```

---

## Step 6 — Credible comparison

```python
from pmh import compare_arms

compare_arms(
    trainer.artifact_,
    model_factory=...,
    setup_model=...,
    train_loader=...,
    val_loader=...,              # target domain
    report_dir="results/arms",
)
```

Sklearn features: [Walkthrough 17](17-compare-arms-your-pipeline.md).

---

## Verify success

- [ ] `artifact_` saved at `artifact_path`
- [ ] `preflight` not `fail`
- [ ] `stats` shows task + pmh losses
- [ ] Controls run ([WT 8](08-falsification-controls.md))

---

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Reused old `artifact_path` after data change | Delete or `reestimate=True` |
| Wrong hook | Same in estimate and train |
| No target batches for D4 | Pass `target_batches` |

---

## Next steps

- [1 — PyTorch D4 (detailed)](01-pytorch-domain-d4.md)
- [8 — Controls](08-falsification-controls.md)
- [Troubleshooting](../TROUBLESHOOTING.md)
