# Walkthrough 10: PyTorch Lightning — full guide


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden path G1–G4](../GOLDEN_PATHS.md#g1b) · **Route:** `pmh-train route --task pytorch_lightning` · **Step 5:** compare_arms after Lightning integrate
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**At a glance**

| | |
|---|---|
| **Stack** | Lightning `LightningModule` |
| **Script** | `examples/09_lightning_module.py` |
| **Pattern** | Estimate once → `PMHLoss` in `training_step` |

[integrations-lightning.md](../GOLDEN_PATHS.md#g1b)

---

## Who this is for

Your training loop is already a `LightningModule` — add PMH without rewriting the trainer.

---

## Your deployment shift sentence

*Lightning project: train hospital A, deploy hospital B; hook on layer4 or CLS.* -> **D4** typical.

---

## Step-by-step

1. Phase A in `setup()` or `on_fit_start`: collect features from `validation` loaders (source/target).
2. Save `artifact`.
3. In `training_step`: `h = self.encoder(x)`; `pmh.capped_total(task_loss, h)`.

```bash
pip install "matching-pmh[lightning]"
python examples/09_lightning_module.py
```

Open `examples/09_lightning_module.py` and copy the `PMHLoss` hook into **your** `LightningModule`.

---

## Adaptation worksheet

| Example module | Your `LightningModule` |
|----------------|------------------------|
| `training_step` | Same hook `h` as estimate |

---

## Next steps

- [1 — PyTorch D4](01-pytorch-domain-d4.md)
- [18 — PMHTrainer](18-pmh-trainer-quickstart.md) (alternative to manual Lightning hooks)
