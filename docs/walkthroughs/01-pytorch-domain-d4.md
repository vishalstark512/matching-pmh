# Walkthrough 1: PyTorch domain shift (D4) — full guide

**At a glance**

| | |
|---|---|
| **Estimator** | D4 — domain Gram (source vs target features, labels optional) |
| **Stack** | Any PyTorch `nn.Module` |
| **Script** | `examples/01_domain_shift_d4.py` |
| **Time** | ~2 min CPU (quick), ~10 min default |
| **API** | `PMHTrainer` (recommended) or manual `collect_features` + `PMHLoss` |

[Adaptation workbook](../ADAPTATION_WORKBOOK.md) · [Choose setup](../CHOOSE_YOUR_SETUP.md) · [Paper presets](paper-presets-by-block.md)

**Paper preset:** `t4_domain_d4` (D4, rank 64, weight 0.5 / cap 0.3, warmup 2). **Recipe card:** [T4 one-pager](../recipes/t4-domain-d4.md) · `pmh-train list-presets`.

---

## Who this is for

Use this walkthrough when:

- You train in **PyTorch** (custom model, Lightning, etc.).
- Deployment looks like a **different site, device, corpus, or sensor**, but **class labels still mean the same thing**.
- You can sample batches from a **source** distribution (train) and a **target** distribution (deploy or unlabeled target pool).

Pick another guide if:

- You only have **frozen `.npy` features** → [Walkthrough 3](03-office31-sklearn-d1.md).
- Shift is **known augmentations** → [Walkthrough 16](16-augmentation-d3.md).
- Shift is **LLM style/format** → [Walkthrough 6](06-llm-style-d7.md).

---

## Prerequisites

```bash
pip install matching-pmh torch
```

You need:

| Requirement | Example |
|-------------|---------|
| `model` | `nn.Module` with encoder + head |
| `hook` | Submodule where `h` has shape `[batch, d]` |
| `source_batches` | Loader for domain A |
| `target_batches` | Loader for domain B (labels not required for D4 estimate) |
| `train_loader` | Your usual supervised training data |

---

## Your nuisance sentence (write this first)

Examples that fit **D4**:

- *“Images come from a new hospital scanner; the disease label definition is unchanged.”*
- *“Audio is recorded on a different microphone; the word transcript is still correct.”*
- *“Documents are from a new customer segment; the intent class is the same.”*

Counter-examples (do **not** use D4 without reframing):

- *“New classes appear at test time.”* → label shift, not PMH nuisance geometry.
- *“I want robustness to any perturbation.”* → name the perturbation (D2/D3) or use controls to test claims.

---

## What the example script does

File: `examples/01_domain_shift_d4.py`

| Code block | Purpose |
|------------|---------|
| `Backbone` + `head` | Tiny MLP so the script runs without your data |
| `_loader(n, shift)` | Synthetic source (`shift=0`) vs target (`shift=0.8`) domains |
| `PMHTrainer(..., nuisance="domain_shift")` | Selects **D4** |
| `trainer.fit(..., source_batches=, target_batches=)` | Phase A estimate + Phase B train |
| `artifact_path="artifacts/demo_d4.pt"` | Saves `Sigma_task` for reuse |

**Important:** the script uses **one** `train_loader` and separate source/target loaders only for estimation. Your project may use the same loaders or different splits — both are fine if the **target loader matches deployment**.

---

## Step-by-step: adapt to your project

### Step 1 — Point `hook` at your representation

```python
# YOUR code — pick one:
hook = your_model.backbone          # nn.Module
hook = "layer4"                     # string path on model
hook = lambda x: your_model.encode(x)
```

Rule: `h = hook(x)` must be `[B, d]` with fixed `d` for all batches.

### Step 2 — Build `PMHTrainer`

```python
from pmh import PMHTrainer
from pmh.benchmark.presets import get_preset

p = get_preset("t4_domain_d4")
trainer = PMHTrainer(
    YOUR_MODEL,
    hook=YOUR_HOOK,
    head=YOUR_CLASSIFIER,              # optional if head inside model
    nuisance=p.nuisance,               # domain_shift / D4
    rank=p.default_rank,               # 64 in paper T4
    pmh_config=p.pmh_config,
    artifact_path="artifacts/YOUR_EXPERIMENT/sigma.pt",
)
```

Compare falsification arms after training:

```python
from pmh import compare_arms

compare_arms(
    YOUR_MODEL,
    YOUR_TRAIN_LOADER,
    source_batches=YOUR_SOURCE_LOADER,
    target_batches=YOUR_TARGET_LOADER,
    hook=YOUR_HOOK,
    preset="t4_domain_d4",
    include_geometry=True,
)
```

### Step 3 — Fit (estimate + train)

```python
stats = trainer.fit(
    YOUR_TRAIN_LOADER,
    source_batches=YOUR_SOURCE_LOADER,
    target_batches=YOUR_TARGET_LOADER,
    epochs=YOUR_EPOCHS,
    max_steps_per_epoch=None,          # optional cap for smoke tests
)
print(stats)
print("preflight:", trainer.artifact_.preflight)
```

### Step 4 — Manual path (if you cannot use `PMHTrainer`)

Phase A:

```python
from pmh import SigmaTaskConfig, collect_features, estimate_from_config

YOUR_BACKBONE.eval()
h_src = collect_features(YOUR_ENCODE_FN, YOUR_SOURCE_LOADER, max_batches=50)
h_tgt = collect_features(YOUR_ENCODE_FN, YOUR_TARGET_LOADER, max_batches=50)
artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=32), h_src, h_tgt)
artifact.save("artifacts/YOUR_EXPERIMENT/sigma.pt")
```

Phase B:

```python
from pmh import PMHLoss, PMHConfig

pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2))
YOUR_BACKBONE.train()
for epoch in range(YOUR_EPOCHS):
    pmh.set_epoch(epoch)
    for x, y in YOUR_TRAIN_LOADER:
        opt.zero_grad()
        h = YOUR_ENCODE_FN(x)
        task_loss = YOUR_CRITERION(YOUR_HEAD(h), y)
        total, pmh_term = pmh.capped_total(task_loss, h)
        total.backward()
        opt.step()
```

---

## Run the example

```bash
# Quick smoke (~30 s)
set PMH_QUICK=1
python examples/01_domain_shift_d4.py

# Default
python examples/01_domain_shift_d4.py
```

**Expected output (similar to):**

```
done  task=0.xxxx  pmh=0.xxxx
preflight=pass  method=D4
```

(`pass` or `marginal` is OK; `fail` → see [Troubleshooting](../TROUBLESHOOTING.md).)

---

## Adaptation worksheet

| In the example | In your project |
|----------------|-----------------|
| `Backbone` MLP | Your encoder module |
| `shift=0.8` target loader | Target hospital / site / corpus loader |
| `rank=6` | 16–32; increase if `marginal` |
| `TensorDataset` synthetic | `ImageFolder`, `Dataset`, HF collator, … |
| `hook=backbone` | Same tensor you would use for linear probe |

---

## Verify success

- [ ] `trainer.artifact_` exists and `method == "D4"`.
- [ ] `preflight` is `pass` or `marginal` (not `fail`).
- [ ] Training logs show **both** task and PMH losses changing.
- [ ] Target-domain metric improves vs B0 (after controls).

---

## Controls (required for claims)

Train **matched**, **wrong_w**, and **isotropic** (same rank):

→ [Walkthrough 8 — Falsification controls](08-falsification-controls.md)

```python
from pmh import compare_arms
compare_arms(trainer.artifact_, model_factory, setup_model, train_loader, val_loader, ...)
```

---

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Different hook in estimate vs train | Use identical `hook` / layer |
| Target loader is only augmented train data | Target loader should reflect **deployment** |
| `preflight=fail` | More target batches, higher `rank`, try D1 if labels on both domains |
| PMH always zero | `warmup_epochs`, `weight`, ensure `h.requires_grad` |
| Claiming win without wrong-W arm | Run [Walkthrough 8](08-falsification-controls.md) |

---

## Next steps

| Goal | Walkthrough |
|------|-------------|
| torchvision ResNet | [2 — ResNet D4](02-resnet-vision-d4.md) |
| sklearn features | [3 — Office-31 / D1](03-office31-sklearn-d1.md) |
| One-object API recap | [18 — PMHTrainer quickstart](18-pmh-trainer-quickstart.md) |
| Compare training arms | [17 — Compare arms](17-compare-arms-your-pipeline.md) |
