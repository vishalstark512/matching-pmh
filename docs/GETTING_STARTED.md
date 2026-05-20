# Integrate your project

**Prerequisites (required):** [START_HERE](START_HERE.md) → [Golden paths](GOLDEN_PATHS.md) (one section) → [First hour](FIRST_HOUR.md) demo.

**Goal:** wire PMH into **your** repo in one afternoon.  
**Not this page first:** [Research → Paper alignment](PAPER_ALIGNMENT.md) is for replication only.

---

<a id="step-0"></a>

## Checklist

### 0. One sentence (30 sec)

What changes at deployment **without changing the label**?

> *Example:* “Pose model trained in studio A; deploy on hospital camera B — same keypoint indices.”

If that fails → [When PMH helps](WHEN_PMH_HELPS.md).

```bash
pmh-train route --task pose_or_keypoints   # pick your task
```

### 1. Stack + path (done if you followed START_HERE)

| You chose | Doc |
|-----------|-----|
| PyTorch | [G1](GOLDEN_PATHS.md#g1) |
| Lightning | [G1b](GOLDEN_PATHS.md#g1b) |
| sklearn / `.npy` | [G2](GOLDEN_PATHS.md#g2) |
| HF corpora | [G3](GOLDEN_PATHS.md#g3) |
| HF `Trainer` | [G3b](GOLDEN_PATHS.md#g3b) |
| Custom Σ̂ | [G4](GOLDEN_PATHS.md#g4) |

Subtype reference (only if wizard asked): [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md).

### 2. Install

```bash
pip install matching-pmh torch
# pip install "matching-pmh[sklearn]"   # G2
# pip install "matching-pmh[hf]"        # G3 / G3b
# pip install "matching-pmh[lightning]" # G1b
pmh-train doctor
```

Folder estimate: [DATA_LAYOUT.md](DATA_LAYOUT.md) · `pmh-train estimate --source-dir A/ --target-dir B/`

### 3. Wire into your training loop

Use the snippet from your golden path section. Rules:

- Keep your **task loss** (pose L2, CE, etc.).
- `source_batches` = train/deploy site A features; `target_batches` = site B (labels optional for D4).
- `hook` = representation **before** task head.

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    your_model,
    hook=your_backbone,
    head=your_head,  # optional
    nuisance="domain_shift",
    pmh_config=PMHConfig.balanced(),
    artifact_path="artifacts/my_sigma.pt",
)
trainer.fit(
    your_train_loader,
    source_batches=loader_site_a,
    target_batches=loader_site_b,
    epochs=20,
)
print("preflight:", trainer.artifact_.preflight)
```

Or one call: `robust_fit(...)` from [G1](GOLDEN_PATHS.md#g1).

### 4. Validate before you claim success

| Check | How |
|-------|-----|
| Applicability | `check_applicability(...)` |
| Target metric | `evaluate_robust_fit` or your val loop |
| Not generic reg | [Falsification controls](walkthroughs/08-falsification-controls.md) |
| CI gate | `pmh-train validate -c examples/configs/validate_sklearn_synthetic.json` |

### 5. Ship (optional)

[DEPLOYMENT.md](DEPLOYMENT.md) — `export_deployment(artifact, "bundle/")`

---

## Domain-specific templates

| Domain | Gallery |
|--------|---------|
| Vision / pose | [gallery/vision.md](gallery/vision.md) |
| Tabular | [gallery/tabular.md](gallery/tabular.md) |
| NLP | [gallery/nlp.md](gallery/nlp.md) |

---

## Stuck?

[TROUBLESHOOTING.md](TROUBLESHOOTING.md) · [hooks.md](hooks.md) · [Doc map](MAP.md)
