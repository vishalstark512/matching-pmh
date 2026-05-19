# Walkthrough 9: CLI and JSON jobs

**Goal:** Run estimation (and some train flows) **without** writing a custom estimate script—useful for HPC / reproducible logs.

**Configs:** `examples/configs/`  
**CLI:** `pmh-train`

---

## Prerequisites

```bash
pip install matching-pmh
pmh-train list-methods
```

---

## Step 1 — D4 from saved feature matrices

1. Export features once from your encoder:

```python
import numpy as np
np.save("features/source.npy", h_src.numpy())
np.save("features/target.npy", h_tgt.numpy())
```

2. Edit `examples/configs/d4_estimate.json`:

```json
{
  "estimator": { "method": "D4", "rank": 32, "shrinkage": 1e-6 },
  "data": {
    "source_npy": "features/source.npy",
    "target_npy": "features/target.npy"
  },
  "output": "artifacts/d4_domain"
}
```

3. Run:

```bash
pmh-train estimate --config examples/configs/d4_estimate.json
pmh-train preflight artifacts/d4_domain.pt
```

---

## Step 2 — D7 style estimate job

```bash
pmh-train estimate --config examples/configs/d7_style_estimate.json
```

Point `jsonl`, `model_id`, and `output` in that file to your paths.

---

## Step 3 — DPO + PMH run job

```bash
pmh-train run --config examples/configs/dpo_train_job.json
```

Mirrors `examples/11_dpo_lora_style_pmh.py` flags in JSON (model, LoRA, artifact path).

---

## Step 4 — Programmatic JSON (no CLI)

`examples/05_yaml_config.py` shows loading estimator + training hyperparameters:

```python
job = {
    "estimator": {"method": "D4", "rank": 32, "shrinkage": 1e-5},
    "training": {"weight": 0.25, "cap_ratio": 0.3, "warmup_epochs": 2},
}
est_cfg = SigmaTaskConfig.from_dict(job["estimator"])
pmh_cfg = PMHConfig.from_dict(job["training"])
```

---

## When CLI vs Python

| CLI | Python loop |
|-----|-------------|
| Batch HPC, fixed configs | Research iteration |
| Team reproducibility | Custom `representation_fn` |
| Precompute `.npy` features | End-to-end fine-tuning |

Full flag reference: [cli.md](../cli.md).
