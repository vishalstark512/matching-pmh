# matching-pmh

**Installable reference library for the [matching principle](https://pypi.org/project/matching-pmh/): estimate deployment nuisance geometry, then train with a matched PMH penalty.**

| | |
|---|---|
| **PyPI** | https://pypi.org/project/matching-pmh/ |
| **GitHub** | https://github.com/vishalstark512/matching-pmh |
| **Import** | `import pmh` |
| **CLI** | `pmh-train` |

Companion to the research line on *The Matching Principle* (grand-unification manuscript and thirteen empirical blocks). This repository is **code only** — theory, proofs, and task numbers live in the paper repo.

---

## What problem does this solve?

Standard training (ERM) encourages the model to use **every input direction that helps on the training set**, including nuisances that are **useless or harmful at deployment** (lighting, domain style, sensor noise, answer formatting, identifier names in code, etc.).

The matching principle separates two steps:

1. **Estimate** \(\Sigma_{\mathrm{task}}\) — the covariance of *label-preserving* nuisance variation you expect at deployment.
2. **Regularize** the encoder Jacobian along that covariance (matched PMH), so the representation does not entangle task signal with nuisance.

`matching-pmh` implements **seven estimators (D1–D7)** for step 1 and a **training loss (`PMHLoss`)** for step 2, with pre-flight checks and falsification controls (wrong-\(W\), isotropic).

---

## The workflow (same for every application)

```
  Your task + deployment story
           │
           ▼
  Pick estimator D1–D7  ◄───  "What kind of nuisance moves labels the same way?"
           │
           ▼
  estimate_from_config / pmh-train estimate  →  artifact (.pt)
           │
           ▼
  preflight (eigengap)  →  pass | marginal | fail
           │
           ▼
  Train:  task_loss + PMHLoss(h)   (PyTorch, HF Trainer, or Lightning)
           │
           ▼
  Report matched vs wrong-W vs isotropic (and signal-W when applicable)
```

You do **not** need the paper codebase to use this library. You need a clear story: *what changes at deployment without changing the label?*

---

## Which estimator (D1–D7) for my case?

| Your situation | Method | What you estimate from | Typical domains |
|----------------|--------|------------------------|-----------------|
| **Domain / dataset shift** (different camera, site, corpus) | **D4** domain Gram | Unlabeled features from source vs target | Vision DA, Office-31-style |
| **Low-rank subspace shift** (with labels on both sides) | **D1** cross-domain SVD | Paired source/target features + labels | Digits, re-ID-style |
| **Unstructured input noise** (sensor, quantization) | **D2** isotropic | Noise level + representation dim | IMU, low-level sensors |
| **Known augmentation modes** (jitter, blur, color) | **D3** augmentation | Stack of aug-induced feature deltas | Photometric robustness |
| **Compositional nuisance** (per-atom, per-token, per-coordinate) | **D5** compositional | Feature matrix + nuisance index list | QM9 atoms, code tokens |
| **Temporal nuisance** (drift along time) | **D6** temporal | Sequences of representations | HAR, finance windows |
| **Style / format vs content** (LLM answers) | **D7** alignment Gram | Style-variant JSONL on fixed content | RLHF / DPO alignment |

**Rule of thumb:** if you can name the nuisance and generate or observe it **without changing the label**, there is a row above. If preflight fails (small eigengap), the estimator is ill-conditioned for your data — try more calibration data or a different \(A_k\) (see paper §5.7).

List methods in the terminal:

```bash
pmh-train list-methods
```

---

## Use cases (how people actually wire this)

### A — Vision / domain adaptation (D4 or D1)

Train a backbone on source data; estimate \(\Sigma_{\mathrm{task}}\) from **unlabeled** target (or paired domains for D1). Add `PMHLoss` on penultimate features during fine-tuning.

```python
from pmh import SigmaTaskConfig, estimate_from_config, PMHLoss, PMHConfig

artifact = estimate_from_config(
    SigmaTaskConfig.for_domain(rank=64),
    h_source,   # [N, d] from frozen or warm encoder
    h_target,
)
artifact.save("artifacts/domain_sigma")
pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2))
# total_loss = task_loss + pmh.capped_total(task_loss, h)[1]
```

Example: `examples/01_domain_shift_d4.py`, `examples/06_office31_sklearn.py`.

### B — Photometric / augmentation nuisance (D3)

Collect feature deltas under each augmentation mode; D3 builds \(\Sigma_{\mathrm{task}}\) from those directions.

Example: `examples/07_vision_multilayer.py` (multi-layer Gram).

### C — Compositional coordinates (D5)

QM9-style: nuisance = atom identity; signal = chemistry. Provide `nuisance_indices` into the representation.

Example: `examples/03_compositional_d5.py`.

### D — Time series (D6)

Windows of sequence features; temporal residual covariance.

Example: estimators in `src/pmh/estimators/d6_temporal.py` + task-specific scripts in the paper repo.

### E — LLM alignment / style (D7)

**Inputs:** JSONL with fixed semantic content and multiple surface forms (T7A schema):

- `style_pairs.jsonl`: `prompt`, `content_fixed`, `style_variants` (dict of paraphrases)
- `preference_pairs.jsonl`: for DPO-style training with `chosen` / `rejected`

```bash
pip install "matching-pmh[hf-lora]"
pmh-train estimate --config examples/configs/d7_style_estimate.json
python examples/11_dpo_lora_style_pmh.py --model-id Qwen/Qwen2.5-0.5B-Instruct --train --lora
```

Example: `examples/08_hf_style_d7.py`, `examples/10_hf_trainer.py`, `examples/11_dpo_lora_style_pmh.py`.

### F — CLI-only / reproducible jobs

```bash
pmh-train estimate --config examples/configs/d4_estimate.json
pmh-train preflight artifacts/d4.pt
pmh-train run --config examples/configs/dpo_train_job.json
```

Configs live under `examples/configs/`. See [docs/nuisance_types.md](docs/nuisance_types.md) and [docs/cli.md](docs/cli.md).

### G — Falsification (always report these)

| Arm | Meaning |
|-----|---------|
| **matched** | \(\Sigma_{\mathrm{task}}\) from the correct nuisance |
| **wrong-W** | Random or misaligned subspace — should behave like weaker or wrong geometry |
| **isotropic** | Uninformative directions — classic VAT-like baseline |
| **signal-W** (when applicable) | Penalize *task* directions — should **hurt** |

```python
PMHLoss(artifact, config, mode="wrong_w")   # or "isotropic"
```

Example: `examples/04_falsification_controls.py`.

---

## Install

```bash
pip install matching-pmh
```

| Extra | Command | When |
|-------|---------|------|
| Hugging Face (D7) | `pip install "matching-pmh[hf]"` | Style Gram from an LM |
| LoRA + DPO demo | `pip install "matching-pmh[hf-lora]"` | Example 11 |
| sklearn / Office-31 | `pip install "matching-pmh[sklearn,vision]"` | Example 06 |
| Lightning | `pip install "matching-pmh[lightning]"` | Example 09 |
| Dev | `pip install "matching-pmh[dev]"` | Tests, ruff |

From source:

```bash
git clone https://github.com/vishalstark512/matching-pmh.git
cd matching-pmh
pip install -e ".[dev]"
pytest
```

---

## Minimal Python example

```python
import torch
from pmh import SigmaTaskConfig, PMHConfig, PMHLoss, estimate_from_config

# h_source, h_target: [N, d] representations (e.g. from a frozen encoder)
artifact = estimate_from_config(
    SigmaTaskConfig.for_domain(rank=32),
    h_source,
    h_target,
)
print("preflight:", artifact.preflight, "eigengap:", artifact.eigengap)

pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2))
h = model(x)                    # [B, d]
task_loss = criterion(h, y)
total, pmh_term = pmh.capped_total(task_loss, h)
total.backward()
```

Load later:

```python
from pmh import SigmaTaskEstimate, PMHLoss
artifact = SigmaTaskEstimate.load("artifacts/domain_sigma.pt")
```

---

## Example scripts

| Script | Nuisance | Shows |
|--------|----------|--------|
| `01_domain_shift_d4.py` | D4 | Features + `PMHLoss` training loop |
| `02_save_load_artifact.py` | any | `.pt` / `.json` artifact I/O |
| `03_compositional_d5.py` | D5 | Coordinate-block \(\Sigma\) |
| `04_falsification_controls.py` | any | matched / wrong-W / isotropic |
| `05_yaml_config.py` | any | JSON job → configs |
| `06_office31_sklearn.py` | D1 | Office-31 + sklearn classifier |
| `07_vision_multilayer.py` | D3/D4 | Per-layer penalties |
| `08_hf_style_d7.py` | D7 | Style JSONL → artifact |
| `10_hf_trainer.py` | D7 | `PMHTrainer` toy loop |
| `11_dpo_lora_style_pmh.py` | D7 | Qwen JSONL + optional LoRA DPO |

Sample data: `examples/data/style_pairs_sample.jsonl`, `preference_pairs_sample.jsonl`.

---

## API map

| Step | API |
|------|-----|
| Choose estimator | `SigmaTaskConfig.for_domain()`, `.for_subspace()`, `.for_alignment()`, … |
| Estimate | `estimate_from_config(cfg, ...)` → `SigmaTaskEstimate` |
| Legacy shorthand | `estimate_sigma_task(..., method="D4")` |
| Collect features | `collect_features(encoder, loader)` |
| Preflight | `artifact.preflight`, `preflight_eigengap(cov, rank)` |
| Train | `PMHLoss(artifact, PMHConfig(...))` |
| Cap PMH scale | `cap_ratio` (~30% of task loss is a common target) |

---

## Documentation

- Local docs: `mkdocs serve` (see `docs/`)
- [Getting started](docs/getting-started.md)
- [Nuisance cookbook D1–D7](docs/nuisance_types.md)
- [CLI reference](docs/cli.md)

---

## Citation

If you use this package, cite the matching-principle manuscript (Grand Unification paper). BibTeX: see `CITATION.cff` in this repo.

---

## Status

**0.6.0** on PyPI — D1–D7 estimators, `PMHLoss`, `pmh-train` CLI, HF/Lightning integrations, Qwen/T7A JSONL example.

MIT License. Issues: https://github.com/vishalstark512/matching-pmh/issues
