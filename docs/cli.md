# CLI: `pmh-train`

```bash
pip install matching-pmh
pmh-train wizard          # start here
pmh-train doctor          # check install / extras
pmh-train list-methods    # D1–D7 + subtype lines
pmh-train list-presets    # paper + subtype presets
```

## Commands

| Command | Purpose |
|---------|---------|
| **`wizard`** | Interactive stack + subtype + snippet |
| **`doctor`** | Environment check (`--stack pytorch\|sklearn\|hf`) |
| **`estimate`** | Phase A — JSON, `.npy` paths, or **folders** |
| **`validate`** | Falsification pass/fail (exit 1 if controls fail) |
| `preflight` | Eigengap on saved artifact |
| `benchmark` | Sklearn arm table → JSON + markdown |
| `list-methods` / `list-presets` | Reference tables |

### Estimate (data)

```bash
# Folders: features.npy (+ optional labels.npy) per site
pmh-train estimate --source-dir data/site_a --target-dir data/site_b \
  --method D4 --rank 32 -o artifacts/my_run

# Explicit matrices
pmh-train estimate --source-npy a.npy --target-npy b.npy -o artifacts/my_run

# JSON job (HPC)
pmh-train estimate --config examples/configs/d4_estimate.json
```

See [DATA_LAYOUT.md](DATA_LAYOUT.md).

### Validate (CI)

```bash
pmh-train validate --config examples/configs/validate_sklearn_synthetic.json
pmh-train validate --config examples/configs/validate_pytorch_smoke.json
pmh-train validate --report results/validate/validate.json
```

### Wizard

```bash
pmh-train wizard
pmh-train wizard --non-interactive --stack pytorch
```

---

## Example configs

| File | Use |
|------|-----|
| `d4_estimate.json` | D4 from `.npy` paths |
| `validate_sklearn_synthetic.json` | Sklearn falsification gate |
| `validate_pytorch_smoke.json` | PyTorch toy arms gate |
| `d7_style_estimate.json` | D7 style JSONL |

---

Python equivalents: `python -m pmh.onboarding --wizard` · [Golden paths](GOLDEN_PATHS.md)
