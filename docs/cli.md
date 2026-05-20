# CLI: `pmh-train`

Installed with the package:

```bash
pip install matching-pmh
pmh-train wizard          # start here (interactive)
pmh-train list-methods    # advanced: D1–D7 estimators
pmh-train list-presets    # paper replication presets
```

## Commands

| Command | Purpose |
|---------|---------|
| **`wizard`** | **Interactive setup** — stack, data flags, install line, copy-paste snippet |
| `list-methods` | Table of D1–D7 inputs (research / advanced) |
| `list-presets` | Paper block presets (`t1_office31_sklearn`, …) |
| `estimate --config job.json` | Run estimator, write `output.pt` + `.json` |
| `preflight ARTIFACT.pt` | Geometry diagnostics on saved artifact |
| `benchmark --config …` | Sklearn falsification table from JSON job |
| `run --config job.json` | Validate training job (artifact + PMH weights) |

### Wizard

```bash
# Interactive (recommended for new users)
pmh-train wizard

# Non-interactive (CI / scripts)
pmh-train wizard --non-interactive --stack pytorch
pmh-train wizard --non-interactive --stack sklearn
pmh-train wizard --non-interactive --stack hf --style-pairs
```

Same logic as `python -m pmh.onboarding --wizard`.

---

## Example jobs (advanced)

- `examples/configs/d4_estimate.json` — domain Gram (numpy paths)
- `examples/configs/d7_style_estimate.json` — HF style JSONL
- `examples/configs/dpo_train_job.json` — training recipe after D7 estimate

```bash
pmh-train estimate --config examples/configs/d7_style_estimate.json
pmh-train preflight artifacts/d7_style.pt
pmh-train run --config examples/configs/dpo_train_job.json
```

Python equivalent: `python -m pmh.cli.main estimate --config ...`
