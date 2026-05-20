# CLI: `pmh-train`

Installed with the package:

```bash
pip install matching-pmh
pmh-train list-methods
pmh-train list-presets
```

## Commands

| Command | Purpose |
|---------|---------|
| `list-methods` | Table of D1–D7 inputs |
| `list-presets` | Paper block presets (`t1_office31_sklearn`, `t4_domain_d4`, …) |
| `estimate --config job.json` | Run estimator, write `output.pt` + `.json` |
| `preflight ARTIFACT.pt` | Eigengap $\gamma_r$ diagnostics |
| `run --config job.json` | Validate training job (artifact + PMH weights) |

## Example jobs

- `examples/configs/d4_estimate.json` — domain Gram (numpy paths)
- `examples/configs/d7_style_estimate.json` — HF style JSONL
- `examples/configs/dpo_train_job.json` — training recipe after D7 estimate

```bash
pmh-train estimate --config examples/configs/d7_style_estimate.json
pmh-train preflight artifacts/d7_style.pt
pmh-train run --config examples/configs/dpo_train_job.json
```

Python equivalent: `python -m pmh.cli.main estimate --config ...`
