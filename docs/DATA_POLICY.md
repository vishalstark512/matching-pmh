# Data policy (no datasets in git or PyPI)

**matching-pmh** is a library only. We do **not** commit datasets, checkpoints, extracted features, or download caches to GitHub, and we do **not** bundle them in PyPI wheels/sdists.

## What stays out of the repo

| Do not commit | Examples |
|---------------|----------|
| Raw datasets | Office-31 images, ImageNet folders |
| Download caches | `tmp.tar.gz`, `data/office31/`, `*_downloads/` |
| Extracted features | `*.npy`, `*.npz`, ResNet embedding dumps |
| Training artifacts | `*.pt`, `artifacts/` (already gitignored) |
| Large reports with tensors | `docs/benchmarks/raw/` |

`.gitignore` enforces this locally; review `git status` before pushing.

## What *is* in the repo (small demos only)

| Allowed | Purpose |
|---------|---------|
| `examples/data/*.jsonl` | Tiny D7 style / preference **samples** (few KB) for smoke tests |
| `docs/benchmarks/*.md` | **Metric tables only** (accuracy, TDI) — no feature matrices |

## Running benchmarks on real data

Use a path **outside** the repo (or gitignored `data/`):

```bash
python examples/21_benchmark_sklearn_table.py \
  --office31-root /path/outside/repo/office31 \
  --report /tmp/pmh_benchmark   # or ./results/ (gitignored)
```

Office-31 is downloaded or placed by **you**; the library only reads it at runtime.

## PyPI package contents

Wheels contain **Python source** under `pmh/` only (`pyproject.toml` → `[tool.hatch.build.targets.wheel]`). No `data/` tree is shipped.

## Reference benchmark results

Published reference numbers live as **Markdown/JSON summaries** under `docs/benchmarks/` (when present). Regenerate locally; do not commit raw inputs or embeddings.
