# Walkthrough 19: Office-31 real data (download → features → T1 table)

**Public dataset:** Office-31 (amazon, dslr, webcam). **Nothing is stored in git** — you download to a folder outside the repo.

| | |
|---|---|
| **Preset** | `t1_office31_sklearn` |
| **Recipe** | [T1 Office-31 D1](../recipes/t1-office31-d1.md) |
| **Script** | `scripts/download_office31.py` · `examples/21_benchmark_sklearn_table.py` |

---

## Prerequisites

```bash
pip install "matching-pmh[sklearn,vision]"
```

| Item | Notes |
|------|--------|
| Disk | ~1–2 GB for images + tar |
| Network | One-time download from Georgia Tech mirror |
| GPU | Optional (ResNet feature extract runs on CPU, slower) |

---

## Step 1 — Download (outside repo)

Pick a path **not** inside `matching-pmh/`:

```bash
# Windows example
python scripts/download_office31.py --root D:/datasets/office31

# Linux / macOS
python scripts/download_office31.py --root ~/datasets/office31
```

Verify layout only:

```bash
python scripts/download_office31.py --root D:/datasets/office31 --verify-only
```

Override URL if the default mirror is down:

```bash
python scripts/download_office31.py --root ~/datasets/office31 --url "https://YOUR_MIRROR/office31.tar"
```

Expected folders: `amazon/`, `dslr/`, `webcam/` (class subfolders per domain).

---

## Step 2 — Run T1 benchmark table

Amazon → DSLR (paper default):

```bash
python examples/21_benchmark_sklearn_table.py \
  --office31-root D:/datasets/office31 \
  --source amazon \
  --target dslr \
  --report results/office31_amazon_dslr
```

This extracts **ResNet-18** 512-d features at runtime (not saved to git) and runs `compare_arms_sklearn` with `preset=t1_office31_sklearn`.

---

## Step 3 — Read results honestly

| Metric | Where |
|--------|--------|
| Markdown table | `--report` folder or stdout |
| Reference (metrics only in git) | [office31_amazon_to_dslr.md](../benchmarks/office31_amazon_to_dslr.md) |
| Expectations | [When PMH helps](../WHEN_PMH_HELPS.md) |

**Do not** expect matched PMH to beat CORAL on this linear frozen-feature setup. Use the table for **protocol + falsification**, not marketing accuracy.

---

## Step 4 — Regenerate reference doc (maintainers)

Writes metrics-only markdown (no `.npy`):

```bash
python scripts/generate_reference_benchmark.py \
  --office31-root D:/datasets/office31 \
  --output docs/benchmarks/office31_amazon_to_dslr.md
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `domain folder not found` | Run download script; check `--root` contains `amazon/`, `dslr/`, `webcam/` |
| Download timeout | Retry or pass `--url` to another mirror |
| Slow feature extract | Add `--max-samples 500` on benchmark script |
| torchvision missing | `pip install "matching-pmh[vision]"` |

---

## Related

- [DATA_POLICY.md](../DATA_POLICY.md)
- [Walkthrough 3 — sklearn D1](03-office31-sklearn-d1.md)
- [examples/06_office31_sklearn.py](https://github.com/vishalstark512/matching-pmh/blob/main/examples/06_office31_sklearn.py)
