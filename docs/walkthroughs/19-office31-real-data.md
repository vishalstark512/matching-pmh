# Walkthrough 19: Office-31 real data (download → features → T1 table)


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden path G1–G4](../GOLDEN_PATHS.md#g2) · **Route:** `pmh-train route --task frozen_embeddings_sklearn` · **Step 5:** compare_arms_sklearn preset t1
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**Public dataset:** Office-31 (amazon, dslr, webcam). **Nothing is stored in git** — you download to a folder outside the repo.

| | |
|---|---|
| **Preset** | `t1_office31_sklearn` |
| **Recipe** | [T1 Office-31 D1](../PAPER_ALIGNMENT.md) |
| **Script** | `scripts/download_office31.py` · `examples/21_benchmark_sklearn_table.py` |

---

## Your deployment shift sentence

*"Amazon vs DSLR vs webcam - same 31 classes, different imaging domain."* -> **D1** subspace on frozen features.

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
| Reference (metrics only in git) | [office31_amazon_to_dslr.md](../walkthroughs/19-office31-real-data.md) |
| Expectations | [When PMH helps](../WHEN_PMH_HELPS.md) |

**Do not** expect matched PMH to beat CORAL on this linear frozen-feature setup. Use the table for **protocol + falsification**, not marketing accuracy.

---

## Step 4 — Regenerate reference doc (maintainers)

Writes metrics-only markdown (no `.npy`):

```bash
python scripts/generate_reference_benchmark.py \
  --office31-root D:/datasets/office31 \
  --output docs/walkthroughs/19-office31-real-data.md
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

- [DATA_POLICY.md](../DOCS_GUIDE.md)
- [Walkthrough 3 — sklearn D1](03-office31-sklearn-d1.md)
- [examples/06_office31_sklearn.py](https://github.com/vishalstark512/matching-pmh/blob/main/examples/06_office31_sklearn.py)
