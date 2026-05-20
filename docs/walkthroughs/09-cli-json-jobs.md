# Walkthrough 9: CLI JSON jobs — full guide


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden paths](../GOLDEN_PATHS.md) · **Step 5:** compare_arms_sklearn on saved benchmark
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**At a glance**

| | |
|---|---|
| **Stack** | `pmh-train` CLI, HPC / reproducible configs |
| **Scripts** | `examples/05_yaml_config.py`, `examples/configs/*.json` |

[cli.md](../cli.md)

---

## Who this is for

You want **reproducible estimate/train jobs** without importing Python in every shell script — clusters, Makefiles, CI.

---

## Your deployment shift sentence

*Batch jobs estimate Sigma_task for site A vs B; same labels, reproducible JSON configs.* -> any Dk your config names.

---

## Step-by-step

1. List methods: `pmh-train list-methods` · paper presets: `pmh-train list-presets`
2. Copy `examples/configs/d4_estimate.json` → edit paths to **your** data (outside repo).
3. Run: `pmh-train estimate --config YOUR_JOB.json`
4. Benchmark: `pmh-train benchmark --config examples/configs/benchmark_sklearn.json`

```bash
python examples/05_yaml_config.py
```

---

## Adaptation worksheet

| Config field | Your value |
|--------------|------------|
| `artifact` output path | |
| Data paths | absolute, outside git |

---

## Verify

- [ ] JSON validates; artifact written
- [ ] Paths not committed ([DATA_POLICY.md](../DOCS_GUIDE.md))

---

## Next steps

- [1 — PyTorch D4](01-pytorch-domain-d4.md)
