# Walkthrough 9: CLI JSON jobs — full guide

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
- [ ] Paths not committed ([DATA_POLICY.md](../DATA_POLICY.md))

---

## Next steps

- [1 — PyTorch D4](01-pytorch-domain-d4.md)
