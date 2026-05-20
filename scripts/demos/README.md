# Runnable demos (optional)

Adoption path is **notebooks** under `notebooks/tasks/` and **`pmh-train`**. These three scripts are for CI smoke and batch jobs only.

| Script | When to use |
|--------|-------------|
| [first_run_domain_shift.py](first_run_domain_shift.py) | Fast PyTorch smoke (`PMH_QUICK=1`) |
| [office31_sklearn.py](office31_sklearn.py) | T1 Office-31 or synthetic sklearn arms |
| [benchmark_sklearn_table.py](benchmark_sklearn_table.py) | Multi-arm benchmark table + `--report` |

```bash
PMH_QUICK=1 python scripts/demos/first_run_domain_shift.py
python scripts/demos/office31_sklearn.py
python scripts/demos/benchmark_sklearn_table.py --report results/sklearn_benchmark
```

CLI sample configs: `scripts/configs/`
