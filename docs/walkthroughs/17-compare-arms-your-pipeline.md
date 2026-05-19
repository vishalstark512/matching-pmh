# Walkthrough 17: Compare training arms on **your** pipeline

**Goal:** Run **B0**, **matched**, **wrong-W**, and **isotropic** on the **same model and data** you already use—then read a comparison table. This is for **your** experiment, not a paper task ID.

**Template:** `examples/20_compare_training_arms.py`

---

## When to use this

- You integrated `PMHLoss` and want evidence the gain is from **matched** Σ, not generic regularization.
- You need a markdown/JSON table for a report or ablation section.
- You want the same weight initialization across arms (fair comparison).

---

## Steps

1. Implement `model_factory()` and `setup(model)` like in your real trainer.
2. Use **your** `train_loader` and a `val_loader` that reflects **deployment** (target domain, stressed sensors, held-out style, …).
3. Run:

```bash
python examples/20_compare_training_arms.py --out results/my_experiment
```

4. Open `results/my_experiment/benchmark.md`.

---

## Interpretation

| Pattern | Meaning |
|---------|---------|
| matched > b0, wrong_w ≈ isotropic | Strong support for matching principle |
| matched > b0, wrong_w also > b0 | May be generic regularization—tune down `weight` or check hook |
| matched ≈ b0 | Weak Σ ID (`preflight` marginal?) or wrong Dk |
| All arms similar | Val metric may not reflect nuisance; change eval |

---

## sklearn / feature-only pipelines

If you only have frozen features (no end-to-end training):

```bash
pmh-train benchmark --config examples/configs/benchmark_features.json
```

Edit the JSON to point at your `.npy` source/target features and labels.

---

## API reference

`pmh.benchmark.run_benchmark_protocol` · `write_benchmark_report` · see [ADAPT_YOUR_PIPELINE.md](../ADAPT_YOUR_PIPELINE.md).
