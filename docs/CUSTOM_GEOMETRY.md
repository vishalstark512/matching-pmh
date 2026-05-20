# Custom geometry (any subtype)

Use this when identification is **yours** (PGD deltas, gradient SVD, external \(W\), calibrator output) but the pipeline stays the same:

**artifact** → **PMHTrainer** / **PMHLoss** → **falsification controls**

See also [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) (pick Dk) · [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) (refinements).

---

## Three inputs (pick one)

| You have | Function | Typical subtype |
|----------|----------|-----------------|
| Stacked deltas `[N, d]` | `artifact_from_deltas` / `estimate_custom(deltas=…)` | D3, D6, D7 |
| Basis `W` `[d, r]` or `.npy` | `artifact_from_w` / `load_w_numpy` | Any |
| Source/target features | `estimate_custom(x_src=, x_tgt=)` | D4 unlabeled |
| Labeled both domains | `estimate_custom(..., y_src=, y_tgt=)` | D1 |

```python
from pmh import (
    artifact_from_deltas,
    artifact_from_w,
    estimate_custom,
    PMHTrainer,
    PMHConfig,
)

# 1) Your delta stack (T7B / T3A style)
art = artifact_from_deltas(my_deltas, method="D7", rank=16)

# 2) Precomputed W
art = artifact_from_w(W_numpy, method="D1")

# 3) Two domain matrices (D4 Gram)
art = estimate_custom(x_src=xs, x_tgt=xt, method="D4", rank=32)
```

Calibrators (paper refinements) return the same artifact type:

```python
from pmh.calibrate import gradient_subspace_numpy, content_residual_subspace

_, art = gradient_subspace_numpy(gradients, rank=16)
w, art = content_residual_subspace(sequences, rank=32, source="content")
```

---

## Train PyTorch (Phase B)

```python
art.save("artifacts/my_sigma.pt")

trainer = PMHTrainer.from_artifact(
    model,
    "artifacts/my_sigma.pt",
    hook=backbone,
    head=classifier,
    pmh_config=PMHConfig.balanced(),
)
trainer.fit(train_loader, epochs=20)  # artifact already loaded
```

Or estimate + train in one object when you have batches:

```python
trainer = PMHTrainer(model, hook=hook, nuisance="domain_shift")
trainer.fit(train_loader, source_batches=src, target_batches=tgt, epochs=20)
```

---

## Load two domains from disk

```python
from pmh.data_adapters import load_domain_arrays, batch_iterators

xs, ys, xt, yt = load_domain_arrays("data/source.npy", "data/target.npy")
src_it, tgt_it = batch_iterators(xs, xt, batch_size=32)
```

Labeled D1: `batch_iterators_labeled(xs, ys, xt, yt)`.

CLI estimate: `pmh-train estimate --config job.json` (see `examples/configs/`).

---

## Validate claims (CI)

```bash
pmh-train validate --config examples/configs/validate_sklearn_synthetic.json
```

Exits **0** only if **matched** beats **wrong-W** and **isotropic** on target accuracy.

Sklearn: `compare_arms_sklearn(..., report_dir="results/run1")` then `pmh-train validate --report results/run1/benchmark.json`.

Subtype tuning preset: `from pmh import get_subtype_preset` → `get_subtype_preset("D4")`.

---

## Example script

[`23_custom_geometry_train.py`](https://github.com/vishalstark512/matching-pmh/blob/main/examples/23_custom_geometry_train.py)

Per-subtype examples: [`examples/by_subtype/`](https://github.com/vishalstark512/matching-pmh/blob/main/examples/by_subtype/README.md)
