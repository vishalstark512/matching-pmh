# Benchmarks and TDI metrics

## Where is TDI?

| Location | What |
|----------|------|
| **This package** | `pmh.tdi` — layout + trajectory TDI, `directional_drift_numpy`, `geometry_report`; `PMHTrainer.measure_trajectory_tdi` |
| **Paper replication** | `Paper2/T3/Task3A/tdi_utils.py`, `T4/*/tdi.py`, `T6/*/eval_paper_metrics.py` (not imported by `pip install matching-pmh`) |
| **Manuscript** | [Theory / diagnostics](THEORY.md) · Grand Unification §6 (trajectory TDI, layout TDI, \(D_N/D_S\)) |

Before v1.3, the library only exposed **eigengap** preflight (`pmh.diagnostics`, `artifact.preflight`). **TDI and directional drift are now first-class** in benchmarks and `pmh.tdi`.

## Metrics (short)

| Metric | Needs labels? | Lower is better? | Use when |
|--------|---------------|------------------|----------|
| `tdi_cls` | Yes (probe set) | Yes | Frozen embeddings / sklearn tables |
| `tdi_feature_isotropic` | No | Yes | Feature-space sensitivity proxy |
| `D_N`, `D_S`, `D_N_over_D_S` | No (needs \(\hat W\)) | \(D_N/D_S\) context-dependent | Nuisance subspace available |
| `target_accuracy` | Yes | No (higher better) | Standard DA benchmark |
| `trajectory_tdi` | No | Yes | Input noise @ `sigma=0.01` on encoder hooks (paper T2A) |

**Trajectory TDI** (layer-averaged ratio of perturbed vs clean representation drift):

```python
from pmh import PMHTrainer, trajectory_tdi_layerwise

# PyTorch encoder (single layer or CLS)
metrics = trainer.measure_trajectory_tdi(val_loader, sigma=0.01)
print(metrics["trajectory_tdi"], metrics["tdi_per_layer"])

# NumPy per-layer stacks (ViT hooks)
tdi, per_layer = trajectory_tdi_layerwise(clean_layers, pert_layers)
```

For frozen sklearn tables use `tdi_feature_isotropic` as a fast proxy.

## Reference tables (metrics only, in git)

| File | Protocol |
|------|----------|
| [office31_synthetic_reference.md](benchmarks/office31_synthetic_reference.md) | Synthetic Office-31-style shift, rank=16 |
| [office31_amazon_to_dslr.md](benchmarks/office31_amazon_to_dslr.md) | **Real** Office-31: ResNet-18 penultimate features, Amazon → DSLR, rank=16, seed=0, ≤2000 samples/domain |

**Real Office-31 snapshot (2026-05-19):** target accuracy — B0 **0.71**, matched **0.63**, wrong-W **0.71**, CORAL **0.68**. Matched does **not** beat B0 on accuracy here; use this table as a **protocol reference**, not a headline win. Re-run with your encoder, rank, and full data before claiming gains. Geometry (`TDI_cls`, `D_N/D_S`) is reported alongside accuracy for falsification (see table notes).

Regenerate locally (no datasets committed):

```bash
pip install -e ".[sklearn,vision]"
# Download Office-31 once (outside repo), e.g. Paper2 script or Tsinghua mirrors:
#   python path/to/download_office31.py --root D:/data/office31
python scripts/generate_reference_benchmark.py --office31-root D:/data/office31 --output docs/benchmarks/office31_amazon_to_dslr.md
```

Synthetic only (no download):

```bash
python scripts/generate_reference_benchmark.py
```

## Standard sklearn benchmark (like scikit-learn example galleries)

One command — synthetic Office-31-style shift or real Office-31 features:

```bash
pip install "matching-pmh[sklearn]"
python examples/21_benchmark_sklearn_table.py
# Real Office-31: path outside the repo (never commit the dataset)
python examples/21_benchmark_sklearn_table.py --office31-root /data/office31 --report results/run1
```

See [DATA_POLICY.md](DATA_POLICY.md) — datasets and `.npy` features are **not** stored in git or PyPI.

Programmatic:

```python
from pmh import compare_arms_sklearn
from pmh.benchmark.report import benchmark_to_markdown

result = compare_arms_sklearn(x_src, y_src, x_tgt, y_tgt, rank=16, report_dir="results/run1")
print(benchmark_to_markdown(result.to_dict()))
```

### Arms (falsification protocol)

| Arm | Meaning |
|-----|---------|
| `b0` | Baseline classifier on source, test on target |
| `matched` | Matched subspace projection (D1) then classify |
| `wrong_w` | Random \(\hat W\) control (Lemma C) |
| `isotropic` | Top-\(r\) SVD of target (non-matched) |
| `coral` | CORAL alignment baseline (optional) |

**Read the table:** matched should improve **target accuracy** vs B0 when the nuisance story holds; **TDI_cls** and **D_N/D_S** should improve with matched vs wrong-W / isotropic (geometry can move without accuracy — see paper T6A).

## PyTorch training benchmark

For end-to-end training (not frozen features):

```python
from pmh import compare_arms

compare_arms(
    trainer.artifact_,
    model_factory,
    setup_model,
    train_loader,
    val_loader,
    epochs=10,
    report_dir="results/pytorch_arms",
)
```

See [walkthrough 17](walkthroughs/17-compare-arms-your-pipeline.md).

## Compare to CORAL / other libraries

| Library | Typical benchmark |
|---------|-------------------|
| sklearn | Iris, digits — **task accuracy only** |
| CORAL / DANN repos | Office-31, VisDA — **accuracy** |
| **matching-pmh** | Same datasets + **accuracy + TDI + D_N/D_S** in one report |

[CORAL migration guide](COMPARE_TO_CORAL.md) · [sklearn path](sklearn.md)
