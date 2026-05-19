# Benchmarks and TDI metrics

## Where is TDI?

| Location | What |
|----------|------|
| **This package** | `pmh.tdi` — `tdi_cls`, `tdi_feature_isotropic`, `directional_drift_numpy`, `geometry_report` |
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

**Trajectory TDI** on deep nets (layer-averaged input Gaussian noise) is defined in the paper; implement with your encoder + `sigma=0.01` probes, or use `tdi_feature_isotropic` on frozen features for a quick sklearn comparison.

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
