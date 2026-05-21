# T1 — Classical ML + matched projection (Type 1)

**Paper evidence:** [main.pdf](../../main.pdf) · [Block findings](../findings.html)

**Nuisance:** D1 subspace · **Mode B:** estimate $\hat{W}$, project features (or matched ridge penalty), then **classical** models — ridge, SVM, k-NN, logistic.

**Notebook:** [t01-classical.ipynb](../../notebooks/tasks/t01-classical.ipynb) — one Run All path:

| § | Content |
|---|---------|
| 1–4 | Core loop + falsification (logistic, SVM, k-NN) |
| 5 | Office-31 ResNet extract + `t1_office31_sklearn` (toggle `RUN_OFFICE31`) |
| 6 | Soft k-NN (`pmh.classical.compare_knn_hard_vs_soft`) |
| 7 | Ridge + California housing tabular |
| 8 | MNIST pixels + data-driven $W$ |

Quick smoke: `python scripts/demos/office31_sklearn.py` or Run All on [t01-classical.ipynb](../../notebooks/tasks/t01-classical.ipynb).

---

## What T1 achieved (headline)

> **Matched anisotropic regularization along a nuisance subspace $W$** beats isotropic and wrong-$W$ controls — across **ridge (theorem)**, **SVM / k-NN / logistic**, **synthetic and real** features.

| Evidence | Result |
|----------|--------|
| **Theorem 1** (ridge, closed form) | Matched-ridge test MSE **flat** in OOD nuisance scale; B0 / iso / wrong grow with $\sigma_{test}$ |
| **Oracle $W$** (MNIST, Fashion-MNIST) | **Strict four-arm ordering** for SVM, k-NN, logistic — matched wins |
| **Data-driven $W$** (DCT drift) | Strong on Fashion-MNIST SVM/logistic; k-NN needs **soft** Mahalanobis metric |
| **SVHN** (real street digits) | SVM matched **+20 pp** over B0 |
| **California Housing** (tabular ridge) | Same flat-MSE behaviour as theorem on real UCI features |
| **EB-GP** | Marginal likelihood picks matched limit when $W$ is right, isotropic when wrong |
| **Office-31** (Amazon → DSLR, ResNet-18 512-d) | **Honest mixed result:** CORAL beats PMH on all three classifiers; PMH still **> B0** on SVM |

Paper arms: **`B0`** · **`E1_iso`** · **`E1_matched`** · **`E1_wrong`**

Library keys: `b0` · `isotropic` · `matched` · `wrong_w` (+ optional `coral`)

---

## How data is plugged in (paper → you)

Every T1 script follows the same pattern:

1. **Fixed features** $x = \phi(\text{input})$ — pixels, ResNet embeddings, or tabular rows.
2. **Site A / site B** — train vs deploy (or oracle-injected nuisance along known $W$).
3. **Estimate $\hat W$** — cross-domain SVD (DCT drift, Office-31) or oracle orthogonal subspace.
4. **Apply PMH** — hard projection $P_{W^\perp}$ for SVM/logistic; soft metric for k-NN under estimated $W$.
5. **Report** on OOD / target test only (pool vs test split on Office-31).

| T1 experiment | Paper script | Data plug-in | Classifiers |
|---------------|--------------|--------------|-------------|
| Ridge theorem | `ridge_theory.py`, `ridge_theorem_check.py` | Synthetic Gaussian, $d{=}50$, $r{=}5$ | Ridge only |
| Oracle-W images | `run_benchmarks.py` | MNIST / Fashion-MNIST subsets | SVM, k-NN, logistic |
| DCT drift | `mnist_drift.py` | Pixels + **estimated** $W$ from cross-domain SVD | SVM, k-NN (hard/soft), logistic |
| Soft k-NN fix | `soft_knn.py` | Same drift features | k-NN with CV on $\alpha,\beta$ |
| Baselines | `baselines.py` | Oracle + drift | PMH vs **CORAL**, LMNN, IRM |
| SVHN | `svhn_subspace.py` | Real SVHN digits | SVM |
| Tabular ridge | `ridge_tabular.py` | `fetch_california_housing` + injected $W$ | Ridge |
| EB-GP | `eb_gp.py` | Synthetic GP inputs | GP regression |
| **Office-31** | `office31_pmh.py` | `download_office31.py` → ResNet-18 per domain | SVM, k-NN, logistic |

Full T1 battery: see [main.pdf](../../main.pdf) §T1 (seven sub-experiments).

---

## Run with matching-pmh (your pipeline)

You only need **frozen** `[N, d]` features + labels on both sites (same meaning).

```bash
pip install "matching-pmh[sklearn,vision]"
```

### Office-31 (paper §9 — main real DA benchmark)

```bash
python scripts/download_office31.py --root YOUR_OFFICE31_ROOT
python scripts/demos/office31_sklearn.py --office31-root YOUR_OFFICE31_ROOT --source amazon --target dslr
```

Preset `t1_office31_sklearn`: rank 32, target pool 200, test 250 (matches paper protocol).

Batch table (accuracy + TDI):

```bash
python scripts/demos/benchmark_sklearn_table.py --office31-root YOUR_OFFICE31_ROOT --report results/sklearn_benchmark
```

### Any classical head on your features

```python
from pmh import compare_arms_sklearn

result = compare_arms_sklearn(
    x_source, y_source, x_target, y_target,
    preset="t1_office31_sklearn",
    include_coral=True,
)
```

Default classifier is **logistic** (one call). Paper Table uses **SVM, k-NN, logistic** separately — pass a custom `classifier_factory` to `run_sklearn_benchmark` for parity.

### Protocol check (not a headline result)

Built-in arrays (`load_g2_demo_arrays`) check D1 + falsification arms without downloading data — same **matched / wrong_w / isotropic** logic, not a substitute for MNIST drift or Office-31.

---

## Office-31 numbers (paper, 3 seeds)

| Method | B0 | PMH | CORAL | LMNN |
|--------|-----|-----|-------|------|
| SVM | 21.6% | 23.3% | **25.2%** | 20.0% |
| k-NN | 20.4% | 15.5% | **24.4%** | 18.0% |
| Logistic | 22.4% | 22.4% | **26.8%** | 18.8% |

**Read:** CORAL wins on ResNet embeddings; PMH is still part of the evidence map (toy → drift → real DA). Deep PMH (Types 3–7) is where feature-level training continues.

---

## Limitations (from FINAL.md)

1. Linear projection fails on severe non-linear shift (e.g. raw USPS↔MNIST at chance).
2. **Hard** projection hurts k-NN when $W$ is noisy — use soft Mahalanobis (paper `soft_knn.py`).
3. Office-31: CORAL > PMH on 512-d ResNet features.
4. IRM can win when **two training environments** are available.
5. Rotation nuisance in pixel space can overlap signal (documented failure mode).

---

## Library ↔ paper map

| Paper | Library |
|-------|---------|
| `estimate_cross_domain_subspace` | `PMHMatcher` / `estimate_sigma_task_numpy`, `nuisance="subspace"` |
| `project_perp` + classifier | `compare_arms_sklearn` arm `matched` |
| `random_orthonormal` wrong $W$ | arm `wrong_w` |
| D4-iso / domain Gram control | arm `isotropic` |
| `coral_align` | arm `coral` |
| Office-31 feature extract | `pmh.datasets.office31.extract_office31_features` |

[← 13 tasks](index.md) · [Quickstart](../QUICKSTART.md) · [main.pdf](../../main.pdf)

<a id="t01-classical"></a>
<a id="t1-ridge-theorem"></a>
<a id="t1-oracle-mnist-fashion"></a>
<a id="t1-dct-drift"></a>
<a id="t1-baselines-coral"></a>
<a id="t1-svhn"></a>
<a id="t1-ridge-tabular"></a>
<a id="t1-office31"></a>
<!-- legacy -->
<a id="t1-office31-classifiers"></a>
<a id="t1-quick-arrays"></a>
<a id="t1-synthetic-sklearn"></a>
