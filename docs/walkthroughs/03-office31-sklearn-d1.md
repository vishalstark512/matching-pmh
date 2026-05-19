# Walkthrough 3: Office-31 / frozen features (D1)

**Goal:** Domain shift with **frozen embeddings** and a classical classifier (logistic / SVM)—the paper’s Office-31 protocol.

**Estimator:** D1 (cross-domain subspace) on features; matched PMH implemented as **subspace projection** before the classifier.  
**Script:** `examples/06_office31_sklearn.py`

---

## Prerequisites

```bash
pip install "matching-pmh[sklearn,vision]"
```

---

## Step 1 — Name nuisance

*“Amazon product photos vs DSLR photos; object category unchanged.”* → low-rank domain subspace → **D1** (and related D4 Gram).

---

## Step 2 — Get features $h \in \mathbb{R}^d$

**Option A — synthetic (no download):**

```bash
python examples/06_office31_sklearn.py
```

**Option B — real Office-31 ResNet-18 features:**

```bash
python examples/06_office31_sklearn.py --office31-root /path/to/office31 --classifier svm
```

The script calls `pmh.datasets.office31.extract_office31_features` (ResNet-18, 512-d).

---

## Step 3 — Estimate and preflight

Inside the script:

```python
from pmh.numpy_api import estimate_sigma_task_numpy
from pmh.config import SigmaTaskConfig

artifact = estimate_sigma_task_numpy(
    x_src, y_src, x_tgt, y_tgt,
    config=SigmaTaskConfig.for_subspace(rank=16),
)
print(artifact.preflight, artifact.eigengap)
```

**Paper note:** amazon→dslr often shows `marginal` eigengap (~1.03); matched PMH can still beat wrong-W but may trail CORAL on raw accuracy—report controls.

---

## Step 4 — Train classifiers on projected features

The example compares:

| Arm | What it does |
|-----|----------------|
| **B0** | Raw source features → test target |
| **Matched** | `MatchedSubspaceProjector` (D1 geometry) |
| **Wrong-W** | Random rank-$r$ subspace complement |
| **CORAL** | Align source covariance to target (baseline estimator of related geometry) |

```bash
python examples/06_office31_sklearn.py --rank 16 --classifier logistic
```

---

## Step 5 — Interpret results

Credible matched claim on this benchmark:

- Matched beats **wrong-W** on target accuracy
- Compare honestly to **CORAL** (another $\Sigma_{\mathrm{task}}$ estimator)

---

## Adapt to your embeddings

| Office-31 | Your setup |
|-----------|------------|
| ResNet-18 512-d | Your frozen ViT / CLIP / custom encoder |
| `x_src`, `x_tgt` `.npy` | Save `collect_features` output once, reuse |

For end-to-end fine-tuning with `PMHLoss` on the same $h$, continue from Walkthrough 1 or 2 after estimation.
