# Walkthrough 3: Frozen features + sklearn (D1) — full guide

**At a glance**

| | |
|---|---|
| **Estimator** | D1 — cross-domain subspace (labels on source **and** target) |
| **Stack** | NumPy features + sklearn classifier |
| **Script** | `examples/06_office31_sklearn.py` · `examples/21_benchmark_sklearn_table.py` |
| **Time** | ~1 min synthetic; ~15 min with ResNet feature extraction on real images |
| **API** | `PMHMatcher`, `compare_arms_sklearn`, optional `Pipeline` / `GridSearchCV` |

[Adaptation workbook](../ADAPTATION_WORKBOOK.md) · [sklearn.md](../sklearn.md) · [BENCHMARKS.md](../BENCHMARKS.md)

---

## Who this is for

Use when:

- You already have (or can export) **fixed embeddings** `x ∈ R^d` per sample.
- You train a **sklearn** (or sklearn-like) classifier on top.
- You have **labels on both** source and target for D1 (or use D4 / `domain_shift` with `PMHMatcher.fit(x_src, x_tgt)` only).

Skip to [Walkthrough 1](01-pytorch-domain-d4.md) if you fine-tune the encoder with PyTorch.

---

## Prerequisites

```bash
pip install "matching-pmh[sklearn]"
# For ResNet feature extraction from image folders:
pip install "matching-pmh[sklearn,vision]"
```

| Requirement | Your notes |
|-------------|------------|
| `x_source`, `y_source` | shape `[N, d]` |
| `x_target`, `y_target` | same `d`, aligned label semantics |
| Classifier | LogisticRegression (default) or SVM |

**Data policy:** do not commit Office-31 or `.npy` files — keep data outside the repo ([DATA_POLICY.md](../DATA_POLICY.md)).

---

## Your nuisance sentence

Examples for **D1** / Office-31 style:

- *“Product photos from Amazon vs DSLR; object category unchanged.”*
- *“CRM exports from EU vs US offices; churn label definition unchanged.”*

---

## Step 1 — Get features (three paths)

### Path A — Run synthetic demo (no download)

```bash
python examples/06_office31_sklearn.py
```

Uses `synthetic_office31_features()` built into the library.

### Path B — Real Office-31 images (local only)

```bash
# YOUR_OFFICE31_ROOT is outside the repo, e.g. D:/datasets/office31
python examples/06_office31_sklearn.py \
  --office31-root YOUR_OFFICE31_ROOT \
  --source amazon --target dslr \
  --max-samples 2000
```

Extracts **ResNet-18** 512-d features via `pmh.datasets.office31.extract_office31_features`.

### Path C — Your own encoder (recommended for production)

```python
# Pseudocode — run once, save outside git
import numpy as np
# h = your_encoder(batch)  # [B, d]
# np.save("features/site_a.npy", ...)
# np.save("labels/site_a.npy", ...)
```

Then load `.npy` in your training script.

---

## Step 2 — Estimate + compare arms

```python
from pmh import PMHMatcher, compare_arms_sklearn, suggest_nuisance

print(suggest_nuisance(
    has_source_labels=True,
    has_target_labels=True,
    has_target_domain=True,
))  # often suggests subspace / D1

matcher = PMHMatcher(nuisance="subspace", rank=16, seed=0)
matcher.fit(x_source, y_source, x_target, y_target)
print(matcher.artifact_.preflight, matcher.artifact_.eigengap)

result = compare_arms_sklearn(
    x_source, y_source, x_target, y_target,
    rank=16,
    include_coral=True,
    report_dir="results/YOUR_RUN",   # gitignored
)
```

Open `results/YOUR_RUN/benchmark.md` — table with **target accuracy**, **TDI_cls**, **D_N/D_S**.

---

## Step 3 — sklearn Pipeline (production pattern)

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pmh import PMHMatcher, default_pmh_param_grid, grid_search_pmh_pipeline

pipe = Pipeline([
    ("pmh", PMHMatcher(nuisance="subspace", rank=16, X_target=x_target)),
    ("clf", LogisticRegression(max_iter=500)),
])
pipe.fit(x_source, y_source)

# Or grid search rank:
grid_search_pmh_pipeline(x_source, y_source, x_target, y_target, cv=5)
```

Details: [sklearn.md](../sklearn.md)

---

## Step 4 — What each arm means

| Arm | What it tests |
|-----|----------------|
| **b0** | Source train → target test (no geometry) |
| **matched** | D1 subspace projection then classify |
| **wrong_w** | Random subspace — must not beat matched |
| **isotropic** | Non-matched high-variance directions |
| **coral** | CORAL alignment baseline (related estimator) |

---

## Run benchmark table example

```bash
python examples/21_benchmark_sklearn_table.py
python examples/21_benchmark_sklearn_table.py --office31-root YOUR_OFFICE31_ROOT --report results/bench1
```

Copy **only** the markdown summary into docs if publishing reference numbers — not raw features.

---

## Adaptation worksheet

| Office-31 example | Your project |
|-------------------|--------------|
| ResNet-18 512-d | Your CLIP / ViT / custom `d` |
| amazon → dslr | site A → site B |
| `rank=16` | Tune 8–32 via `grid_search_pmh_pipeline` |
| LogisticRegression | Your sklearn / XGBoost on `matcher.transform(x)` |

---

## Verify success

- [ ] `compare_arms_sklearn` completes; `benchmark.md` written.
- [ ] **matched** target accuracy ≥ **wrong_w** (ideally > **b0**).
- [ ] **TDI_cls** lower for matched than b0 (geometry; see [BENCHMARKS.md](../BENCHMARKS.md)).
- [ ] Honest comparison to **coral** if claiming SOTA.

---

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Labels not aligned across domains | D1 needs same class semantics |
| Test on source only | Report **target-domain** accuracy |
| `marginal` preflight | More samples or higher `rank`; still run controls |
| Committing `.npy` / images | [DATA_POLICY.md](../DATA_POLICY.md) |

---

## Fine-tune encoder later

1. Use this walkthrough to validate **geometry + controls** on frozen `h`.
2. Switch to [Walkthrough 1](01-pytorch-domain-d4.md) with `PMHLoss` on the **same** hook layer.

---

## Next steps

- [8 — Controls](08-falsification-controls.md)
- [17 — Compare arms on your PyTorch model](17-compare-arms-your-pipeline.md)
- [2 — ResNet feature extraction](02-resnet-vision-d4.md)
