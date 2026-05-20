# The PMH principle (short guide)

Full theory, proofs, and thirteen block results: **[`main.pdf`](../main.pdf)**. This page is the **product spine** — what to estimate, what to train, and how to know it worked.

---

## One object: $\Sigma_{\text{task}}$

**Deploy nuisance** = label-preserving change between train (A) and deploy (B): camera, site, aug stress, style, temporal drift, compositional coordinates, etc.

The paper models it as a random displacement $n$ with covariance $\Sigma_{\text{task}} = \mathrm{Cov}(n)$. CORAL, DANN, augmentation penalties, and metric learning are different **estimators** of related geometry — PMH makes the target explicit: estimate $\hat{\Sigma}_{\text{task}}$, then train so the encoder Jacobian (or feature response) is **matched** to that matrix.

---

## Range matching (what “matched” means)

Training adds a penalty using $\Sigma'$ whose **column space should cover** the nuisance range. When $\Sigma'$ is **matched** to $\hat{\Sigma}_{\text{task}}$, deploy representation drift can fall; when $\Sigma'$ is **isotropic** or **orthogonal to the true nuisance**, the theory predicts specific failures — run those as **controls**, not optional extras.

**Mode A — train:** `PMHLoss` / `PMHTrainer` on hook $h$.  
**Mode B — frozen features:** project onto the complement of matched $W$ (`PMHMatcher`, T1).

---

## Five-step recipe (every task notebook §1–§8)

| Step | You answer | Library |
|------|------------|---------|
| 0 Scope | Same labels on A and B? | `check_applicability()` |
| 1 Identify | Which nuisance family? | `suggest_nuisance()` → D1–D7 |
| 2 Estimate | $\hat{\Sigma}_{\text{task}}$ | `PMHTrainer.estimate()` |
| 3 Apply | Matched PMH | `fit` / `robust_fit` / `PMHLoss` |
| 4 Protocol | Weight, cap, warmup | `PMHConfig` |
| 5 Evidence | Matched vs wrong vs isotropic on **deploy** | `evaluate_robust_fit`, `compare_arms` |

Human-readable checklist: `format_five_step_guide()` from `pmh.recipe`.

---

## Seven estimators (Lemmas D1–D7)

| Lemma | $\Sigma_{\text{task}}$ from… | `nuisance=` |
|-------|------------------------------|-------------|
| D1 | Cross-domain subspace (labeled features) | `subspace` |
| D2 | Isotropic noise level | `isotropic` |
| D3 | Augmentation-induced deltas | `augmentation` |
| D4 | Source−target feature Gram (**class-aligned** when labels exist) | `domain_shift` |
| D5 | Named coordinate blocks | `compositional` |
| D6 | Temporal / content-residual | `temporal` |
| D7 | Style or attack directions | `style` |

**D4 default (library):** `PMHTrainer` zips source/target loaders and, when batches are `(x, y)`, aligns target rows by class before Gram — same protocol as paper T4. Unlabeled batches fall back to pooled zip.

**T4B multiscale (paper):** per-layer aligned Gram + feature-diff train — `PMHTrainer(train_mode="feature_diff")` + `estimate_multilayer()`, or `pmh.vision.domain_multilayer` helpers. T4A notebook stays single-hook Jacobian D4.

---

## Three layers in this repo

| Layer | What you use | When |
|-------|--------------|------|
| **Theory** | `main.pdf`, `paper_code/T*/**/FINAL.md` | Proofs, block numbers, reproduction |
| **Library** | `pmh` trainers, estimators, `compare_arms`, benchmarks | Ship on your stack |
| **Demos** | `notebooks/tasks/`, `docs/tasks/` | Copy §8 into your pipeline |

**Library vs paper:** Headline metrics in the paper and task docs (e.g. DomainNet +3 pp, Cityscapes mIoU) are produced by **`paper_code/`** — block-specific reproduction scripts, not by `pip install matching-pmh` alone. The **library** implements the same estimators and Step 5 protocol for **your** data; you should expect to **tune and validate** (hook, rank, loss scale, data volume) rather than match `FINAL.md` numbers on first run. Notebooks use **demo loaders** unless noted; see each task page “Paper vs library” note.

---

## Evidence you should expect

- **12/13 blocks** pass pre-registered criteria in `paper_code` (see block `FINAL.md` files).
- **Office-31 (T1):** documented **D1 eigengap failure** — PMH does not silently “fix” every domain benchmark; wrong estimator or weak gap means stop or change nuisance family.
- **Step 5 is mandatory:** matched must beat wrong-direction and isotropic on deploy holdout before you ship.

```python
from pmh import evaluate_robust_fit, format_five_step_guide
print(format_five_step_guide("domain_shift"))
```

Benchmark-style arms (matched / wrong / isotropic + geometry): `run_benchmark_protocol` in `pmh.benchmark.protocol`.

---

## Choose your depth

| Goal | Start here |
|------|------------|
| Pick nuisance + five steps | [WHEN_PMH_HELPS.md](WHEN_PMH_HELPS.md) · [tasks/index.md](tasks/index.md) |
| PyTorch train + estimate | [QUICKSTART.md](QUICKSTART.md) · `PMHTrainer` |
| Frozen sklearn / Office-31 | [tasks/t01-classical.md](tasks/t01-classical.md) · `compare_arms_sklearn` |
| Multilayer vision (T4B) | [tasks/t04b-multilayer-vision.md](tasks/t04b-multilayer-vision.md) · `pmh.vision.domain_multilayer` |
| Full block reproduction | `paper_code/` + `main.pdf` |
| API reference | [api/index.md](api/index.md) |
