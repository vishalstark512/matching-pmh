# Correct usage: paper principles in the library

This guide explains **how to use matching-pmh correctly** so your run matches the Grand Unification paper’s rigor—not a generic “add a penalty” plugin.

**Related:** [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) (task map) · [BENCHMARKS.md](BENCHMARKS.md) · [Walkthrough 8 — Controls](walkthroughs/08-falsification-controls.md)

---

## 1. Two ways PMH is applied (pick one)

| Mode | When | API | Falsification |
|------|------|-----|----------------|
| **A. Jacobian penalty** | End-to-end training (ViT, Whisper, LLM, …) | `PMHTrainer` + `PMHLoss` | `compare_arms(..., include_geometry=True)` |
| **B. Feature projection** | Frozen embeddings + linear classifier (T1 / Office-31) | `MatchedSubspaceProjector` or `compare_arms_sklearn` | `preset='t1_office31_sklearn'` |

**Do not** compare Mode A accuracy to Mode B tables—they are different protocols.

---

## 2. Three different “isotropic” names

| Name | Meaning | How to invoke |
|------|---------|----------------|
| **D2 nuisance** | Known \(\sigma^2 I\) on representations | `nuisance="isotropic"`, `SigmaTaskConfig.for_isotropic(dim, noise_level)` |
| **Training arm `trace_iso`** | Falsification: \((\mathrm{tr}\,\Sigma/d)\,I\) from **matched** \(\Sigma\) | `PMHLoss(..., mode="trace_iso")` or `mode="isotropic"` (alias) |
| **Sklearn arm `isotropic`** | Falsification: project out **D4 domain Gram** (unmatched nuisance) | `compare_arms_sklearn` only |

Using the wrong one invalidates Lemma C comparisons.

---

## 3. Phase A: estimate the right \(\Sigma_{\mathrm{task}}\) (Lemma D1–D7)

### D1 — class-aligned subspace (T1, Office-31 with labels)

**Requires labels on source and target** (for estimation pool, not necessarily for B0 training).

```python
from pmh import SigmaTaskConfig, estimate_from_config

artifact = estimate_from_config(
    SigmaTaskConfig.for_subspace(rank=32),
    h_src, y_src, h_tgt_pool, y_tgt_pool,  # pool only — no test leakage
    n_pairs_per_class=40,
    seed=0,
)
# artifact.metadata["w"]  →  [d, r] basis
```

**Wrong:** `estimate_from_config(D1, h_src, h_tgt)` with two tensors only → use **`method="D4"`** for unlabeled Gram.

### D4 — domain shift (default PyTorch domain adaptation)

```python
trainer = PMHTrainer(model, hook=..., nuisance="domain_shift", rank=16)
trainer.estimate(source_batches, target_batches=...)
```

### D2, D3, D5, D6, D7

See [nuisance_types.md](nuisance_types.md). Task-specific calibration:

| Paper block | Prefer | Module |
|-------------|--------|--------|
| T3A (gradient W) | Gradient-SVD, not default D3 aug | `pmh.calibrate.gradient_subspace_numpy` |
| T6A Whisper | Content-residual W | `pmh.calibrate.content_residual_subspace` |
| T7A LLM style | Style-pair Gram | `pmh.calibrate.style_gram_from_deltas` |
| T7B PGD ViT | Stacked PGD deltas | `pmh.calibrate.subspace_artifact_from_deltas` |

---

## 4. Phase B: training with falsification arms

### Matched

```python
PMHLoss(artifact, mode="matched", config=PMHConfig(weight=0.3, cap_ratio=0.3))
```

Use **paper tuning** from the block’s `FINAL.md`, not only defaults.

### Wrong-W (Lemma C)

Random subspace **orthogonal to matched \(W\)**:

```python
PMHLoss(artifact, mode="wrong_w", wrong_rank=32, wrong_seed=0)
```

### Trace-isotropic control (training)

```python
PMHLoss(artifact, mode="trace_iso")  # preferred name
```

### Full PyTorch comparison

```python
from pmh import compare_arms

compare_arms(
    trainer.artifact_,
    model_factory,
    setup_model,
    train_loader,
    val_loader,
    preset="t4_domain_d4",       # optional paper defaults
    include_geometry=True,       # TDI_cls, D_N/D_S on val embeddings
    epochs=15,
    report_dir="results/arms",
)
```

---

## 5. Sklearn / frozen features (T1)

### Correct Office-31 protocol

```python
from pmh import compare_arms_sklearn

result = compare_arms_sklearn(
    x_amazon, y_a,
    x_dslr_full, y_d,           # full target; split inside
    preset="t1_office31_sklearn",
    seeds=[0, 42, 142],         # optional multi-seed mean
    include_geometry=True,
)
```

**Inside the preset:**

- Train on ≤1500 source rows  
- Estimate \(\hat W\) on **200** target pool rows only  
- Test on **250** held-out target rows  
- rank=32, 40 pairs/class, D1 with class-mean shifts  
- wrong-W ⊥ \(W\); isotropic arm = D4 directions  

### Regenerate reference table

```bash
pip install -e ".[sklearn,vision]"
python scripts/generate_reference_benchmark.py \
  --office31-root /path/to/office31 \
  --output docs/benchmarks/office31_amazon_to_dslr.md
```

Expect ~22% accuracy scale (not ~70%—that was a broken protocol).

---

## 6. Paper block presets

```python
from pmh.benchmark import get_preset, list_presets

print(list_presets())
p = get_preset("t1_office31_sklearn")
print(p.description, p.sklearn_benchmark, p.pmh_config)
```

| Preset | Block | Lemma | Mode |
|--------|-------|-------|------|
| `t1_office31_sklearn` | T1 Office-31 | D1 | projection |
| `t2a_vit_isotropic` | T2A ViT | D2 | jacobian (no wrong-W) |
| `t4_domain_d4` | T4 DomainNet | D4 | jacobian |
| `t7a_style_d7` | T7A LLM | D7 | jacobian |
| `t7b_pgd_d7` | T7B CIFAR | D7 | jacobian + PGD calibrator |

Pass to `compare_arms(..., preset=...)` or `compare_arms_sklearn(..., preset=...)`.

---

## 7. Checklist before you claim “PMH worked”

1. Wrote the **deployment sentence** (label fixed, nuisance named).  
2. Chose the **correct Dk** and estimator (labeled D1 vs D4, etc.).  
3. **No test leakage** into \(\hat W\) (pool vs test for Office-31).  
4. Ran **B0, matched, wrong-W, isotropic** (as appropriate for block).  
5. **Matched** beats B0 on your metric *and* wrong-W does not beat matched on both accuracy and geometry.  
6. Reported **preflight** / eigengap when using D1/D4.  
7. Used **paper hyperparameters** (rank, σ, weight, cap), not only library defaults.

---

## 8. What the library does not do automatically

- Re-run all thirteen `Paper2/T*` scripts (they remain the replication source).  
- Pick nuisance type for you without `suggest_nuisance` / your sentence.  
- Guarantee matched beats B0 on Office-31 (paper predicts CORAL can win on linear heads).  

The library gives **faithful estimators, arms, and protocols**; your task still needs the right data and tuning from the paper block.
