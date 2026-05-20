# Identification fidelity by subtype (D1–D7)

This page answers: **“If I pick subtype Dk, does the default library estimator match the paper exemplar, or do I need a calibrator?”**

The product model is **subtype first** ([NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md)). Paper blocks T1–T7 are **identification refinements** within a subtype—not separate products.

**Legend**

| Tag | Meaning |
|-----|---------|
| **Strong** | Default `nuisance=` / `SigmaTaskConfig` path matches the cited paper script on the same data contract |
| **Partial** | Same subtype structure; use `pmh.calibrate.*` or a preset flag for paper-faithful \(W\) |
| **Bring your own** | Library applies PMH once you supply indices / deltas / trajectories |

---

## Summary table

| Subtype | Default estimator | Fidelity vs paper exemplar | When to refine |
|---------|-------------------|----------------------------|----------------|
| **D1** | `nuisance="subspace"` · class-aligned cross-domain SVD | **Strong** for T1 sklearn (`paper_protocol=True`) | Rank, shrinkage; hook choice for deep D1 |
| **D2** | `nuisance="isotropic"` · σ²I | **Strong** for T2A/B ERM-vs-PMH arms | Set σ from domain knowledge |
| **D3** | `nuisance="augmentation"` · Gram over mode deltas | **Strong** for T3B (finite aug modes) | T3A gradient-SVD → `gradient_subspace_numpy` |
| **D4** | `nuisance="domain_shift"` · centred source−target Gram | **Strong** for single-hook T4 | Multilayer T4 → `MultiPMHLoss` / multi-hook trainer |
| **D5** | `nuisance="compositional"` · block on `nuisance_indices` | **Bring your own** indices | T5A/B validated structure, not auto-inferred |
| **D6** | `nuisance="temporal"` · sequence / consecutive scatter | **Partial** for T6B HAR | T6A Whisper → `content_residual_subspace` |
| **D7** | `nuisance="style"` · style-pair / corpus Gram | **Strong** for T7A JSONL / two corpora | T7B PGD δ stack → `subspace_artifact_from_deltas` |

---

## D1 — Cross-domain subspace

| | |
|---|---|
| **Default** | `SigmaTaskConfig.for_subspace(rank=…)` via `estimate_from_config` or `PMHMatcher` / `nuisance="subspace"` |
| **Paper exemplar** | T1 Office-31 ridge + projection |
| **Fidelity** | **Strong** when `compare_arms_sklearn(..., paper_protocol=True)` (pool-on-source, test-on-target split—matches `Paper2/T1`) |
| **Refinement** | Rank grid; frozen ResNet pool vs end-to-end \(h\) (then use Mode A + D4/D1 on hook) |

```python
from pmh.benchmark.sklearn_protocol import compare_arms_sklearn

compare_arms_sklearn(x_src, y_src, x_tgt, y_tgt, paper_protocol=True, rank=32)
```

---

## D2 — Isotropic

| | |
|---|---|
| **Default** | `SigmaTaskConfig.for_isotropic(sigma=…)` · `nuisance="isotropic"` |
| **Paper exemplar** | T2A ViT CLS |
| **Fidelity** | **Strong** for Jacobian isotropic penalty (no learned \(W\) directions) |
| **Refinement** | σ from calibration or ablation grid |

---

## D3 — Augmentation modes

| | |
|---|---|
| **Default** | `collect_augmentation_deltas` + `nuisance="augmentation"` |
| **Paper exemplar** | T3B depth / photometric modes |
| **Fidelity** | **Strong** when you pass the same finite mode stack as the paper |
| **Refinement (T3A)** | Input-gradient SVD is still **D3 structurally**, different identification: |

```python
from pmh.calibrate import gradient_subspace_numpy

artifact = gradient_subspace_numpy(gradients, rank=16)  # [N, d] or batched
```

---

## D4 — Domain Gram

| | |
|---|---|
| **Default** | Pooled source vs target batches, centred Gram, optional rank truncate |
| **Paper exemplar** | T4 vision DA (single hook) |
| **Fidelity** | **Strong** for `PMHTrainer(..., nuisance="domain_shift")` with matched source/target loaders |
| **Refinement** | T4B multilayer: register multiple hooks and use `MultiPMHLoss` / `MultiLayerPMHLoss` patterns in [walkthrough 04](walkthroughs/04-multilayer-convnet.md) |

Closed-form check (synthetic): `tests/test_subtype_fidelity.py` compares library Gram to hand-derived formula and D1 \(W\) to `Paper2/T1/classical_pmh/common.py` when available.

---

## D5 — Compositional

| | |
|---|---|
| **Default** | `nuisance="compositional"`, `nuisance_indices=(...)` |
| **Paper exemplar** | T5A QM9 positions, T5B code tokens |
| **Fidelity** | **Bring your own** — library does not infer which coordinates are nuisance |
| **Refinement** | None beyond correct index map and rank |

---

## D6 — Temporal / sequence

| | |
|---|---|
| **Default** | `nuisance="temporal"`, `sequences_batches=` or `collect_sequence_features` |
| **Paper exemplar** | T6B HAR trajectories |
| **Fidelity** | **Strong** when identification is temporal-difference scatter (T6B-style) |
| **Refinement (T6A)** | Content vs temporal residual subspace: |

```python
from pmh.calibrate import content_residual_subspace

W = content_residual_subspace(sequences, rank=16, source="content")  # or "temporal"
```

---

## D7 — Style / alignment

| | |
|---|---|
| **Default** | `nuisance="style"`, `style_jsonl`, or HF two-corpus estimate |
| **Paper exemplar** | T7A LLM format shift |
| **Fidelity** | **Strong** for style-pair / paired-corpus Gram (T7A) |
| **Refinement (T7B)** | PGD representation deltas: |

```python
from pmh.calibrate import subspace_artifact_from_deltas

artifact = subspace_artifact_from_deltas(deltas, rank=16)  # [N, d] stack
```

Also: `style_gram_from_deltas`, `style_gram_from_jsonl` in `pmh.calibrate.style`.

---

## Controls (all subtypes)

Falsification is **not** subtype-specific: matched \(W\) vs wrong-\(W\) (⊥ matched) vs isotropic negatives. See [walkthrough 08](walkthroughs/08-falsification-controls.md).

The library implements **lemmas** (estimate → PMH → controls). Task scripts in `Paper2/` add benchmark protocol and identification choices within a subtype.

---

## Related

| Doc | Purpose |
|-----|---------|
| [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) | Pick D1–D7 |
| [NUISANCE_SUBTYPE_PLAN.md](NUISANCE_SUBTYPE_PLAN.md) | Rollout phases |
| [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) | Block → subtype map (research tab) |
| [recipes/](recipes/README.md) | Exemplar cards (subtype primary) |
