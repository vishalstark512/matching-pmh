# Paper replication ↔ library alignment

The Grand Unification paper runs **thirteen task-specific pipelines** under `Paper2/T1`–`T7`. Each block fixes:

1. **Which lemma (D1–D7)** defines deployment nuisance \(\Sigma_{\mathrm{task}}\)
2. **How \(\hat W\) / \(\Sigma\) is estimated** from data (signal vs nuisance identification)
3. **How PMH is applied** (Jacobian penalty on \(h\) vs feature projection vs input noise)
4. **Tuning** (rank, \(\sigma\), weight, cap, warmup, task-specific knobs)
5. **Falsification arms** (matched, wrong-\(W\), isotropic, task-specific negatives)

`matching-pmh` is a **principle library**, not a re-run of every script. It must expose the **same estimator semantics and control arms** per \(D_k\), with task recipes that say which path to use.

This document is the contract. When a walkthrough or benchmark disagrees with a row here, the **paper row wins** until the library is fixed.

---

## Two PMH application modes (do not mix blindly)

| Mode | Paper examples | Library API | Falsification |
|------|----------------|-------------|---------------|
| **A. Jacobian penalty** | T2–T7 deep training, most blocks | `PMHLoss` + `PMHTrainer.fit` | `compare_arms` — train separate models per arm |
| **B. Feature projection** | T1 classical (Office-31, ridge) | `MatchedSubspaceProjector`, `compare_arms_sklearn` | Project then linear classifier — **not** the same numbers as mode A |

Mode B was broken for Office-31 before v1.4.1 (random split, wrong isotropic arm, D1 without mean shifts). Fixed protocol: `paper_protocol=True` in `run_sklearn_benchmark`.

---

## Task → library recipe (target state)

| Block | Lemma | Identify nuisance / \(\hat W\) | Apply PMH | Paper tuning (indicative) | Library entry | Falsification arms |
|-------|-------|----------------------------------|-----------|---------------------------|---------------|-------------------|
| **T1** Office-31 | D1 | Class-aligned cross-domain SVD + mean shift | **Mode B** projection | rank 32, pool 200, test 250 | [Recipe T1](recipes/t1-office31-d1.md) · `compare_arms_sklearn`, `paper_protocol=True` | matched, wrong-W⊥W, D4-iso; CORAL baseline |
| **T1** synthetic ridge | D1 | Oracle or estimated W | Mode B | \(\lambda_W,\lambda_\perp\) | `project_onto_complement`, examples | B0, matched, wrong-W, iso |
| **T2A** ViT | D2 | \(\sigma^2 I\) input noise | Mode A on CLS | σ=0.10, w=0.3, cap 0.3 | `nuisance="isotropic"`, `PMHLoss` | ERM vs iso PMH (no wrong-W) |
| **T2B** CheXpert | D2 | \(\sigma^2 I\) on embeddings | Mode A | σ=0.08, w=0.5, warmup 5 | D2 + augment path | aug-only vs embed PMH |
| **T3A** pose | D3† | †Paper aug-delta; **code** gradient-SVD | Mode A along W | r=16, aniso σ=0.05 | `aug_deltas` or custom W | E1 iso, VAT mismatch |
| **T3B** depth | D3 | Photometric aug-delta Gram | Mode A | r=32, wrong-W flag | `collect_augmentation_deltas` | E1, E1_wrong |
| **T4A/B** DA vision | D4 | Per-layer domain Gram | Mode A multilayer | gram_rank 64, layer hooks | [Recipe T4](recipes/t4-domain-d4.md) · `nuisance="domain_shift"`, `MultiLayerPMHLoss` | E1 pixel iso vs multiscale |
| **T5A** QM9 | D5 | Position / node noise covariance | Mode A graph | σ_pos, σ_node, cap | `nuisance_indices`, `perturb` patterns | VAT as mismatch |
| **T5B** clone | D5 | Identifier token block | Mode A on token emb | w=0.5, ramp | D5 indices + HF hook | E1 vs E1S (wrong partition) |
| **T6A** Whisper | D6† | †Temporal spec; **code** content-residual | Mode A on speech emb | λ=0.05, r=32 | `sequences_batches` or custom W | content vs wrong_W |
| **T6B** HAR | D6 | Sensor aug-delta PCA | Mode A | λ=0.03, r=48 | D6 + `collect_sequence_features` | baseline / pmh / wrong_W |
| **T7A** LLM | D7 | Style-pair hidden diffs | Mode A RM/DPO | rank 128, shrink 0.1, w 0.7 | `style_jsonl`, HF trainer | wrong (content Σ), iso, random |
| **T7B** CIFAR PGD | D7 | PGD delta subspace | Mode A + p-mix | r=16, w=0.5, p-sweep | D7 deltas or `collect_pgd` recipe | pgd_W vs random_W |

† = manuscript lemma vs task folder implementation differ; library must document **which estimator the user intends**.

---

## Known semantic gaps (fix in priority order)

### P0 — Estimator fidelity

| Issue | Status |
|-------|--------|
| **Torch D1 requires labels** | **Done** — `estimate_d1(x_src, y_src, x_tgt, y_tgt)`; unlabeled → `estimate_d1_gram_unlabeled` or D4 |
| **Training `wrong_w` ⊥ matched W** | **Done** — `PMHLoss(mode="wrong_w", wrong_seed=...)` |
| **Three meanings of “isotropic”** | **Docs** — training: `trace_iso` (alias `isotropic`); sklearn arm: D4 in `PAPER_ALIGNMENT`; nuisance: D2 |

### P1 — Protocol per block

| Issue | Status |
|-------|--------|
| Block presets (`t1_office31_sklearn`, …) | **Done** — `pmh.benchmark.presets`, `preset=` on compare helpers |
| PyTorch geometry on val | **Done** — `include_geometry=True` on `compare_arms` |
| Multi-seed sklearn | **Done** — `seeds=[...]` |
| Task calibrators (PGD, style, gradient, content-residual) | **Done** — `pmh.calibrate` (thin wrappers) |

### P2 — Task-specific calibration hooks

Expose paper calibration as optional modules (thin wrappers, not copies of full task trees):

- `pmh.calibrate.gradient_subspace` (3A-style, documented as non-default D3)
- `pmh.calibrate.pgd_subspace` (7B)
- `pmh.calibrate.style_gram` (7A)
- `pmh.calibrate.content_residual` (6A)

Each returns `SigmaTaskEstimate` or `W` for `PMHLoss` / `PMHTrainer`.

---

## Principle borrowing checklist (for each new integration)

Before claiming “replicates block Tx”:

1. **Lemma** — Which \(D_k\)? Does `PMHTrainer.estimate(...)` receive the same tensors the paper used (labels, pool vs test, aug list, sequences)?
2. **Identification** — Is \(\hat W\) class-aligned (D1), domain Gram (D4), aug deltas (D3), etc.? Same rank and `n_pairs_per_class`?
3. **Application** — Jacobian (A) or projection (B)? Same hook \(h\)?
4. **Tuning** — weight, cap, warmup, \(\sigma\) from paper `FINAL.md`, not library defaults?
5. **Controls** — Same arms as paper (some blocks omit wrong-W by design)?
6. **Metric** — Accuracy vs TDI vs \(D_N/D_S\) — report what the paper reports?

---

## What v1.4 fixed vs what remains

**Fixed:** sklearn Office-31 protocol (T1 mode B): pool/test split, D1 mean shifts, Lemma-C wrong-W, D4 isotropic control.

**Still generic / misaligned:**

- `estimate_from_config` D1 without labels
- `PMHLoss` wrong-W and isotropic arms vs sklearn arms
- No first-class presets for T2–T7 tuning
- Paper task scripts do not import `pmh` (by design); recipes must bridge manually

See [BENCHMARKS.md](BENCHMARKS.md), [walkthrough 08](walkthroughs/08-falsification-controls.md), and `Paper2/LIBRARY.md`.
