# Walkthrough 8: Falsification controls — full guide

**At a glance**

| | |
|---|---|
| **Purpose** | Prove gains come from **matched** Σ_task, not generic regularization |
| **Stack** | Any (PyTorch `PMHLoss` or sklearn arms) |
| **Scripts** | `examples/04_falsification_controls.py` · `06_office31_sklearn.py` · `20_compare_training_arms.py` |
| **Required** | Before any paper/blog claim “PMH improved our model” |

[Adaptation workbook](../ADAPTATION_WORKBOOK.md) · [BENCHMARKS.md](../BENCHMARKS.md)

---

## Who this is for

**Everyone.** If you skip this walkthrough, reviewers and you cannot tell whether:

- matched PMH helped because you estimated the **right** nuisance geometry, or
- **any** Jacobian penalty would have helped.

---

## The four arms (what each proves)

| Arm | PyTorch `PMHLoss(..., mode=)` | sklearn benchmark key | If matched is real… |
|-----|------------------------------|------------------------|---------------------|
| **B0** | (no PMH) | `b0` | Matched should beat B0 on **target** metric |
| **Matched** | `"matched"` | `matched` | Your Dk estimate |
| **Wrong-W** | `"wrong_w"` | `wrong_w` | Should **not** beat matched |
| **Isotropic** | `"isotropic"` | `isotropic` | Should **not** beat matched |
| **CORAL** (sklearn) | — | `coral` | Optional baseline estimator |

**Strong claim:** matched > B0 **and** matched > wrong-W ≈ isotropic.

**Weak claim (suspect):** matched > B0 but wrong-W also wins → likely generic shrinkage.

---

## Path A — PyTorch toy (one batch)

```bash
python examples/04_falsification_controls.py
```

Shows PMH scalar for three modes on one forward pass — useful for wiring check, **not** enough for publication.

### Full training comparison

```python
from pmh import compare_arms, PMHTrainer, PMHConfig

# After trainer.estimate() or trainer.fit Phase A:
compare_arms(
    trainer.artifact_,
    model_factory=lambda: build_your_model(),
    setup_model=your_setup_fn,       # returns encoder, head, optimizer
    train_loader=YOUR_TRAIN_LOADER,
    val_loader=YOUR_TARGET_VAL_LOADER,
    epochs=10,
    report_dir="results/arms",
)
```

Read `results/arms/benchmark.md`.

---

## Path B — sklearn frozen features

```bash
python examples/06_office31_sklearn.py --report results/sklearn_arms
# or
python examples/21_benchmark_sklearn_table.py --report results/bench1
```

```python
from pmh import compare_arms_sklearn

compare_arms_sklearn(x_src, y_src, x_tgt, y_tgt, rank=16, report_dir="results/YOUR_RUN")
```

Table includes **target_accuracy**, **TDI_cls**, **D_N/D_S**.

---

## Path C — Manual `PMHLoss` loop

```python
from pmh import PMHLoss, PMHConfig

cfg = PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2)

for mode in ("matched", "wrong_w", "isotropic"):
    pmh = PMHLoss(artifact, cfg, mode=mode)
    # train separate model copies; same epochs, seed, data
    # log YOUR_TARGET_METRIC per mode
```

Do **not** reuse the same weights across arms — train from the same init per arm.

---

## Adaptation worksheet

| Question | Your answer |
|----------|-------------|
| Target validation loader | |
| Metric (accuracy / F1 / WER / …) | |
| Rank for wrong-W / isotropic | same as matched |
| Number of seeds | ≥ 1 (3+ for publication) |
| Where reports are saved | `results/` (gitignored) |

---

## Reporting checklist (copy into your experiment log)

```markdown
## PMH falsification — YOUR_PROJECT

- Nuisance sentence: "..."
- Estimator: D?
- Hook / features: ...
- Target metric: ...

| Arm | Target metric | TDI_cls | Notes |
|-----|---------------|---------|-------|
| B0 | | | |
| matched | | | |
| wrong_w | | | |
| isotropic | | | |
```

---

## Verify success

- [ ] All four arms ran with **same** `rank`, data, and training budget.
- [ ] Evaluation on **target-like** data (not train-source only).
- [ ] wrong-W and isotropic do not clearly beat matched.
- [ ] Report saved under `results/` (not committed if large).

---

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Only compare B0 vs matched | Add wrong-W and isotropic |
| Tune hyperparams separately per arm | Same grid / same defaults except `mode` |
| Source-domain validation | Use target-domain val |
| Single seed | Report variance or multiple seeds |

---

## Next steps

- [1 — PyTorch D4](01-pytorch-domain-d4.md) — training setup
- [3 — sklearn / Office-31](03-office31-sklearn-d1.md) — feature pipeline
- [17 — Compare arms template](17-compare-arms-your-pipeline.md)
