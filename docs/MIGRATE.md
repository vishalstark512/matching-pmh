# Migrate from what you already use

PMH is not “another regularizer.” It estimates **how deploy changes your representations** (same labels), trains a **matched** penalty, and requires **controls** on deploy holdout before you ship.

You do **not** need to name D1–D7 — pass data flags to `robust_fit` / `try_pmh` or run `infer_applicability()`.

---

## CORAL or linear domain adaptation (sklearn)

**Today:** fit CORAL on source features, then classify.

**PMH:** matched subspace projection + falsification arms.

```python
from pmh import evaluate_baseline_vs_pmh, load_g2_demo_arrays

xs, ys, xt, yt = load_g2_demo_arrays()
report = evaluate_baseline_vs_pmh(xs, ys, xt, yt, compare_to=("coral",))
print(report.deploy_summary())
```

**Honest expectation:** On frozen Office-31-style setups, CORAL can beat projection-only PMH on accuracy — that is a documented falsification case, not a bug. Step 5 still tells you whether **matched** beats **wrong-direction** and **isotropic** on *your* holdout.

Demo: `python scripts/demos/office31_sklearn.py --office31-root PATH`

---

## PyTorch + site/camera shift

**Today:** train ERM on site A; hope site B works.

**PMH:**

```python
from pmh import try_pmh

report = try_pmh(model, train_loader, val_loader,
                 source_batches=src_loader, target_batches=tgt_loader,
                 hook="auto", epochs=10)
print(report.ship_verdict())
```

CLI equivalent: `pmh-train try` (add `--quick` for a ~1 min smoke run).

`hook="auto"` picks a backbone layer. `nuisance=None` picks shift type from flags (default: unlabeled target → `domain_shift`).

---

## Heavy augmentation only

**Today:** train with blur/crop/jitter.

**When deploy shift is *not* in your aug list** (new camera, new hospital), augmentation does not estimate deploy geometry. Use target-site batches:

```python
robust_fit(model, train_loader,
           source_batches=src_loader, target_batches=deploy_loader,
           has_augmentation_modes=False, has_target_domain=True)
```

If your deploy stress *is* a known transform list, set `has_augmentation_modes=True`.

---

## Hugging Face `Trainer`

**Style / format shift (same facts, different surface):** style-pair JSONL + `estimate_style_sigma` — [T7A notebook](../notebooks/tasks/t07a-llm-style.ipynb).

**Two corpora (same labels):** `robust_fit_text_domains(model, tokenizer, train_loader, source_texts, target_texts)`.

Full DPO + margin PMH: `paper_code/T7/task7B/` (library: `estimate_pgd_subspace_from_model` for the estimate phase).

---

## Adversarial / PGD robustness (T7B)

**Library estimate (hook representation deltas):**

```python
from pmh.calibrate.pgd import estimate_pgd_subspace_from_model

artifact = estimate_pgd_subspace_from_model(
    model, hook="enc", head=model.head, source_batches=train_loader,
    rank=16, epsilon=0.1, steps=3,
)
# attach artifact → PMHTrainer.from_artifact(...) for training
```

Paper-scale DPO runs: `paper_code/T7/task7B/`.

---

## Multilayer vision (T4B)

When shift is visual domain and you want **per-layer** matched Gram + feature-diff (not single-hook Jacobian):

```python
trainer = PMHTrainer(
    model, train_mode="feature_diff",
    forward_features=model.forward_features,
    layer_names=("conv1", "conv2"),
    head_layer="conv2", head=model.head,
    nuisance="domain_shift",
)
trainer.estimate_multilayer(src_loader, tgt_loader)
trainer.fit(train_loader, source_batches=src_loader, target_batches=tgt_loader)
```

Notebook: [t04b-multilayer-vision](../notebooks/tasks/t04b-multilayer-vision.ipynb)  
Smoke: `PMH_QUICK=1 python scripts/smoke_t04b_multilayer.py`

---

## Loss scale (all stacks)

PMH must stay **~5--30% of task loss** during training (hard cap at `pmh_max_task_ratio`, default 25--30%). Use `PMHConfig.golden_path()` — see [LOSS_SCALING.md](LOSS_SCALING.md).

---

## Decision checklist

1. Same label semantics on A and B? If no → PMH is not the right tool.  
2. Run `try_pmh` or `evaluate_baseline_vs_pmh` with **Step 5** controls.  
3. Read `report.ship_verdict()` — only ship if matched beats wrong-direction **and** isotropic.  
4. Compare to your baseline (CORAL, ERM) in the same table — do not cherry-pick one metric.

More: [WHEN_PMH_HELPS.md](WHEN_PMH_HELPS.md) · [PRINCIPLE.md](PRINCIPLE.md)
