# Integrate your project

**Before this page:** [What is PMH?](WHAT_IS_PMH.md) → [First hour](FIRST_HOUR.md) → [Shift types D1–D7](NUISANCE_SUBTYPES.md) → [Golden paths](GOLDEN_PATHS.md).

**Goal:** wire PMH into **your** repo in one afternoon.

**Paper / benchmarks:** [Research → Paper alignment](PAPER_ALIGNMENT.md) — only after a basic run works.

---

## Step 0 — One sentence (required)

What changes at deployment **without changing the label**?

> *Example:* “Images from a new hospital camera, but the disease label still means the same thing.”

If that does not describe your problem, see [When PMH helps](WHEN_PMH_HELPS.md) and [What is PMH — when not to use](WHAT_IS_PMH.md#when-not-to-use-it).  
Pick subtype D1–D7: [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) or `pmh-train wizard`.

---

## Step 1 — Pick subtype + stack (2 minutes)

| Step | Action |
|------|--------|
| Shift type | [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) decision tree or `suggest_subtype()` |
| Code path | [GOLDEN_PATHS.md](GOLDEN_PATHS.md) G1 (PyTorch) / G2 (sklearn) / G3 (HF) |
| Examples | [Gallery](gallery/README.md) by domain |

---

## Step 2 — Install

```bash
pip install matching-pmh torch
# optional:
pip install "matching-pmh[sklearn]"   # classical ML path
pip install "matching-pmh[hf]"        # D7 style JSONL
pip install "matching-pmh[vision]"    # ResNet / timm examples
```

Verify:

```bash
python -c "import pmh; print(pmh.__version__)"
pmh-train doctor
pmh-train list-methods
```

Estimate from folders: [DATA_LAYOUT.md](DATA_LAYOUT.md) · `pmh-train estimate --source-dir ... --target-dir ...`

---

## Step 3 — Minimal working example (PyTorch)

This is the **recommended** API: one object, Phase A + B.

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    your_model,
    hook=your_backbone,          # nn.Module or "avgpool" / path string
    head=your_classifier,        # optional
    nuisance="domain_shift",     # or "auto" + data flags
    pmh_config=PMHConfig.balanced(),
    artifact_path="artifacts/my_sigma.pt",
)

trainer.fit(
    your_train_loader,
    source_batches=loader_site_a,   # deployment A (estimate)
    target_batches=loader_site_b,   # deployment B (estimate)
    epochs=20,
)

print("preflight:", trainer.artifact_.preflight)  # pass / marginal / fail
```

**Rules that matter:**

1. **`hook` must be the same** in estimate and train (same layer, same `d`).
2. **Save** `artifact_path` when you change data or hook.
3. Use **`PMHConfig.balanced()`** first; tune later.

→ Hooks: [hooks.md](hooks.md) · optional worksheets: [Adaptation workbook](ADAPTATION_WORKBOOK.md)

---

## Step 4 — Minimal example (sklearn / frozen features)

```python
from pmh import PMHMatcher, suggest_nuisance, compare_arms_sklearn

# Optional: pick nuisance from what you have
print(suggest_nuisance(has_target_labels=True, has_target_domain=True))

matcher = PMHMatcher(nuisance="domain_shift", rank=16)
matcher.fit(x_source, x_target)   # or fit(x_src, y_src, x_tgt, y_tgt) for D1

# Credible comparison table (B0 / matched / wrong-W / isotropic)
compare_arms_sklearn(x_source, y_source, x_target, y_target, report_dir="results/run1")
```

---

## Step 5 — Credible claims (do not skip)

Matched PMH must beat **wrong-W** and **isotropic** on a **deployment-like** metric—not only beat B0.

```python
from pmh import compare_arms

compare_arms(
    trainer.artifact_,
    model_factory=...,
    setup_model=...,
    train_loader=...,
    val_loader=...,              # prefer target-domain val
    report_dir="results/arms",
)
```

→ [Walkthrough 8 — Controls](walkthroughs/08-falsification-controls.md)

---

## Step 6 — Tune (only after it runs)

| Knob | Start | If… |
|------|-------|-----|
| `PMHConfig.balanced()` | default | — |
| `rank` | 16–32 | preflight `marginal` → try more data or D1 |
| `weight` / `cap_ratio` | presets | PMH dominates loss → `conservative()` |
| `nuisance` | `suggest_nuisance(...)` | unsure D1 vs D4 |

```python
from pmh.tune import tune_sklearn_matcher  # rank grid on frozen features
```

---

## More docs

| I need… | Read |
|---------|------|
| Reading order / hub | [index.md](index.md) |
| D1–D7 + wizard | [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) |
| Copy-paste APIs | [GOLDEN_PATHS.md](GOLDEN_PATHS.md) |
| Hooks (ResNet / ViT / HF) | [hooks.md](hooks.md) |
| Errors | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Paper blocks | [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) · [walkthroughs](walkthroughs/index.md) |

---

## What success looks like

- [ ] One-sentence nuisance story documented  
- [ ] Training runs with `task` and `pmh` losses both non-zero  
- [ ] `artifact.preflight` is `pass` or `marginal` (if `fail`, see troubleshooting)  
- [ ] **matched** beats **wrong_w** and **isotropic** on your deployment metric  
- [ ] `artifact` path saved in experiment config  

You do **not** need paper datasets, task IDs, or our model classes.
