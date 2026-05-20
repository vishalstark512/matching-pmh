# Adaptation workbook (optional)

**Most users:** follow [index](index.md) → [Golden paths](GOLDEN_PATHS.md) → [Integrate your project](GETTING_STARTED.md).

Use this workbook when filling in a [walkthrough](walkthroughs/index.md) with `YOUR_*` placeholders — not required for the default path.

---

## Part A — Before you touch code

### A1. One-sentence nuisance (required)

Finish this sentence for **your** deployment:

> “At deployment, \_\_\_\_\_\_\_\_ can change, but the label still means the same thing because \_\_\_\_\_\_\_\_.”

| Good examples | Bad examples |
|---------------|--------------|
| “Hospital B’s CT scanner contrast shifts; diagnosis unchanged.” | “I want better accuracy.” |
| “Customer emails use bullet lists instead of paragraphs; intent unchanged.” | “Add robustness to noise.” (too vague — use D2/D3 only if you name the noise) |

Pick subtype: [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) or:

```python
from pmh import suggest_nuisance
print(suggest_nuisance(
    has_source_labels=True,
    has_target_labels=False,      # your flags
    has_target_domain=True,
))
```

### A2. What you must have

| Item | Your value (fill in) |
|------|----------------------|
| Training framework | PyTorch / sklearn / HF Trainer / Lightning |
| Representation `h` | layer name, shape `[B, d]` = |
| Source data | (site A / train corpus / …) |
| Target-like data | (site B / deploy corpus / …) |
| Task metric | accuracy / WER / RM score / … |
| Deployment metric | **same as above, on target domain** |

### A3. Pick your walkthrough

| If your story is… | Walkthrough | Example script |
|-------------------|-------------|----------------|
| Two visual domains, train full model | [1 — PyTorch D4](walkthroughs/01-pytorch-domain-d4.md) | `01_domain_shift_d4.py` |
| ResNet / torchvision | [2 — ResNet D4](walkthroughs/02-resnet-vision-d4.md) | `12_resnet_hook_d4.py` |
| Frozen embeddings + sklearn | [3 — Office-31 / D1](walkthroughs/03-office31-sklearn-d1.md) | `06_office31_sklearn.py` |
| LLM formatting / style | [6 — D7 style](walkthroughs/06-llm-style-d7.md) | `08_hf_style_d7.py` |
| Not sure | [18 — PMHTrainer](walkthroughs/18-pmh-trainer-quickstart.md) | `01_domain_shift_d4.py` |

---

## Part B — Two-phase recipe (all stacks)

```
Phase A (once per hook + data snapshot)
  YOUR source batches  ──┐
  YOUR target batches  ──┼──> estimate Sigma_task  -->  artifact (.pt)
                         │
Phase B (every training step)
  batch (x, y)  -->  h = phi(x)  -->  L_task(h,y) + PMH(h, Sigma_hat)
```

**Rules**

1. Same `hook` / layer in Phase A and B.  
2. Re-run Phase A if you change data, hook, or backbone architecture.  
3. Save `artifact_path` in your experiment config.

---

## Part C — PyTorch (`PMHTrainer`)

### C1. Minimal integration

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    YOUR_MODEL,                    # nn.Module
    hook=YOUR_BACKBONE,              # submodule or "avgpool"
    head=YOUR_CLASSIFIER,            # optional
    nuisance="domain_shift",         # or suggest_nuisance result
    pmh_config=PMHConfig.balanced(),
    artifact_path="artifacts/YOUR_RUN/sigma.pt",
)

trainer.fit(
    YOUR_TRAIN_LOADER,
    source_batches=YOUR_SOURCE_LOADER,   # for estimate
    target_batches=YOUR_TARGET_LOADER,
    epochs=YOUR_EPOCHS,
)
```

### C2. Map from `examples/01_domain_shift_d4.py`

| Example | You replace with |
|---------|------------------|
| `Backbone` MLP | Your encoder |
| `_loader(..., shift=0.8)` | Your target-domain `DataLoader` |
| `hook=backbone` | Same module you use in forward |
| `rank=6` | 16–32; check `trainer.artifact_.preflight` |
| `artifacts/demo_d4.pt` | Your run directory |

### C3. Verify

```bash
python examples/01_domain_shift_d4.py
# Expect: preflight=pass|marginal, task and pmh losses printed
```

On your code: both `task_loss` and `pmh_loss` non-zero; `preflight` not `fail`.

---

## Part D — sklearn (frozen features)

### D1. Minimal integration

```python
from pmh import PMHMatcher, compare_arms_sklearn

matcher = PMHMatcher(
    nuisance="subspace",           # or domain_shift
    rank=16,
    X_target=x_target,             # enables Pipeline.fit(X, y)
)
matcher.fit(x_source, y_source, x_target, y_target)  # D1

compare_arms_sklearn(
    x_source, y_source, x_target, y_target,
    report_dir="results/YOUR_RUN",
)
```

Read `results/YOUR_RUN/benchmark.md` — matched should beat wrong-W on **target accuracy** and ideally on **TDI_cls**.

### D2. Extract features once (vision)

```python
from pmh.datasets.office31 import extract_office31_features
# Or your own: h = your_encoder(x); save .npy
```

**Do not commit** `.npy` or datasets — see [DATA_POLICY.md](DATA_POLICY.md).

---

## Part E — Credible claims (all tasks)

Run **four arms** before claiming “PMH works”:

| Arm | Purpose |
|-----|---------|
| B0 | No PMH |
| matched | Your estimated Sigma_task |
| wrong_w | Random subspace control |
| isotropic | Generic shrinkage control |

→ [Walkthrough 8 — Controls](walkthroughs/08-falsification-controls.md)  
→ PyTorch: `compare_arms(...)`  
→ sklearn: `compare_arms_sklearn(...)`

---

## Part F — Troubleshooting quick links

| Symptom | Fix |
|---------|-----|
| `preflight=fail` | More target data, higher `rank`, try D1 if labels on both domains |
| PMH loss always 0 | `h.requires_grad`, warmup finished, weight > 0 |
| Only B0 improves | Wrong nuisance story or hook not aligned with shift |
| wrong-W beats matched | Not a matched claim — see controls |

Full list: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## Part G — Documentation map

| Doc | Role |
|-----|------|
| **This workbook** | Generic fill-in guide |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Fast path |
| [walkthroughs/index.md](walkthroughs/index.md) | 18 full stack-specific guides |
| [hooks.md](hooks.md) | ResNet, ViT, HF |
| [BENCHMARKS.md](BENCHMARKS.md) | Accuracy + TDI tables |
