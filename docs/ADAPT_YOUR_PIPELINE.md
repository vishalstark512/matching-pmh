# Adapt PMH to your pipeline

**Start here if you are new:** [Getting started (adoption guide)](GETTING_STARTED.md)  
**Pick stack / data:** [Choose your setup](CHOOSE_YOUR_SETUP.md)  
**Something broke:** [Troubleshooting](TROUBLESHOOTING.md)

**matching-pmh** is a **drop-in layer** for *your* training stack—not a paper reproduction kit.

---

## The three decisions (only yours)

| # | Question | Doc |
|---|----------|-----|
| 1 | What changes at deployment **without changing the label**? | [nuisance_types.md](nuisance_types.md) · `suggest_nuisance()` |
| 2 | Which tensor is `h = φ(x)`? (`[B, d]`) | [hooks.md](hooks.md) · [ARCHITECTURES.md](ARCHITECTURES.md) |
| 3 | PyTorch, sklearn, or HF? | [CHOOSE_YOUR_SETUP.md](CHOOSE_YOUR_SETUP.md) |

---

## Universal recipe

```
YOUR data  →  YOUR encoder  →  h  →  YOUR task loss
                    ↑              ↑
               Phase A once    Phase B: + PMH on h
               estimate Σ̂
```

### Fastest path — PyTorch

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    model, hook=backbone, head=head,
    nuisance="domain_shift",              # or "auto"
    pmh_config=PMHConfig.balanced(),
    artifact_path="artifacts/sigma.pt",
)
trainer.fit(train_loader, source_batches=src, target_batches=tgt, epochs=20)
```

### Fastest path — sklearn features

```python
from pmh import PMHMatcher, compare_arms_sklearn

matcher = PMHMatcher(nuisance="domain_shift", rank=16).fit(x_source, x_target)
compare_arms_sklearn(x_source, y_source, x_target, y_target, report_dir="results/")
```

---

## Estimator coverage (D1–D7)

| Method | Story (one line) | `PMHMatcher` | `PMHTrainer` |
|--------|------------------|--------------|--------------|
| **D4** | New site/camera, same labels | `fit(xs, xt)` | `source_batches` + `target_batches` |
| **D1** | Two domains + class labels | `fit(xs, ys, xt, yt)` | labeled loaders |
| **D2** | Known noise level | `dim=` + `fit(xs)` | `source_batches` |
| **D3** | Known aug pipeline | `aug_deltas=` | `augmentations=` |
| **D5** | Nuisance coordinates | `nuisance_indices=` | same + batches |
| **D6** | Drift over time | `fit([N,T,d])` | `sequences_batches` |
| **D7** | LLM style pairs | HF API | `HFPMHTrainer` + JSONL |
| **Hybrid** | Two stories | — | [HYBRID_NUISANCE.md](HYBRID_NUISANCE.md) |

```python
from pmh import suggest_nuisance
print(suggest_nuisance(has_target_domain=True, has_target_labels=False))
```

---

## Architecture → template

| Stack | Hook | Template |
|-------|------|----------|
| Custom PyTorch | `hook=backbone` | [WT 1](walkthroughs/01-pytorch-domain-d4.md) · [gallery/vision](gallery/vision.md) |
| ResNet / timm | [hooks.md](hooks.md) | [WT 2](walkthroughs/02-resnet-vision-d4.md) |
| sklearn features | N/A (precomputed `h`) | [gallery/tabular](gallery/tabular.md) |
| HF LM | `HFPMHTrainer` | [gallery/nlp](gallery/nlp.md) |
| Lightning | callback | [WT 10](walkthroughs/10-lightning.md) |
| GNN | pool + D5 indices | [WT 14](walkthroughs/14-qm9-molecule-d5.md) |

---

## Credible comparison (recommended)

```python
from pmh import compare_arms

compare_arms(
    trainer.artifact_,
    model_factory=...,
    setup_model=...,       # encoder, head, optimizer
    train_loader=...,
    val_loader=...,        # deployment-like when possible
    report_dir="results/arms",
)
```

| Arm | Meaning |
|-----|---------|
| `b0` | No PMH |
| `matched` | Your Σ̂ |
| `wrong_w` | Random subspace control |
| `isotropic` | Uniform control |

Template script: `examples/20_compare_training_arms.py`

---

## Checklist before you ship

- [ ] One-sentence nuisance story written
- [ ] Same `h` in Phase A and B
- [ ] `artifact.preflight` checked
- [ ] **matched** vs **wrong_w** vs **isotropic** on deployment metric
- [ ] `artifact` path in experiment config

---

## Doc index

| Doc | Use when |
|-----|----------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | First integration |
| [CHOOSE_YOUR_SETUP.md](CHOOSE_YOUR_SETUP.md) | Which API / nuisance |
| [hooks.md](hooks.md) | Where to attach `h` |
| [gallery/](gallery/README.md) | Copy-paste by domain |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Errors |
| [THEORY.md](THEORY.md) | Mathematics |
| [walkthroughs/](walkthroughs/index.md) | Stack-specific tutorials |
