# Adapt PMH to your pipeline

**matching-pmh** is not a paper reproduction kit. It is a **drop-in layer** for *your* training stack: your dataset, your model, your loss, your evaluator.

You bring three decisions; the library handles estimation and the matched penalty.

---

## The three decisions (only yours to make)

| # | Question | Library helps |
|---|----------|----------------|
| 1 | What changes at deployment **without changing the label**? | [Symptom → D1–D7](nuisance_types.md) |
| 2 | Which tensor is `h = phi(x)`? (layer, shape `[B, d]`) | [ARCHITECTURES.md](ARCHITECTURES.md) |
| 3 | How do you train today? (plain PyTorch, HF, Lightning, sklearn features) | [Walkthroughs](walkthroughs/index.md) |

Everything else is mechanical.

---

## Universal recipe (any architecture)

```
YOUR dataloader(s)  →  YOUR encoder  →  h  →  YOUR task loss
                              ↑              ↑
                         Phase A once    Phase B every step
                         estimate Σ       + PMHLoss(h, Σ̂)
```

### Phase A — estimate once (or when deployment shifts)

**Frozen NumPy features (sklearn path):**

```python
from pmh import PMHMatcher

artifact = PMHMatcher(nuisance="domain_shift", rank=32).fit(h_source, h_target).artifact_
artifact.save("checkpoints/sigma_task")
```

**PyTorch hook `h`:**

```python
from pmh import SigmaTaskConfig, collect_features, estimate_from_config

encoder.eval()
h_source = collect_features(encoder, your_source_batches, max_batches=100)
h_target = collect_features(encoder, your_target_batches, max_batches=100)

artifact = estimate_from_config(
    SigmaTaskConfig.for_domain(rank=32),  # pick D1–D7 for your story
    h_source,
    h_target,
)
artifact.save("checkpoints/sigma_task")  # version with your data + hook layer
```

Use **your** batches: unlabeled target domain is fine for D4; style JSONL for D7; augmentation deltas for D3; coordinate indices for D5.

### Phase B — one line in your existing training step

```python
from pmh import PMHConfig, PMHLoss

pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2))

# inside your loop (unchanged optimizer, scheduler, AMP, etc.):
h = your_encoder(x)                    # same h as Phase A
task_loss = your_loss(head(h), y)      # CE, MSE, DPO, ...
total, pmh_term = pmh.capped_total(task_loss, h)
total.backward()
```

No need to adopt our model class, config system, or dataloaders.

---

## Pick estimator by **symptom** (not by paper task)

| Your deployment story | Use | Config |
|-----------------------|-----|--------|
| New site / camera / corpus; labels still mean the same thing | **D4** | `for_domain(rank=…)` |
| Paired domains + class labels | **D1** | `for_subspace(rank=…)` |
| Unstructured sensor noise level known | **D2** | `for_isotropic(dim, noise_level)` |
| You know the aug pipeline (color, blur, …) | **D3** | `for_augmentation()` + `aug_deltas` |
| Nuisance on known coordinates (atoms, tokens, channels) | **D5** | `for_compositional(indices)` |
| Drift along time in a window | **D6** | `for_temporal()` |
| LLM: same content, different style/format | **D7** | `for_alignment(rank=…)` + style JSONL |

Hybrid: estimate two artifacts, two `PMHLoss` terms.

---

## Architecture cheat sheet

| You train | Hook `h` | Walkthrough |
|-----------|----------|-------------|
| Custom `nn.Module` | `model.encode(x)` or backbone output | [1 — PyTorch](walkthroughs/01-pytorch-domain-d4.md) |
| torchvision / timm | Penultimate / CLS / pooled tokens | [2 ResNet](walkthroughs/02-resnet-vision-d4.md), [12 ViT](walkthroughs/12-vit-cls-d4.md) |
| Hugging Face `Trainer` | `representation_fn` → hidden states | [7 — HF Trainer](walkthroughs/07-hf-trainer-d7-dpo.md) |
| Lightning | `add_pmh_to_loss` on backbone | [10 — Lightning](walkthroughs/10-lightning.md) |
| Frozen features + sklearn | `.npy` features, D1/D4 | [3 — Office-31 style](walkthroughs/03-office31-sklearn-d1.md) |
| GNN | Graph readout vector | [14 — molecules](walkthroughs/14-qm9-molecule-d5.md) |
| Speech encoder | Pooled encoder embedding | [13 — speech](walkthroughs/13-speech-whisper-d4.md) |
| Token / code embeddings | Pooled or CLS before head | [15 — tokens](walkthroughs/15-codebert-tokens-d5.md) |

Copy the closest walkthrough, swap your dataset and model, keep the two phases.

---

## Compare to baseline (optional but recommended)

To show PMH is doing something **principled**, train the same pipeline four ways:

| Arm | Meaning |
|-----|---------|
| `b0` | Your pipeline, no PMH |
| `matched` | `PMHLoss` with your estimated Σ |
| `wrong_w` | Random subspace control |
| `isotropic` | Uniform penalty control |

**PyTorch (your model factory):**

```python
from pmh.benchmark import run_benchmark_protocol, write_benchmark_report

result = run_benchmark_protocol(
    artifact,
    model_factory=your_model_fn,
    setup_model=your_setup_fn,      # returns encoder, head, optimizer
    train_loader=your_train_loader,
    val_loader=your_val_loader,     # use *deployment-like* val when possible
    epochs=your_epochs,
    pmh_config=your_pmh_cfg,
)
write_benchmark_report(result, "results/pmh_compare")
# → results/pmh_compare/benchmark.md table
```

Or run `python examples/20_compare_training_arms.py` as a template.

This is **your** A/B test harness—not a fixed benchmark suite.

---

## What we deliberately do **not** require

- Paper datasets (ImageNet, QM9, Office-31, …)
- Paper architectures or checkpoints
- Replacing your trainer with ours
- Running thirteen predefined tasks

The research paper **validates** the principle; the library **implements** it for arbitrary pipelines.

---

## Checklist before you ship

- [ ] One-sentence nuisance story written down
- [ ] Same `h` in Phase A and Phase B
- [ ] `artifact.preflight` checked (re-estimate if deployment changes)
- [ ] Compared `b0` vs `matched` vs `wrong_w` vs `isotropic` on a **deployment-relevant** metric
- [ ] Saved `artifact` path in experiment config for reproducibility

---

## Next reads

- [QUICKSTART.md](QUICKSTART.md) — first run in 10 minutes  
- [THEORY.md](THEORY.md) — why matched vs isotropic matters  
- [PHILOSOPHY.md](PHILOSOPHY.md) — API design choices
