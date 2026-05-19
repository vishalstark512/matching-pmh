# Choose your setup

Pick **one row** that matches how you train today. Each row links to a **copy-paste template** and the **API** to use.

---

## By training stack

| Stack | Estimate (Phase A) | Train (Phase B) | Template |
|-------|-------------------|-----------------|----------|
| **PyTorch** (custom `nn.Module`) | `PMHTrainer.estimate` or `estimate_from_config` | `PMHTrainer.fit` or `PMHLoss` | [Gallery: vision](gallery/vision.md) · [WT 1](walkthroughs/01-pytorch-domain-d4.md) |
| **torchvision / timm** | same + `hook="avgpool"` or `encoder_timm` | `PMHTrainer` | [hooks.md](hooks.md) · [WT 2](walkthroughs/02-resnet-vision-d4.md) |
| **sklearn** (frozen features) | `PMHMatcher.fit` | sklearn clf on `transform()` or separate torch train | [Gallery: tabular](gallery/tabular.md) · [WT 3](walkthroughs/03-office31-sklearn-d1.md) |
| **Hugging Face Trainer** | D7 JSONL or hidden states | `PMHTrainer` / HF callback | [Gallery: NLP](gallery/nlp.md) · [WT 7](walkthroughs/07-hf-trainer-d7-dpo.md) |
| **Lightning** | `estimate_from_config` + save artifact | `add_pmh_to_loss` | [WT 10](walkthroughs/10-lightning.md) |
| **GNN** | D5 + `nuisance_indices` | `PMHTrainer` + pool hook | [WT 14](walkthroughs/14-qm9-molecule-d5.md) |

---

## By data you have

| You have… | Set `nuisance=` | Phase A input |
|-----------|-----------------|---------------|
| Source domain + target domain (labels optional on target) | `domain_shift` / D4 | `source_batches`, `target_batches` |
| Source + target **with class labels** both sides | `subspace` / D1 | labeled loaders or `fit(xs, ys, xt, yt)` |
| List of augmentation functions | `augmentation` / D3 | `augmentations=[fn, ...]` |
| Known coordinate indices (atoms, tokens) | `compositional` / D5 | `nuisance_indices=[...]` |
| Sequences `[N, T, d]` | `temporal` / D6 | `sequences_batches` |
| Style JSONL (same content, different style) | `style` / D7 | `style_jsonl` + HF model |
| Only noise level, single domain | `isotropic` / D2 | `dim`, `noise_level` |

Unsure? Use auto mode:

```python
from pmh import suggest_nuisance, DataContext

ctx = DataContext(
    has_target_domain=True,
    has_target_labels=False,
)
print(ctx.suggest())   # → domain_shift, D4, reason=...
```

```python
PMHTrainer(model, hook=..., nuisance="auto", has_target_domain=True, ...)
```

---

## By experience level

| Level | Do this |
|-------|---------|
| **First time** | [GETTING_STARTED.md](GETTING_STARTED.md) → run `examples/01_domain_shift_d4.py` |
| **Integrating one model** | [hooks.md](hooks.md) + [ADAPT checklist](ADAPT_YOUR_PIPELINE.md#checklist-before-you-ship) |
| **Publishing results** | [Controls walkthrough](walkthroughs/08-falsification-controls.md) + `compare_arms` |
| **Multiple nuisances** | [HYBRID_NUISANCE.md](HYBRID_NUISANCE.md) |
| **Low-level control** | `estimate_from_config` + `PMHLoss` (escape hatch) |

---

## API cheat sheet (v1.2)

```python
import pmh

# High level (prefer these)
pmh.PMHTrainer(...)       # PyTorch: estimate + fit
pmh.PMHMatcher(...)       # NumPy/sklearn: fit + transform
pmh.HFPMHTrainer(...)    # Transformers + D7
pmh.compare_arms(...)     # B0 / matched / wrong-W / isotropic
pmh.suggest_nuisance(...) # pick D1–D7

# Config
pmh.PMHConfig.balanced()
pmh.PMHConfig.conservative()

# Low level (power users)
pmh.estimate_from_config(...)
pmh.PMHLoss(artifact, ...)
pmh.MultiPMHLoss([art1, art2], ...)
```

---

## Next step

→ [GETTING_STARTED.md](GETTING_STARTED.md) if you have not run an example yet  
→ [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if something failed
