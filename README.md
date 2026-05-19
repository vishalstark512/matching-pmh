# matching-pmh

**Deployment geometry in. Matched robustness out.**

Estimate **Sigma_task** (D1–D7) · train any encoder with matched PMH · falsify with controls

[![PyPI](https://img.shields.io/pypi/v/matching-pmh.svg)](https://pypi.org/project/matching-pmh/)
[![Python](https://img.shields.io/pypi/pyversions/matching-pmh.svg)](https://pypi.org/project/matching-pmh/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/vishalstark512/matching-pmh/blob/main/LICENSE)
[![CI](https://github.com/vishalstark512/matching-pmh/actions/workflows/ci.yml/badge.svg)](https://github.com/vishalstark512/matching-pmh/actions/workflows/ci.yml)

[PyPI](https://pypi.org/project/matching-pmh/) ·
[GitHub](https://github.com/vishalstark512/matching-pmh) ·
[Walkthroughs](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/index.md) ·
[Theory](https://github.com/vishalstark512/matching-pmh/blob/main/docs/THEORY.md) ·
[Quickstart](https://github.com/vishalstark512/matching-pmh/blob/main/docs/QUICKSTART.md)

---

**matching-pmh** implements the **Matching Principle**: make your encoder robust along the directions that actually shift between train and deploy—**not** every input direction that happens to correlate with labels.

| Step | What you do | What the library does |
|------|-------------|------------------------|
| **1. Nuisance** | Name what can change at deployment **without changing the label** (site, lighting, sensor, format, …). | Registry + `suggest_nuisance` / `nuisance="auto"` to pick an estimator family. |
| **2. Geometry** | Collect source/target (or augmentation) batches that expose that variation. | Estimate **Σ_task** — covariance of label-preserving deployment nuisance — via **D1–D7** (shift, augment, sequence, style Gram, …). |
| **3. Training** | Keep your task loss; point a hook at representations `h = φ_θ(x)`. | Add **matched PMH**: shrink Jacobian sensitivity **along Σ_task** (matched penalty), not isotropic VAT/CORAL-on-weights alone. |

**Your stack, your hook.** ResNet / ViT (`timm`, `torchvision`), GNN mean-pool, Whisper-style encoders, causal LMs with LoRA, or frozen features + `PMHMatcher` / sklearn. Two phases (estimate once → train every step), one tensor `h`, no framework lock-in.

**Theory:** definitions, lemmas, and loss forms in [docs/THEORY.md](https://github.com/vishalstark512/matching-pmh/blob/main/docs/THEORY.md) (LaTeX-friendly).

> **Not a paper reproduction kit** — adapt your own pipeline.  
> **Start here:** [Getting started](https://github.com/vishalstark512/matching-pmh/blob/main/docs/GETTING_STARTED.md) → [Choose your setup](https://github.com/vishalstark512/matching-pmh/blob/main/docs/CHOOSE_YOUR_SETUP.md) → [Gallery](https://github.com/vishalstark512/matching-pmh/blob/main/docs/gallery/README.md)

---

## 30-second start

```bash
pip install matching-pmh
python examples/01_domain_shift_d4.py
pmh-train list-methods
```

```python
from pmh import PMHMatcher, PMHTrainer, PMHConfig

# NumPy / sklearn frozen features
matcher = PMHMatcher(nuisance="domain_shift", rank=32).fit(x_source, x_target)

# PyTorch — estimate + train in one call
trainer = PMHTrainer(model, hook="backbone", nuisance="auto", pmh_config=PMHConfig.balanced())
trainer.fit(train_loader, source_batches=src_loader, target_batches=tgt_loader, epochs=20)
```

[Getting started](https://github.com/vishalstark512/matching-pmh/blob/main/docs/GETTING_STARTED.md) · [Choose setup](https://github.com/vishalstark512/matching-pmh/blob/main/docs/CHOOSE_YOUR_SETUP.md) · [Benchmarks & TDI](https://github.com/vishalstark512/matching-pmh/blob/main/docs/BENCHMARKS.md) · [Troubleshooting](https://github.com/vishalstark512/matching-pmh/blob/main/docs/TROUBLESHOOTING.md) · [18 walkthroughs](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/index.md)

---

## Problem, object, repair, unification

| | |
|---|---|
| **Problem** | ERM uses every input direction that predicts labels—including nuisances harmful at deployment (lighting, site, sensor noise, formatting, renameable identifiers, …). |
| **Object** | **Sigma_task** = covariance of label-preserving deployment nuisance `n` (under law Q_n). |
| **Repair** | **Matched PMH** shrinks encoder sensitivity **along Sigma_task**, not uniformly (isotropic PMH / generic VAT). |
| **Unification** | CORAL, domain Grams, augmentation stacks, metric-learning directions, adversarial subspaces, and style Grams are different **estimators** of the same Sigma_task (Lemma D1–D7). |

Matched loss (schematic): `L = L_task + lambda * Tr(J_phi^T J_phi Sigma')` with `range(Sigma')` covering `range(Sigma_task)`. Details: [THEORY.md](https://github.com/vishalstark512/matching-pmh/blob/main/docs/THEORY.md).

---

## How it fits your codebase

```
 Phase A (once)              Phase B (every step)
 ----------------              --------------------
 source/target data    ->      x, y from your loader
       |                            |
 encoder (eval)        ->      encoder (train) -> h
       |                            |
 estimate D1-D7        ->      L_task(h, y) + PMHLoss(h, Sigma_hat)
       |
 artifact.pt
```

| You keep | Library adds |
|----------|----------------|
| Model, optimizer, task loss | `SigmaTaskConfig`, `estimate_from_config` |
| Data loaders | `collect_features` (optional) |
| Training loop / Trainer | `PMHLoss.capped_total` or `PMHTrainer` |

---

## Walkthroughs (18 templates)

| # | Guide | Run |
|---|--------|-----|
| 1 | [PyTorch + D4](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/01-pytorch-domain-d4.md) | `examples/01_domain_shift_d4.py` |
| 2 | [ResNet + D4](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/02-resnet-vision-d4.md) | `examples/12_resnet_hook_d4.py` |
| 3 | [Office-31 + sklearn](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/03-office31-sklearn-d1.md) | `examples/06_office31_sklearn.py` |
| 4 | [Multi-layer CNN](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/04-multilayer-convnet.md) | `examples/07_vision_multilayer.py` |
| 5 | [Compositional D5](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/05-compositional-d5.md) | `examples/13_compositional_train_d5.py` |
| 6 | [LLM style D7](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/06-llm-style-d7.md) | `examples/08_hf_style_d7.py` |
| 7 | [HF Trainer + DPO](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/07-hf-trainer-d7-dpo.md) | `examples/11_dpo_lora_style_pmh.py` |
| 8 | [Falsification controls](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/08-falsification-controls.md) | `examples/04_falsification_controls.py` |
| 9 | [CLI JSON jobs](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/09-cli-json-jobs.md) | `pmh-train estimate --config ...` |
| 10 | [Lightning](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/10-lightning.md) | `examples/09_lightning_module.py` |
| 11 | [Temporal D6](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/11-temporal-d6.md) | API in guide |
| 12 | [ViT / CLS + D4](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/12-vit-cls-d4.md) | `examples/14_vit_cls_d4.py` |
| 13 | [Speech encoder + D4](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/13-speech-whisper-d4.md) | `examples/15_speech_encoder_d4.py` |
| 14 | [QM9 / molecules D5](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/14-qm9-molecule-d5.md) | `examples/16_qm9_molecule_d5.py` |
| 15 | [Code / tokens D5](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/15-codebert-tokens-d5.md) | `examples/17_code_tokens_d5.py` |
| 16 | [Augmentations D3](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/16-augmentation-d3.md) | `examples/18_augmentation_d3.py` |
| 17 | [Compare arms on your pipeline](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/17-compare-arms-your-pipeline.md) | `examples/20_compare_training_arms.py` |
| 18 | [PMHTrainer quickstart](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/18-pmh-trainer-quickstart.md) | `examples/01_domain_shift_d4.py` |

---

## Estimators at a glance (D1–D7)

| Deployment story | Method | `SigmaTaskConfig` |
|------------------|--------|-------------------|
| Different site / camera / corpus; **P(y given x) stable** | **D4** | `SigmaTaskConfig.for_domain(rank=32)` |
| Low-rank shift; labels on both domains | **D1** | `SigmaTaskConfig.for_subspace(rank=32)` |
| Unstructured sensor / acquisition noise | **D2** | `SigmaTaskConfig.for_isotropic(dim, noise_level)` |
| Known augmentation modes (color, blur, crop, …) | **D3** | `SigmaTaskConfig.for_augmentation()` + `aug_deltas` |
| Nuisance on specific coordinates (atoms, tokens) | **D5** | `SigmaTaskConfig.for_compositional(indices)` |
| Drift along time within a sequence | **D6** | `SigmaTaskConfig.for_temporal()` |
| LLM style / format; semantics fixed | **D7** | `SigmaTaskConfig.for_alignment(rank=32)` |

```bash
pmh-train list-methods
```

Hybrid nuisances: estimate separate Sigma matrices and add separate `PMHLoss` terms.

---

## Install

```bash
pip install matching-pmh
```

| Extra | Use case |
|-------|----------|
| `pip install "matching-pmh[vision]"` | ResNet / ViT examples |
| `pip install "matching-pmh[hf]"` | D7 style Gram (Transformers) |
| `pip install "matching-pmh[hf-lora]"` | LoRA + DPO example |
| `pip install "matching-pmh[sklearn,vision]"` | Office-31 pipeline |
| `pip install "matching-pmh[lightning]"` | Lightning callback |
| `pip install "matching-pmh[all]"` | Development + docs |

**From source:**

```bash
git clone https://github.com/vishalstark512/matching-pmh.git
cd matching-pmh && pip install -e ".[dev]" && pytest -q
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[GETTING_STARTED.md](https://github.com/vishalstark512/matching-pmh/blob/main/docs/GETTING_STARTED.md)** | **Main adoption guide (start here)** |
| [CHOOSE_YOUR_SETUP.md](https://github.com/vishalstark512/matching-pmh/blob/main/docs/CHOOSE_YOUR_SETUP.md) | Pick API by stack and data |
| [TROUBLESHOOTING.md](https://github.com/vishalstark512/matching-pmh/blob/main/docs/TROUBLESHOOTING.md) | Errors, preflight, hook dim |
| [gallery/](https://github.com/vishalstark512/matching-pmh/blob/main/docs/gallery/README.md) | Copy-paste: vision / tabular / NLP |
| [hooks.md](https://github.com/vishalstark512/matching-pmh/blob/main/docs/hooks.md) | ResNet, timm, HF hooks |
| [ADAPT_YOUR_PIPELINE.md](https://github.com/vishalstark512/matching-pmh/blob/main/docs/ADAPT_YOUR_PIPELINE.md) | Integration checklist |
| [walkthroughs/](https://github.com/vishalstark512/matching-pmh/blob/main/docs/walkthroughs/index.md) | 18 stack-specific tutorials |
| [THEORY.md](https://github.com/vishalstark512/matching-pmh/blob/main/docs/THEORY.md) | Mathematics |

---

## Citation

Cite the Grand Unification / Matching Principle manuscript. See `CITATION.cff` in the repository.

```bibtex
@software{matching_pmh,
  title  = {matching-pmh: Matched PMH training from estimated deployment nuisance geometry},
  author = {Rajput, Vishal},
  year   = {2026},
  url    = {https://github.com/vishalstark512/matching-pmh}
}
```

---

## Contributing

See [CONTRIBUTING.md](https://github.com/vishalstark512/matching-pmh/blob/main/CONTRIBUTING.md).

## License

MIT — see [LICENSE](https://github.com/vishalstark512/matching-pmh/blob/main/LICENSE).
