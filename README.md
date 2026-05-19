# matching-pmh

<p align="center">
  <strong>Deployment geometry in. Matched robustness out.</strong><br>
  <sub>Estimate <code>Σ_task</code> (D1–D7) · train any encoder with matched PMH · falsify with controls</sub>
</p>

<p align="center">
  <a href="https://pypi.org/project/matching-pmh/"><img src="https://img.shields.io/pypi/v/matching-pmh.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/matching-pmh/"><img src="https://img.shields.io/pypi/pyversions/matching-pmh.svg" alt="Python"></a>
  <a href="https://github.com/vishalstark512/matching-pmh/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/vishalstark512/matching-pmh/actions/workflows/ci.yml"><img src="https://github.com/vishalstark512/matching-pmh/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/matching-pmh/">PyPI</a> ·
  <a href="https://github.com/vishalstark512/matching-pmh">GitHub</a> ·
  <a href="docs/walkthroughs/index.md">Walkthroughs</a> ·
  <a href="docs/THEORY.md">Theory</a> ·
  <a href="docs/ARCHITECTURES.md">Integration</a> ·
  <a href="docs/QUICKSTART.md">Quickstart</a>
</p>

---

**matching-pmh** is a research-grade PyTorch library for the [*Matching Principle*](docs/THEORY.md): name what changes at deployment **without changing the label**, estimate that nuisance geometry $\Sigma_{\mathrm{task}}$, and add a **matched** Jacobian penalty on **your** representations $h=\phi_\theta(x)$—ResNet, ViT, GNN, Whisper-style encoders, causal LMs with LoRA, or frozen features + sklearn.

> **Design goal:** two phases, one hook tensor `h`, no framework lock-in. The paper’s thirteen task blocks are **validation examples**; this repo is built so a new lab member can integrate in an afternoon.

---

## 30-second start

```bash
pip install matching-pmh
python examples/01_domain_shift_d4.py          # minimal PyTorch loop
pmh-train list-methods                         # D1–D7 catalog
```

```python
import pmh
from pmh import SigmaTaskConfig, PMHConfig, PMHLoss, collect_features, estimate_from_config

# Phase A — estimate (frozen encoder)
artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=32), h_source, h_target)
artifact.save("artifacts/sigma")

# Phase B — train (your loop)
pmh_loss = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2))
total, _ = pmh_loss.capped_total(task_loss, h)
```

→ Full path: **[docs/QUICKSTART.md](docs/QUICKSTART.md)** · Pick your stack: **[walkthroughs](docs/walkthroughs/index.md)**

---

## Problem → object → repair → unification

**Problem.** ERM uses every input direction that predicts training labels—including nuisances harmful at deployment (lighting, site, sensor noise, answer formatting, renameable identifiers, …).

**Object.**

$$
\Sigma_{\mathrm{task}} = \mathrm{Cov}_{Q_n}(n)
$$

for label-preserving deployment nuisance $n \sim Q_n$.

**Repair.** Matched PMH shrinks the encoder Jacobian **along** $\Sigma_{\mathrm{task}}$, not uniformly (isotropic PMH / generic VAT):

$$
\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \lambda \,\mathbb{E}_x\left[\mathrm{Tr}\left(J_\phi(x)^\top J_\phi(x)\,\Sigma'\right)\right],
\quad
\mathrm{range}(\Sigma') \supseteq \mathrm{range}(\Sigma_{\mathrm{task}}).
$$

**Unification.** CORAL, domain Grams, augmentation stacks, metric-learning directions, adversarial subspaces, and style Grams are **estimators** of the same object (D1–D7); matched PMH is one **loss** with $\Sigma' \approx \hat\Sigma_{\mathrm{task}}$.

---

## How it fits your codebase

```
 Phase A (once)              Phase B (every step)
 ───────────────              ────────────────────
 source/target data    →      x, y ~ your loader
       ↓                            ↓
 encoder (eval)        →      encoder (train) → h
       ↓                            ↓
 estimate D1–D7        →      L_task(h, y) + PMHLoss(h, Σ̂)
       ↓
 artifact.pt
```

| You keep | Library adds |
|----------|----------------|
| Model, optimizer, task loss | `SigmaTaskConfig`, `estimate_from_config` |
| Data loaders | `collect_features` (optional) |
| Training loop / Trainer | `PMHLoss.capped_total` or `PMHTrainer` |

---

## Walkthroughs (16 guides)

| # | Guide | Paper block | Run |
|---|--------|-------------|-----|
| 1 | [PyTorch + D4](docs/walkthroughs/01-pytorch-domain-d4.md) | Generic | `examples/01_domain_shift_d4.py` |
| 2 | [ResNet + D4](docs/walkthroughs/02-resnet-vision-d4.md) | Vision | `examples/12_resnet_hook_d4.py` |
| 3 | [Office-31 + sklearn](docs/walkthroughs/03-office31-sklearn-d1.md) | T1 | `examples/06_office31_sklearn.py` |
| 4 | [Multi-layer CNN](docs/walkthroughs/04-multilayer-convnet.md) | T2 | `examples/07_vision_multilayer.py` |
| 5 | [Compositional D5](docs/walkthroughs/05-compositional-d5.md) | T5 | `examples/13_compositional_train_d5.py` |
| 6 | [LLM style D7](docs/walkthroughs/06-llm-style-d7.md) | T7A | `examples/08_hf_style_d7.py` |
| 7 | [HF Trainer + DPO](docs/walkthroughs/07-hf-trainer-d7-dpo.md) | T7A | `examples/11_dpo_lora_style_pmh.py` |
| 8 | [Falsification controls](docs/walkthroughs/08-falsification-controls.md) | All | `examples/04_falsification_controls.py` |
| 9 | [CLI JSON jobs](docs/walkthroughs/09-cli-json-jobs.md) | Repro | `pmh-train estimate --config …` |
| 10 | [Lightning](docs/walkthroughs/10-lightning.md) | — | `examples/09_lightning_module.py` |
| 11 | [Temporal D6](docs/walkthroughs/11-temporal-d6.md) | T6B | API in guide |
| 12 | [ViT / CLS + D4](docs/walkthroughs/12-vit-cls-d4.md) | T2 ViT | `examples/14_vit_cls_d4.py` |
| 13 | [Speech encoder + D4](docs/walkthroughs/13-speech-whisper-d4.md) | T6A | `examples/15_speech_encoder_d4.py` |
| 14 | [QM9 / molecules D5](docs/walkthroughs/14-qm9-molecule-d5.md) | T5A | `examples/16_qm9_molecule_d5.py` |
| 15 | [Code / tokens D5](docs/walkthroughs/15-codebert-tokens-d5.md) | T5B | `examples/17_code_tokens_d5.py` |
| 16 | [Augmentations D3](docs/walkthroughs/16-augmentation-d3.md) | T2 aug | `examples/18_augmentation_d3.py` |

**Index:** [docs/walkthroughs/index.md](docs/walkthroughs/index.md) · **Example catalog:** [examples/README.md](examples/README.md)

---

## Estimators at a glance (D1–D7)

| Story | Method | `SigmaTaskConfig` |
|-------|--------|-------------------|
| Domain / site; $P(y\mid x)$ stable | **D4** | `for_domain(rank=…)` |
| Low-rank shift + labels | **D1** | `for_subspace(rank=…)` |
| Unstructured noise | **D2** | `for_isotropic(dim, noise_level)` |
| Known aug modes | **D3** | `for_augmentation()` + `aug_deltas` |
| Nuisance coordinates (atoms, tokens) | **D5** | `for_compositional(indices)` |
| Temporal drift in window | **D6** | `for_temporal()` |
| LLM style vs fixed content | **D7** | `for_alignment(rank=…)` |

```bash
pmh-train list-methods
```

---

## Install

```bash
pip install matching-pmh
```

| Extra | Use case |
|-------|----------|
| `[vision]` | ResNet / ViT walkthroughs |
| `[hf]` | D7 style Gram (Transformers) |
| `[hf-lora]` | LoRA + DPO example |
| `[sklearn,vision]` | Office-31 pipeline |
| `[lightning]` | `LightningModule` callback |
| `[all]` | Development + docs |

**From source (contributors):**

```bash
git clone https://github.com/vishalstark512/matching-pmh.git
cd matching-pmh
pip install -e ".[dev,all]"
pytest -q
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](docs/QUICKSTART.md) | First successful run in 10 minutes |
| [THEORY.md](docs/THEORY.md) | $\Sigma_{\mathrm{task}}$, recipe, falsification |
| [ARCHITECTURES.md](docs/ARCHITECTURES.md) | Hook points per stack |
| [PHILOSOPHY.md](docs/PHILOSOPHY.md) | Design principles for integrators |
| [walkthroughs/](docs/walkthroughs/index.md) | End-to-end guides |
| [nuisance_types.md](docs/nuisance_types.md) | Data formats |
| [cli.md](docs/cli.md) | `pmh-train` reference |

---

## Citation

If you use this software, cite the Grand Unification / Matching Principle manuscript ([`CITATION.cff`](CITATION.cff)).

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

We welcome issues, walkthrough improvements, and estimator integrations. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
