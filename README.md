# matching-pmh

**Use the matching principle on your task, with your architecture.**

This library estimates deployment nuisance geometry \(\Sigma_{\mathrm{task}}\) and trains with a **matched PMH** penalty on representations \(h=\phi_\theta(x)\). It is the public implementation companion to the Grand Unification research line on *The Matching Principle*.

| | |
|---|---|
| **PyPI** | https://pypi.org/project/matching-pmh/ |
| **GitHub** | https://github.com/vishalstark512/matching-pmh |
| **Import** | `import pmh` |
| **CLI** | `pmh-train` |
| **Theory (long)** | [docs/THEORY.md](docs/THEORY.md) |

The paper’s thirteen task blocks (ViT, ResNet, QM9, Whisper, Qwen DPO, …) are **validation examples**, not a closed API. If you can describe what changes at deployment **without changing the label**, you can run the same five-step recipe on **any** model you already train.

---

## The idea in one box

**Problem.** Standard training (ERM) encourages the network to use every input direction that predicts training labels—including nuisances that are useless or harmful at deployment (lighting, domain style, sensor noise, answer formatting, renameable code tokens, …).

**Object.** \(\Sigma_{\mathrm{task}} = \mathrm{Cov}_{Q_n}(n)\): the covariance of **label-preserving** variation you expect when the system is deployed.

**Repair.** Add a penalty that shrinks the encoder Jacobian **along** \(\Sigma_{\mathrm{task}}\) (matched PMH), not uniformly in all directions (isotropic PMH / generic VAT).

**Unification.** CORAL, domain Gram matrices, augmentation consistency, metric learning, adversarial directions, and style-PMH for alignment are different **estimators** of the same \(\Sigma_{\mathrm{task}}\); matched PMH is the same **loss family** with \(\Sigma' \approx \Sigma_{\mathrm{task}}\).

---

## Architecture-agnostic by design

You keep **your** model, **your** dataloader, **your** task loss. The library only needs:

1. A representation \(h = \phi_\theta(x)\) you can differentiate (any layer you treat as the “encoder”).
2. A story for deployment nuisance → pick estimator **D1–D7**.
3. `PMHLoss` added to your training loop (PyTorch, Hugging Face `PMHTrainer`, or Lightning).

| You might use | Nuisance story often maps to |
|---------------|------------------------------|
| ResNet, ViT, ConvNeXt, U-Net | D3 photometric, D4 domain, D5 per-pixel/partition |
| GNN / message passing (QM9, molecules) | D5 compositional (per-atom nuisance) |
| Transformer (BERT, CodeBERT, Whisper) | D4 domain, D5 tokens, D7 style |
| Causal LM + LoRA / DPO | D7 style Gram from style-pair JSONL |
| Classical features + sklearn head | D1 subspace, D4 domain |
| Sequence models (HAR, finance windows) | D6 temporal |

No requirement to use the paper’s checkpoints, datasets, or training scripts.

---

## The five-step recipe (same everywhere)

```
 1. Name nuisance     What changes at deployment without changing y?
        ↓
 2. Pick A_k → Dk     D1 … D7 (table below)
        ↓
 3. Estimate Σ̂       pmh-train estimate / estimate_from_config → .pt
        ↓
 4. Preflight         eigengap pass | marginal | fail
        ↓
 5. Train + controls  L_task + PMHLoss(h); report matched / wrong-W / isotropic / signal-W
```

**Credible claim:** matched beats baseline on deployment-relevant metrics **and** wrong-W ≈ isotropic **and** signal-W hurts when you have a clear signal direction. Matched-only gains are inconclusive about the principle.

Full theory: [docs/THEORY.md](docs/THEORY.md).

---

## Which estimator (D1–D7)?

| Deployment story | Method | Estimate from | `SigmaTaskConfig` |
|------------------|--------|---------------|-------------------|
| Different site / camera / corpus; \(P(y\mid x)\) stable | **D4** | Unlabeled source vs target features | `for_domain(rank=…)` |
| Low-rank shift + labels on both domains | **D1** | Cross-domain SVD on features | `for_subspace(rank=…)` |
| Unstructured sensor / acquisition noise | **D2** | Noise level + dim | `for_isotropic(dim, noise_level)` |
| Known aug modes (color, blur, crop, …) | **D3** | Stack of aug feature deltas | `for_augmentation()` |
| Nuisance on specific coordinates (atoms, tokens) | **D5** | Features + `nuisance_indices` | `for_compositional(indices)` |
| Drift along time within a sequence | **D6** | Sequence of representations | `for_temporal()` |
| LLM style / format; semantics fixed | **D7** | `style_pairs.jsonl` on an LM | `for_alignment(rank=…)` |

```bash
pmh-train list-methods
```

**Hybrid nuisances** (e.g. domain + photometric): estimate two \(\Sigma\) and add two penalties (paper §5).

---

## Minimal code (any PyTorch model)

```python
import torch
from pmh import SigmaTaskConfig, PMHConfig, PMHLoss, estimate_from_config

# Your encoder: any architecture; h = phi(x) shape [B, d]
# h_source, h_target: [N, d] from a frozen or warm encoder (D4 example)
artifact = estimate_from_config(
    SigmaTaskConfig.for_domain(rank=32),
    h_source,
    h_target,
)
print(artifact.preflight, artifact.eigengap)  # aim for pass; marginal → weak ID
artifact.save("artifacts/sigma_task")

pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2))

for x, y in loader:
    h = your_encoder(x)
    task_loss = your_criterion(h, y)
    total, pmh_term = pmh.capped_total(task_loss, h)
    total.backward()
    optimizer.step()
```

**Cap ratio** (~0.3): PMH term capped to a fraction of task loss so \(\lambda\) does not need heavy tuning (paper Prop. 7).

**Controls:**

```python
PMHLoss(artifact, cfg, mode="wrong_w")      # Lemma C — should ≈ isotropic
PMHLoss(artifact, cfg, mode="isotropic")
# signal-W: see examples/04_falsification_controls.py
```

---

## Worked paths by stack

### Vision / domain adaptation → D4 or D1

`examples/01_domain_shift_d4.py`, `examples/06_office31_sklearn.py`, `examples/07_vision_multilayer.py`

### Compositional (molecules, tokens) → D5

`examples/03_compositional_d5.py`

### LLM alignment (style vs content) → D7

Style JSONL: `prompt`, `content_fixed`, `style_variants` (dict).  
Preference JSONL for DPO-style training.

```bash
pip install "matching-pmh[hf-lora]"
pmh-train estimate --config examples/configs/d7_style_estimate.json
python examples/11_dpo_lora_style_pmh.py --model-id Qwen/Qwen2.5-0.5B-Instruct --train --lora
```

Also: `examples/08_hf_style_d7.py`, `examples/10_hf_trainer.py`

### Reproducible jobs (no custom script)

```bash
pmh-train estimate --config examples/configs/d4_estimate.json
pmh-train preflight artifacts/d4.pt
```

See [docs/nuisance_types.md](docs/nuisance_types.md), [docs/cli.md](docs/cli.md).

---

## Install

```bash
pip install matching-pmh
```

| Extra | When |
|-------|------|
| `[hf]` | D7 style Gram from Hugging Face models |
| `[hf-lora]` | Example 11: LoRA + DPO + PMH |
| `[sklearn,vision]` | Office-31 / torchvision helpers |
| `[lightning]` | Lightning callback |

```bash
git clone https://github.com/vishalstark512/matching-pmh.git
cd matching-pmh && pip install -e ".[dev]" && pytest
```

---

## Example scripts (paper blocks → your templates)

| Script | D | Template for |
|--------|---|----------------|
| `01_domain_shift_d4.py` | D4 | Custom encoder + `PMHLoss` |
| `02_save_load_artifact.py` | * | Artifact I/O |
| `03_compositional_d5.py` | D5 | Coordinate nuisance |
| `04_falsification_controls.py` | * | wrong-W / isotropic / signal-W |
| `06_office31_sklearn.py` | D1 | sklearn + features |
| `07_vision_multilayer.py` | D3/D4 | Multi-layer Gram |
| `08_hf_style_d7.py` | D7 | HF style estimation |
| `10_hf_trainer.py` | D7 | `PMHTrainer` |
| `11_dpo_lora_style_pmh.py` | D7 | Qwen JSONL + optional DPO |

Samples: `examples/data/*.jsonl`, configs: `examples/configs/`.

---

## What this library does **not** claim

- Beating every SOTA baseline on every leaderboard.
- Fixing causal spurious correlation when the “nuisance” is not label-preserving.
- Replacing full multi-epoch RLHF from a one-epoch DPO demo.

It **does** provide a principled, repeatable way to **design the loss** from deployment geometry and to **falsify** whether matched \(\Sigma_{\mathrm{task}}\) is doing the work.

---

## Documentation map

| Doc | Content |
|-----|---------|
| [THEORY.md](docs/THEORY.md) | \(\Sigma_{\mathrm{task}}\), recipe, \(A_k\), falsification, scope |
| [getting-started.md](docs/getting-started.md) | Symptom → method |
| [nuisance_types.md](docs/nuisance_types.md) | D1–D7 data formats |
| [cli.md](docs/cli.md) | `pmh-train` jobs |

---

## Citation

Cite the Grand Unification / Matching Principle manuscript. See `CITATION.cff`.

---

## Status

**0.6.2** — architecture-agnostic docs; D1–D7; `pmh-train`; PyTorch / HF / Lightning. MIT License.
