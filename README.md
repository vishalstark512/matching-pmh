# matching-pmh

**Independent library** for the *matching principle*: estimate $\Sigma_{\mathrm{task}}$ (Lemmas D1–D7), run matched PMH penalties, save/load artifacts, and wire into training loops.

Paper: *The Matching Principle* (separate repository). **v0.2** adds typed configs, artifacts, pre-flight eigengap, `PMHLoss`, and `collect_features`.

## Install

```bash
cd matching-pmh
pip install -e ".[dev]"
pytest
```

## Quick start (v0.2)

```python
import torch
from pmh import (
    SigmaTaskConfig,
    PMHConfig,
    PMHLoss,
    estimate_from_config,
    collect_features,
)

# 1) Estimate + diagnostics
cfg = SigmaTaskConfig.for_domain(rank=64)
artifact = estimate_from_config(cfg, source_feats, target_feats)
print(artifact.preflight, artifact.eigengap)  # pass | marginal | fail

# 2) Save for another job / machine
artifact.save("checkpoints/style_sigma")

# 3) Train
pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2))
h = backbone(x)
task_loss = ...
total, pmh_term = pmh.capped_total(task_loss, h)
```

### Legacy one-liner (still supported)

```python
from pmh import estimate_sigma_task, pmh_penalty_on_rep

sigma = estimate_sigma_task(src, tgt, method="D4", rank=64)
pen = pmh_penalty_on_rep(h, sigma)
```

### Load a saved estimate

```python
from pmh import SigmaTaskEstimate, PMHLoss

artifact = SigmaTaskEstimate.load("checkpoints/style_sigma.pt")
pmh = PMHLoss(artifact)
```

## Examples

| Script | What it shows |
|--------|----------------|
| `examples/01_domain_shift_d4.py` | `collect_features` + D4 + `PMHLoss` training |
| `examples/02_save_load_artifact.py` | `.pt` + `.json` artifact I/O |
| `examples/03_compositional_d5.py` | D5 coordinate-block $\Sigma$ |
| `examples/04_falsification_controls.py` | matched / wrong-W / isotropic modes |
| `examples/05_yaml_config.py` | JSON job dict → configs |
| `examples/minimal_loop.py` | Short end-to-end loop |
| `examples/06_office31_sklearn.py` | D1 + `MatchedSubspaceProjector` + logistic |
| `examples/07_vision_multilayer.py` | `MultiLayerPMHLoss` + per-layer Gram noise |

## API map

| Goal | API |
|------|-----|
| Pick estimator | `SigmaTaskConfig.for_domain()`, `.for_isotropic()`, … |
| Estimate | `estimate_from_config(cfg, ...)` → `SigmaTaskEstimate` |
| Pre-flight | `artifact.preflight`, `preflight_eigengap(cov, rank)` |
| Train | `PMHLoss(artifact, PMHConfig(...))` |
| Controls | `PMHLoss(..., mode="wrong_w")`, `signal_W_projector` |
| Data hook | `collect_features(encoder, loader)` |

## Estimators (`method=`)

| Method | Lemma | Config helper |
|--------|-------|----------------|
| D1 | Subspace SVD | `for_subspace(rank=)` |
| D2 | Isotropic | `for_isotropic(dim, noise_level)` |
| D3 | Aug modes | `for_augmentation()` |
| D4 | Domain Gram | `for_domain(rank=)` |
| D5 | Compositional | `for_compositional(indices)` |
| D6 | Temporal | `for_temporal()` |
| D7 | Style / alignment | `for_alignment(rank=)` |

## Status

**0.5.0** — `PMHTrainer` (HF), CORAL baseline, GitHub Actions CI + PyPI publish guide.

| Extra | Install | Example |
|-------|---------|---------|
| HF Trainer | `pip install "matching-pmh[hf]"` | `examples/10_hf_trainer.py` |
| CORAL baseline | `pmh.baselines.coral` | `examples/06_office31_sklearn.py` |
| CI / PyPI | see `PUBLISHING.md` | tag `v0.5.0` |

**0.4.0** — Hugging Face D7 (`estimate_style_sigma`), Lightning (`add_pmh_to_loss`), Office-31 features (`--office31-root`).

| Extra | Install | Example |
|-------|---------|---------|
| HF | `pip install "matching-pmh[hf]"` | `examples/08_hf_style_d7.py` |
| Lightning | `pip install "matching-pmh[lightning]"` | `examples/09_lightning_module.py` |
| Vision / Office-31 | `pip install "matching-pmh[vision]"` | `examples/06_office31_sklearn.py --office31-root ...` |
