# Getting started

## What you are doing

1. **Name the deployment nuisance** — what can change at test time without changing the label?
2. **Pick estimator D1–D7** that matches that nuisance family.
3. **Estimate** \(\Sigma_{\mathrm{task}}\) and check **preflight** (eigengap).
4. **Train** with `PMHLoss` on top of your task loss.
5. **Evaluate** matched vs **wrong-W** and **isotropic** (and **signal-W** when you have a clear signal subspace).

See the [README](../README.md) for a full decision table and use-case walkthroughs.

## 1. Choose an estimator (Lemma D1–D7)

| Symptom / deployment story | Use | Details |
|----------------------------|-----|---------|
| Different dataset, camera, or site (unlabeled target OK) | [D4](estimators/d4.md) | Domain Gram |
| Paired domains + labels (subspace) | [D1](estimators/d1.md) | Cross-domain SVD |
| Unstructured sensor / acquisition noise | [D2](estimators/d2.md) | Isotropic |
| Known augmentation modes (color, blur, …) | [D3](estimators/d3.md) | Augmentation stack |
| Nuisance lives on specific coordinates (atoms, tokens) | [D5](estimators/d5.md) | Compositional block |
| Drift along time in sequences | [D6](estimators/d6.md) | Temporal residual |
| LLM style / format with fixed semantics | [D7](estimators/d7.md) | Style Gram + JSONL |

## 2. Estimate and save

```python
from pmh import SigmaTaskConfig, estimate_from_config

cfg = SigmaTaskConfig.for_domain(rank=64)
artifact = estimate_from_config(cfg, h_source, h_target)
print(artifact.preflight)  # pass | marginal | fail
artifact.save("checkpoints/run1/sigma")
```

CLI equivalent: `pmh-train estimate --config job.json` — see [CLI](cli.md).

## 3. Train (PyTorch)

```python
from pmh import PMHLoss, PMHConfig

pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2))
h = backbone(x)
total, term = pmh.capped_total(task_loss, h)
```

Integrations: [PyTorch](integrations.md), [Hugging Face](integrations-hf.md), [Lightning](integrations-lightning.md).

## 4. Controls (required for credible claims)

| Arm | `PMHLoss` mode | Expected |
|-----|----------------|----------|
| Matched | default | Best geometry / robustness when theory applies |
| Wrong subspace | `wrong_w` | Weaker or misaligned |
| Isotropic | `isotropic` | Uninformative directions |
| Signal (optional) | `signal_W` via projector | Should **hurt** task |

Example: `examples/04_falsification_controls.py`.
