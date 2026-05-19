# Getting started

## 1. Choose an estimator (Lemma D1–D7)

| Symptom | Use |
|---------|-----|
| Domain / dataset shift | [D4](estimators/d4.md) or [D1](estimators/d1.md) |
| Unstructured noise | [D2](estimators/d2.md) |
| Known aug modes | [D3](estimators/d3.md) |
| Nuisance coordinates | [D5](estimators/d5.md) |
| Temporal drift | [D6](estimators/d6.md) |
| Style / alignment | [D7](estimators/d7.md) |

## 2. Estimate and save

```python
from pmh import SigmaTaskConfig, estimate_from_config

cfg = SigmaTaskConfig.for_domain(rank=64)
artifact = estimate_from_config(cfg, h_source, h_target)
print(artifact.preflight)  # pass | marginal | fail
artifact.save("checkpoints/run1/sigma")
```

## 3. Train (PyTorch)

```python
from pmh import PMHLoss, PMHConfig

pmh = PMHLoss(artifact, PMHConfig(weight=0.3, warmup_epochs=2))
h = backbone(x)
total, term = pmh.capped_total(task_loss, h)
```

Or use [PMHCallback](integrations.md) for a full epoch loop.

## 4. Controls

Always report **matched**, **wrong-W**, and (when applicable) **signal-W** arms together.
