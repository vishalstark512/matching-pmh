# Training

## Representation PMH

```python
pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2))
total, raw = pmh.capped_total(task_loss, h)
```

Modes: `PMHLoss(..., mode="wrong_w")`, `mode="isotropic"`.

## Multilayer vision PMH

```python
from pmh.vision import MultiLayerPMHLoss, gram_sample_noise

pmh = MultiLayerPMHLoss(("layer3", "layer4"), PMHConfig(weight=0.2))
total, term = pmh.capped_total(task_loss, feats_clean, feats_noisy)
```

See `examples/07_vision_multilayer.py`.
