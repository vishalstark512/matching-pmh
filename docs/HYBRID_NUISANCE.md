# Hybrid nuisances (multiple Σ̂)

When deployment has **more than one** label-preserving variation (e.g. domain shift **and** augmentation modes), estimate **separate** artifacts and sum matched PMH terms.

## PyTorch

```python
from pmh import PMHTrainer, PMHConfig, build_hybrid_trainer
from pmh.estimate import estimate_from_config
from pmh.config import SigmaTaskConfig

# Phase A — two estimates
art_domain = estimate_from_config(SigmaTaskConfig.for_domain(rank=32), h_src, h_tgt)
art_aug = estimate_from_config(SigmaTaskConfig.for_augmentation(), aug_deltas=deltas)

trainer = build_hybrid_trainer(model, [art_domain, art_aug], hook=backbone, pmh_config=PMHConfig.balanced())
trainer.fit(train_loader)  # skip re-estimate if artifacts passed
```

Or inside one trainer:

```python
trainer = PMHTrainer(model, hook=backbone, nuisance="domain_shift")
trainer.estimate(source_batches=src, target_batches=tgt)
trainer.add_artifact(art_aug)  # second Sigma
trainer._bind_pmh_loss()
```

Uses :class:`pmh.multi.MultiPMHLoss` (additive penalties, single cap vs task loss).

## sklearn / NumPy

Train with two `PMHLoss` terms in your PyTorch loop, or project features twice (rare). Prefer PyTorch `MultiPMHLoss` for hybrid training.
