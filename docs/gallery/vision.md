# Gallery: vision domain shift (D4)

```python
import torch
from torch.utils.data import DataLoader
from pmh import PMHTrainer, PMHConfig, compare_arms

# --- YOUR model (example: timm or torchvision) ---
# backbone = ...
# head = ...
# model = nn.Sequential(backbone, head)

# --- YOUR loaders: source domain, target domain (unlabeled OK for D4), train ---
# src_loader, tgt_loader, train_loader = ...

trainer = PMHTrainer(
    model,
    hook="backbone",  # or encoder_timm(backbone)
    head=head,
    nuisance="domain_shift",
    rank=32,
    pmh_config=PMHConfig.balanced(),
    artifact_path="artifacts/vision_sigma.pt",
)
trainer.fit(
    train_loader,
    source_batches=src_loader,
    target_batches=tgt_loader,
    epochs=20,
)

# Optional: falsification table on your val loader
# compare_arms(trainer.artifact_, model_factory, setup_model, train_loader, val_loader, epochs=10)
```

Walkthrough: [ResNet D4](../walkthroughs/02-resnet-vision-d4.md) · Hooks: [hooks.md](../hooks.md)
