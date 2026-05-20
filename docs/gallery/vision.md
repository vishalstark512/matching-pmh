# Gallery: vision — two sites, same labels

**You have:** a PyTorch model (CNN, ViT, …), training data, and batches from a **second site** at deploy time (target labels optional).

**You do:** point `hook` at your backbone, pass source + target loaders, call `trainer.fit`.

```python
import torch
from torch.utils.data import DataLoader
from pmh import PMHTrainer, PMHConfig

# --- YOUR model ---
# backbone = ...
# head = ...
# model = nn.Sequential(backbone, head)

# --- YOUR loaders: site A (source), site B (target), supervised train ---
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
```

**Try first:** [Colab — Hospital A→B](../COLAB.md) · `examples/00_first_run_domain_shift.py`

**Go deeper:** [ResNet walkthrough](../walkthroughs/02-resnet-vision-d4.md) · [Hook cookbook](../hooks.md) · [Falsification](../walkthroughs/08-falsification-controls.md) (`compare_arms`)
