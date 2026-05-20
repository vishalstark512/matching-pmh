# PyTorch Lightning (golden path G1b)

Use this when you **already** have a `LightningModule` and want PMH added inside `training_step`.

| Path | Doc |
|------|-----|
| **G1b (this page)** | `add_pmh_to_loss` + `PMHLightningCallback` |
| **G1** | [Golden paths — plain PyTorch](GOLDEN_PATHS.md#g1--pytorch-two-domains) · `robust_fit` / `PMHTrainer` |

Install: `pip install "matching-pmh[lightning]"`

---

## Minimal pattern

```python
from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, estimate_from_config
from pmh.integrations.lightning import PMHLightningCallback, add_pmh_to_loss, _require_lightning
import torch.nn.functional as F

pl = _require_lightning()

# Phase A
with torch.no_grad():
    h_src = net.backbone(x_source)
    h_tgt = net.backbone(x_target)
artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=32), h_src, h_tgt)
pmh_loss = PMHLoss(artifact, PMHConfig.balanced())

class Lit(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        task = F.cross_entropy(self.net(x), y)
        total, _ = add_pmh_to_loss(self.net, (x,), task, pmh_loss, backbone_attr="backbone")
        return total

trainer = pl.Trainer(callbacks=[PMHLightningCallback.from_artifact(artifact, PMHConfig.balanced())])
trainer.fit(Lit(), train_loader)
```

Template: [`lightning_g1b_minimal.py`](https://github.com/vishalstark512/matching-pmh/blob/main/templates/matching-pmh-starter/lightning_g1b_minimal.py)  
Example: `examples/09_lightning_module.py` · [walkthrough 10](walkthroughs/10-lightning.md)

---

## Notes

- **`PMHLightningCallback`** only sets epoch for cap/warmup; you must call **`add_pmh_to_loss`** in `training_step`.
- **`backbone_attr`**: name of the submodule that maps `x → h` (default `"backbone"`).
- Dict batches: pass `inputs_key="image"` or include `"x"` / `"inputs"` keys.
- Custom artifact: [G4 — Custom geometry](GOLDEN_PATHS.md#g4--custom-geometry-your-deltas-or-w) · `PMHLoss(artifact, ...)`.
