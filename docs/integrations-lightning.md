# PyTorch Lightning

Install: `pip install "matching-pmh[lightning]"`

Lightning does not auto-wrap `training_step`; add PMH explicitly:

```python
from pmh.integrations.lightning import PMHLightningCallback, add_pmh_to_loss

class LitModel(pl.LightningModule):
    def __init__(self, pmh_loss):
        super().__init__()
        self.pmh_loss = pmh_loss
        ...

    def training_step(self, batch, batch_idx):
        x, y = batch
        task = F.cross_entropy(self.head(self.backbone(x)), y)
        total, pmh = add_pmh_to_loss(self, (x,), task, self.pmh_loss)
        return total

trainer = pl.Trainer(callbacks=[PMHLightningCallback.from_artifact(artifact)])
```

`PMHLightningCallback` calls `pmh_loss.set_epoch` each epoch.

Example: `examples/09_lightning_module.py`
