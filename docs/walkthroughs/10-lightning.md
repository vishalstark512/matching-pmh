# Walkthrough 10: PyTorch Lightning

**Goal:** Add PMH inside a `LightningModule` without rewriting the Trainer.

**Estimator:** D4 (example).  
**Script:** `examples/09_lightning_module.py`

---

## Prerequisites

```bash
pip install "matching-pmh[lightning]"
```

---

## Step 1 — Estimate artifact (Phase A)

```python
artifact = estimate_from_config(
    SigmaTaskConfig.for_domain(rank=4),
    net.backbone(x[:100]).detach(),
    net.backbone(x[100:] + shift).detach(),
)
```

---

## Step 2 — `training_step` with `add_pmh_to_loss`

```python
from pmh.integrations.lightning import add_pmh_to_loss, PMHLightningCallback

class PMHLit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.pmh_loss = PMHLoss(artifact, pmh_cfg)

    def training_step(self, batch, batch_idx=0):
        x, y = batch
        task = F.cross_entropy(self.net(x), y)
        total, _ = add_pmh_to_loss(
            self.net, (x,), task, self.pmh_loss, backbone_attr="backbone"
        )
        return total
```

`backbone_attr` must name the submodule whose output is $h$.

---

## Step 3 — Callback (optional)

```python
trainer = pl.Trainer(
    callbacks=[PMHLightningCallback.from_artifact(artifact, pmh_cfg)],
    ...
)
```

Callback syncs epoch for PMH warmup schedule.

---

## Run

```bash
python examples/09_lightning_module.py
```

---

## Adapt

| Toy `SmallNet` | Your module |
|----------------|-------------|
| `self.net.backbone` | `self.encoder` / `self.model.visual` |
| `add_pmh_to_loss(..., backbone_attr="backbone")` | Match attribute name |

See [integrations-lightning.md](../integrations-lightning.md).
