# PyTorch integration

## PMHCallback

```python
from pmh.integrations import PMHCallback, train_epoch_with_pmh

callback = PMHCallback.from_artifact(artifact, encoder=model.backbone, head=model.head)
stats = train_epoch_with_pmh(model, callback, train_loader, optimizer, epoch=1)
print(stats)
```

Manual step:

```python
callback.on_epoch_start(epoch)
loss, info = callback.training_step((x, y))
loss.backward()
callback.on_epoch_end()
```
