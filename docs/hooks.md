# Hook cookbook

Pick one representation `h = φ(x)` for **Phase A and Phase B**. Use `pmh.hooks` helpers or `PMHTrainer(..., hook=...)`.

## Quick pick

| Stack | Helper | `PMHTrainer` hook |
|-------|--------|-------------------|
| Custom `nn.Module` | `hook=your_submodule` | pass module |
| ResNet (torchvision) | `encoder_torchvision_resnet(m, layer="avgpool")` | `"avgpool"` or `"layer4"` |
| ViT / timm | `encoder_timm(m)` | `"backbone"` (uses `forward_features`) |
| HF causal LM | `encoder_hf_hidden_states(m, pool="last")` | use helper (input_ids batch) |
| GNN | `encoder_gnn_mean_pool(node_enc)` | custom callable |
| MLP / Sequential | `hook="0"` or first child | `"backbone"` |

```python
from pmh.hooks import list_hook_families, detect_model_family, resolve_hook

print(detect_model_family(model))
print(list_hook_families())
encoder = resolve_hook(model, "backbone")
```

---

## torchvision ResNet

```python
from pmh.hooks import encoder_torchvision_resnet
from pmh import PMHTrainer

# model = torchvision.models.resnet18(weights=...)
enc = encoder_torchvision_resnet(model, layer="avgpool")
trainer = PMHTrainer(model, hook=enc, head=model.fc, nuisance="domain_shift")
```

---

## timm / ViT

```python
# pip install timm
import timm
from pmh.hooks import encoder_timm

backbone = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
enc = encoder_timm(backbone)
```

---

## Hugging Face (hidden states)

Batch must provide `input_ids` `[B, T]`.

```python
from pmh.hooks import encoder_hf_hidden_states

enc = encoder_hf_hidden_states(model, layer=-1, pool="last")
h = enc(input_ids)  # [B, d]
```

For D7 style estimation use `pmh.integrations.huggingface` and walkthrough 6–7.

---

## GNN (mean pool)

```python
from pmh.hooks import encoder_gnn_mean_pool

enc = encoder_gnn_mean_pool(gnn_layer, batch_index=batch_vector)
```

Use D5 compositional indices when nuisance lives on specific atom/token coordinates.

---

## Register your architecture

```python
from pmh.hooks import register_hook_family

register_hook_family("my_company_backbone", {"default": "encoder", "backbone": "encoder"})
```

---

See also [ARCHITECTURES.md](ARCHITECTURES.md) and [ADAPT_YOUR_PIPELINE.md](ADAPT_YOUR_PIPELINE.md).
