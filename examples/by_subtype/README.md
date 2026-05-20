# Examples by nuisance subtype (D1–D7)

One **canonical script** per subtype. All use the same pipeline: **estimate Σ̂ → PMH → controls**.

| Subtype | Script | Notes |
|---------|--------|--------|
| **D1** | [`../06_office31_sklearn.py`](../06_office31_sklearn.py) | Frozen features, `paper_protocol=True` |
| **D2** | [`../14_vit_cls_d4.py`](../14_vit_cls_d4.py) | ViT CLS; set `nuisance="isotropic"` for pure D2 |
| **D3** | [`../18_augmentation_d3.py`](../18_augmentation_d3.py) | Finite augmentation modes |
| **D4** | [`../01_domain_shift_d4.py`](../01_domain_shift_d4.py) | Default PyTorch domain Gram |
| **D5** | [`../03_compositional_d5.py`](../03_compositional_d5.py) | `nuisance_indices` |
| **D6** | Walkthrough [11 temporal](../../docs/walkthroughs/11-temporal-d6.md) | `sequences_batches` API |
| **D7** | [`../08_hf_style_d7.py`](../08_hf_style_d7.py) | Style / format shift |

**Custom geometry (any subtype):** [`../23_custom_geometry_train.py`](../23_custom_geometry_train.py)

**Save/load artifact:** [`../02_save_load_artifact.py`](../02_save_load_artifact.py)

Docs: [NUISANCE_SUBTYPES.md](../../docs/NUISANCE_SUBTYPES.md) · [CUSTOM_GEOMETRY.md](../../docs/CUSTOM_GEOMETRY.md)
