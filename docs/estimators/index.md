# Estimators overview (D1–D7)

!!! tip "Adoption path"
    Pick nuisance in plain English: [Find your application](../APPLICATIONS.md).  
    Read lemma pages here when you need formulas or factory kwargs.

| Lemma | Page | Plain nuisance |
|-------|------|----------------|
| D1 | [d1.md](d1.md) | Cross-site class geometry |
| D2 | [d2.md](d2.md) | Isotropic |
| D3 | [d3.md](d3.md) | Named augmentations |
| D4 | [d4.md](d4.md) | Site / sensor appearance |
| D5 | [d5.md](d5.md) | Indexed coordinates |
| D6 | [d6.md](d6.md) | Temporal drift |
| D7 | [d7.md](d7.md) | Style / alignment |

```python
from pmh import SigmaTaskConfig, estimate_from_config

SigmaTaskConfig.for_subspace(rank=16)       # D1
SigmaTaskConfig.for_isotropic(dim, 0.1)    # D2
SigmaTaskConfig.for_augmentation()         # D3
SigmaTaskConfig.for_domain(rank=64)        # D4
SigmaTaskConfig.for_compositional([...])   # D5
SigmaTaskConfig.for_temporal()             # D6
SigmaTaskConfig.for_alignment(rank=128)    # D7
```

[NUISANCE_SUBTYPES.md](../NUISANCE_SUBTYPES.md) · [FIDELITY_BY_SUBTYPE.md](../FIDELITY_BY_SUBTYPE.md)
