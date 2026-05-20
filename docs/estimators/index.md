# Estimators (D1–D7)

One $\hat{\Sigma}_{\text{task}}$ per lemma. Pick your row in [tasks/index.md](../tasks/index.md) or use `suggest_subtype()`.

```python
from pmh import SigmaTaskConfig, estimate_from_config

SigmaTaskConfig.for_subspace(rank=16)           # D1 — labels on A and B
SigmaTaskConfig.for_isotropic(dim=768, noise_level=0.1)  # D2
SigmaTaskConfig.for_augmentation()               # D3 — aug_deltas
SigmaTaskConfig.for_domain(rank=32)              # D4 — default product path
SigmaTaskConfig.for_compositional(nuisance_indices=[...])  # D5
SigmaTaskConfig.for_temporal()                   # D6 — sequences
SigmaTaskConfig.for_alignment(rank=128)          # D7 — style / PGD deltas
```

| Dk | Name | Config factory | Main inputs |
|----|------|----------------|-------------|
| D1 | Subspace SVD | `for_subspace` | `x_src, y_src, x_tgt, y_tgt` |
| D2 | Isotropic | `for_isotropic` | `dim`, `noise_level` |
| D3 | Aug modes | `for_augmentation` | `aug_deltas` |
| D4 | Domain Gram | `for_domain` | source + target features |
| D5 | Compositional | `for_compositional` | features + `nuisance_indices` |
| D6 | Temporal | `for_temporal` | `sequences` |
| D7 | Alignment | `for_alignment` | `style_jsonl` or deltas |

`pmh.estimators` registry · optional calibrators: `pmh.calibrate.*` · theory: [`main.pdf`](../../main.pdf)
