# Estimators overview

Each Lemma **D*k*** maps a structural assumption **A*k*** to an estimator $\hat\Sigma_{\mathrm{task}}$.

```python
from pmh import SigmaTaskConfig, estimate_from_config

# Factory helpers
SigmaTaskConfig.for_subspace(rank=16)      # D1
SigmaTaskConfig.for_isotropic(dim, 0.1)    # D2
SigmaTaskConfig.for_augmentation()         # D3
SigmaTaskConfig.for_domain(rank=64)        # D4
SigmaTaskConfig.for_compositional([...])   # D5
SigmaTaskConfig.for_temporal()             # D6
SigmaTaskConfig.for_alignment(rank=128)    # D7
```

Legacy API:

```python
from pmh import estimate_sigma_task
sigma = estimate_sigma_task(src, tgt, method="D4", rank=64)
```

| Page | Lemma | Paper blocks |
|------|-------|----------------|
| [D1](d1.md) | Subspace SVD | T1 |
| [D2](d2.md) | Isotropic | T2 |
| [D3](d3.md) | Aug modes | T3 |
| [D4](d4.md) | Domain Gram | T4 |
| [D5](d5.md) | Compositional | T5 |
| [D6](d6.md) | Temporal | T6 |
| [D7](d7.md) | Style / PGD | T7 |
