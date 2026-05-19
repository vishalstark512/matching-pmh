# matching-pmh

Estimate **$\Sigma_{\mathrm{task}}$** (deployment nuisance covariance) and train with **matched PMH** penalties.

Companion to the paper *The Matching Principle* — this package is a **separate repository** from the manuscript experiments.

## Install

```bash
pip install matching-pmh
# or from source
pip install -e ".[dev,sklearn,docs]"
```

## Minimal example

```python
from pmh import SigmaTaskConfig, estimate_from_config, PMHLoss, PMHConfig
import torch

artifact = estimate_from_config(
    SigmaTaskConfig.for_domain(rank=32),
    source_feats, target_feats,
)
pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3))
```

See [Getting started](getting-started.md) and [Estimators](estimators/index.md).
