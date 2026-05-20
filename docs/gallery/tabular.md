# Gallery: tabular — frozen features, sklearn

**You have:** matrices `x_source`, `y_source` and `x_target` (same feature dimension), already embedded (CLIP, ResNet pool, tabular encoder, …).

**You do:** fit `PMHMatcher` on source+target, then train a sklearn classifier on adapted source features (or use a `Pipeline`).

```python
import numpy as np
from pmh import PMHMatcher, compare_arms_sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# YOUR: x_source, y_source, x_target, y_target  shape [N, d]
# Target labels optional for adaptation; needed for some benchmarks.

matcher = PMHMatcher(nuisance="domain_shift")
matcher.fit(x_source, x_target)  # or fit(xs, ys, xt, yt) if both labeled

x_train_adapted = matcher.transform(x_source)

clf = LogisticRegression(max_iter=500)
clf.fit(x_train_adapted, y_source)
# Evaluate clf on held-out TARGET domain — not source only.

# Optional (after basics): falsification table
# compare_arms_sklearn(xs, ys, xt, yt, preset="t1_synthetic_sklearn", report_dir="results/run1")
```

**Try first:** [Colab sklearn notebook](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/sklearn_frozen_features_first_run.ipynb) · `examples/06_office31_sklearn.py` · [Walkthrough 3](../walkthroughs/03-office31-sklearn-d1.md)

**Researchers:** Office-31 preset `t1_office31_sklearn` (rank 32, pool/test split) — see [CORRECT_USAGE](../CORRECT_USAGE.md), not required for first integration.
