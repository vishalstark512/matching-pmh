# Gallery: tabular / frozen features (D1 or D4)

```python
import numpy as np
from pmh import PMHMatcher, suggest_nuisance, compare_arms_sklearn, tune_sklearn_matcher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# YOUR features: x_source, y_source, x_target, y_target  shape [N, d]

sug = suggest_nuisance(
    has_source_labels=True,
    has_target_labels=True,  # False → D4
    has_target_domain=True,
)
matcher = PMHMatcher(nuisance=sug.nuisance, rank=16)  # or nuisance="auto", ...
matcher.fit(x_source, y_source, x_target, y_target)

pipe = Pipeline([
    ("pmh", matcher),
    ("clf", LogisticRegression(max_iter=500)),
])
# pipe.fit(x_train, y_train)  # if using sklearn Pipeline on source only

compare_arms_sklearn(
    x_source, y_source, x_target, y_target,
    rank=16,
    report_dir="results/tabular_benchmark",
)
```

Example script: `examples/06_office31_sklearn.py`
