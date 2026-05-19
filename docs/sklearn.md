# Classical ML (NumPy / scikit-learn)

No PyTorch required for **estimation** on frozen features.

```bash
pip install "matching-pmh[sklearn]"
python examples/06_office31_sklearn.py
```

```python
from pmh import PMHMatcher
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Domain shift on frozen features (D4)
matcher = PMHMatcher(nuisance="domain_shift", rank=16)
matcher.fit(x_source, X_target=x_target)

# Pipeline: store target domain on the matcher (or use metadata routing)
pipe = Pipeline([
    ("pmh", PMHMatcher(nuisance="domain_shift", rank=16, X_target=x_target)),
    ("clf", LogisticRegression(max_iter=500)),
])
pipe.fit(x_source, y_source)

# Metadata routing on the matcher (sklearn >= 1.4; enable routing first):
import sklearn
sklearn.set_config(enable_metadata_routing=True)
pmh = PMHMatcher(nuisance="domain_shift", rank=16)
pmh.set_fit_request(X_target=True)
pmh.fit(x_source, y_source, X_target=x_target)

# Class-aligned subspace (D1)
matcher = PMHMatcher(nuisance="subspace", rank=16)
matcher.fit(x_source, y_source, x_target, y_target)
```

`matcher.artifact_` is a `SigmaTaskEstimate` for PyTorch training with `PMHLoss`.

```python
from pmh import (
    suggest_nuisance,
    compare_arms_sklearn,
    make_pmh_pipeline,
    default_pmh_param_grid,
    grid_search_pmh_pipeline,
)
from sklearn.model_selection import GridSearchCV

# GridSearchCV (target domain fixed on the matcher)
search = grid_search_pmh_pipeline(
    x_source, y_source, x_target,
    param_grid=default_pmh_param_grid(rank_grid=(8, 16, 32)),
    cv=5,
    return_search=True,
)
print(search.best_params_, search.best_score_)
best_pipe = search.best_estimator_

# Or build the pipeline yourself
pipe = make_pmh_pipeline(x_target, nuisance="domain_shift")
GridSearchCV(pipe, {"pmh__rank": [8, 16, 32]}, cv=5).fit(x_source, y_source)
```

```python
from pmh import suggest_nuisance, compare_arms_sklearn, tune_sklearn_matcher

print(suggest_nuisance(has_target_labels=True, has_target_domain=True))
compare_arms_sklearn(x_src, y_src, x_tgt, y_tgt, rank=16, report_dir="results/run1")

# Same grid via helper flag
tune_sklearn_matcher(
    x_src, y_src, x_tgt, y_tgt,
    scorer=None,
    use_gridsearchcv=True,
    nuisance="domain_shift",
    rank_grid=(8, 16, 32),
)
```

Use `nuisance="auto"` on `PMHMatcher` when unsure (see `suggest_nuisance` flags).

Legacy: `MatchedSubspaceProjector` (D1 only) in `pmh.sklearn_match`.

## CORAL baseline

```python
from pmh.baselines.coral import coral_align

x_src_aligned, x_tgt = coral_align(x_src, x_tgt)
```

See `examples/06_office31_sklearn.py` (includes B0, matched, wrong-W, CORAL).
