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
matcher.fit(x_source, x_target)

# Or class-aligned subspace (D1)
matcher = PMHMatcher(nuisance="subspace", rank=16)
matcher.fit(x_source, y_source, x_target, y_target)

pipe = Pipeline([
    ("pmh", matcher),
    ("clf", LogisticRegression(max_iter=500)),
])
pipe.fit(x_train, y_train)
```

`matcher.artifact_` is a `SigmaTaskEstimate` for PyTorch training with `PMHLoss`.

```python
from pmh import suggest_nuisance, compare_arms_sklearn, tune_sklearn_matcher

print(suggest_nuisance(has_target_labels=True, has_target_domain=True))
compare_arms_sklearn(x_src, y_src, x_tgt, y_tgt, rank=16, report_dir="results/run1")
```

Use `nuisance="auto"` on `PMHMatcher` when unsure (see `suggest_nuisance` flags).

Legacy: `MatchedSubspaceProjector` (D1 only) in `pmh.sklearn_match`.

## CORAL baseline

```python
from pmh.baselines.coral import coral_align

x_src_aligned, x_tgt = coral_align(x_src, x_tgt)
```

See `examples/06_office31_sklearn.py` (includes B0, matched, wrong-W, CORAL).
