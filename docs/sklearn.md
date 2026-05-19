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

Legacy helper: `MatchedSubspaceProjector` (D1 only) remains in `pmh.sklearn_match`.

## CORAL baseline

```python
from pmh.baselines.coral import coral_align

x_src_aligned, x_tgt = coral_align(x_src, x_tgt)
```

See `examples/06_office31_sklearn.py` (includes B0, matched, wrong-W, CORAL).
