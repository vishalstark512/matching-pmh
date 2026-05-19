# Classical ML (NumPy / scikit-learn)

No PyTorch required for **estimation** on frozen features.

```bash
pip install "matching-pmh[sklearn]"
python examples/06_office31_sklearn.py
```

```python
from pmh.numpy_api import estimate_sigma_task_numpy
from pmh.sklearn_match import MatchedSubspaceProjector

artifact = estimate_sigma_task_numpy(
    x_src, y_src, x_tgt, y_tgt,
    config=SigmaTaskConfig.for_subspace(rank=16),
)
proj = MatchedSubspaceProjector(rank=16).fit(x_src, y_src, x_tgt, y_tgt)
x_proj = proj.transform(x_src)
```

Use with `LogisticRegression`, `SVC`, etc. on projected features.

## CORAL baseline

```python
from pmh.baselines.coral import coral_align

x_src_aligned, x_tgt = coral_align(x_src, x_tgt)
```

See `examples/06_office31_sklearn.py` (includes B0, matched, wrong-W, CORAL).
