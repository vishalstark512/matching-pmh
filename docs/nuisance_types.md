# Nuisance types D1–D7 (redirect)

**User guide (pick subtype first):** [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md)

**API / CLI cheat sheet:** [Estimators reference](estimators/index.md) · `pmh-train list-methods`

**Paper script vs default estimator:** [FIDELITY_BY_SUBTYPE.md](FIDELITY_BY_SUBTYPE.md)

---

## Quick CLI

```bash
pmh-train list-methods    # D1–D7 + subtype one-liners
pmh-train wizard          # interactive routing
```

```python
from pmh import suggest_subtype, SigmaTaskConfig, estimate_from_config

rec = suggest_subtype(has_target_domain=True, has_target_labels=True)
# rec.method, rec.nuisance, rec.reason
```

Legacy per-method snippets from this page now live under [estimators/](estimators/index.md) (D1–D7).
