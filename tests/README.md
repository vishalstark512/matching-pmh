# Tests

Default CI (fast):

```bash
pytest
```

Uses `addopts = -m 'not slow'` from `pyproject.toml` (skips nbconvert notebook runs and other slow smokes).

Optional slow checks:

```bash
pytest -m slow
```

`paper_code/` is excluded via `norecursedirs` (paper reproduction only).
