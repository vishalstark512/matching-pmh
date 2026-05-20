# Contributing to matching-pmh

Thank you for helping make deployment-geometry training easy to adopt. This project aims for **research-lab quality**: clear docs, runnable examples, and falsifiable defaults.

---

## Development setup

```bash
git clone https://github.com/vishalstark512/matching-pmh.git
cd matching-pmh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[dev,all]"
pytest -q
ruff check src tests examples
```

---

## Data and benchmarks

**Do not commit datasets, archives, or extracted features** (Office-31, `.npy`, `.tar.gz`, checkpoints).  
See [docs/DATA_POLICY.md](docs/DATA_POLICY.md). Benchmark docs may include **metric tables only**.

---

## What to contribute

| Type | Where |
|------|--------|
| Bug fix | PR + test in `tests/` |
| New walkthrough | `docs/walkthroughs/` + `examples/*.py` + index table |
| Estimator improvement | `src/pmh/estimators/` + `tests/` |
| Integration (framework X) | `src/pmh/integrations/` + doc page |
| Docs only | `docs/`, `README.md` |

---

## Pull request checklist

- [ ] `pytest -q` passes locally
- [ ] New behavior has a test or a smoke-runnable example
- [ ] `CHANGELOG.md` updated under “Unreleased” or new version
- [ ] User-facing changes reflected in walkthroughs, `docs/GOLDEN_PATHS.md`, or `docs/index.md` (see `docs/DOCS_GUIDE.md`)
- [ ] No secrets or API tokens in the diff

---

## Code style

- Python ≥ 3.10, type hints encouraged
- `ruff` line length 100 (see `pyproject.toml`)
- Keep examples **short and linear**—one clear Phase A, one Phase B

---

## Reporting issues

Include:

1. `pip show matching-pmh` version
2. Minimal script to reproduce
3. Expected vs actual `artifact.preflight` / loss behavior

Use GitHub Issues: https://github.com/vishalstark512/matching-pmh/issues

---

## Citation

Contributors are acknowledged via git history; research citations should point to the Matching Principle manuscript (see `CITATION.cff`).
