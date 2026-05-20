# Documentation guide (contributors)

## Slim site (v3)

**Users see ~12 nav pages.** Everything else stays in the repo for search / deep links but is a **redirect stub** or unlisted.

| Nav tab | Pages |
|---------|--------|
| **Adopt** | `FIVE_STEP_RECIPE`, `APPLICATIONS`, `GOLDEN_PATHS`, `INTEGRATE`, `WHEN_PMH_HELPS`, `TROUBLESHOOTING` |
| **Reference** | `WHAT_IS_DEPLOYMENT_SHIFT` (linked from Adopt), `PMH_PARAMETERS`, `NUISANCE_SUBTYPES` (appendix), `api/index` |
| **Evidence** | `walkthroughs/index`, `08-falsification`, `PAPER_ALIGNMENT` |
| **Reference** | `estimators/index`, `THEORY` |
| **Contributors** | `DOCS_GUIDE`, `META_STRUCTURE` |

Regenerate compact pages:

```bash
python scripts/gen_compact_applications.py
python scripts/gen_compact_nuisance_subtypes.py
```

Removed pages redirect via `mkdocs.yml` only (no stub files). To delete more consolidated files:

```bash
python scripts/delete_consolidated_docs.py
```

---

## What to edit for a feature

| Change | Edit |
|--------|------|
| New user task | `task_router.py` → run `gen_compact_applications.py` |
| Five-step / product story | `FIVE_STEP_RECIPE.md`, `recipe.py` |
| Stack integration | `INTEGRATE.md`, `GOLDEN_PATHS.md` (one section) |
| New estimator | `estimators/dk.md` + `estimators/index.md` (not Adopt tab) |
| Paper block | `walkthroughs/` + `PAPER_ALIGNMENT.md` (Evidence only) |
| CLI | `cli/main.py`, one line in `INTEGRATE.md` |

**Do not** add new top-level Adopt pages without removing one or folding into `INTEGRATE.md`.

---

## Sync tests

```bash
python -m pytest tests/test_applications_doc_sync.py tests/test_task_router.py -q
python -m mkdocs build
```

---

## Code layout

[META_STRUCTURE.md](META_STRUCTURE.md)
