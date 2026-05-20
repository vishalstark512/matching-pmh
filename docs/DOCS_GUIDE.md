# Documentation guide (contributors)

## Information architecture (v2)

**Rule:** one adoption ladder. Never add a second “start here.”

| Layer | Tab | Audience |
|-------|-----|----------|
| L0 | Home `index.md` | 10-line pointer to APPLICATIONS |
| L1 | **Adopt** | All new developers |
| L2 | **Integrate** | Wiring PMH into a stack |
| L3 | **Gallery** | Copy-paste by domain |
| L4 | **Research** | Paper / benchmarks / walkthroughs |
| L5 | **Reference** | Lemmas, theory, primitives |
| L6 | **Contributors** | Plans, roadmap |

**Source of truth for applications:** `src/pmh/task_router.py` + [APPLICATIONS.md](APPLICATIONS.md) (keep in sync).

---

## What to edit for a feature

| Change | Edit |
|--------|------|
| New user-facing task | `task_router.py` + `APPLICATIONS.md` section + optional `gallery/` |
| New integration pattern | `GOLDEN_PATHS.md` only |
| New subtype / estimator | `NUISANCE_SUBTYPES.md` + `estimators/dk.md` |
| Paper block | `PAPER_ALIGNMENT.md` + `recipes/` |
| CLI command | `cli.md` + `src/pmh/cli/main.py` |

**Do not** add top-level Adopt pages without updating this guide.

---

## Redirect stubs

Keep these files **short** (redirect only). Do not expand them.

| Stub | Target |
|------|--------|
| `QUICKSTART.md`, `getting-started.md`, `CHOOSE_YOUR_SETUP.md` | `APPLICATIONS.md` |
| `ADAPT_YOUR_PIPELINE.md` | `GETTING_STARTED.md` |
| `nuisance_types.md` | `NUISANCE_SUBTYPES.md` |

---

## Adoption banner (optional on long pages)

```markdown
!!! tip "Adoption path"
    New here? [Find your application](APPLICATIONS.md) → [Golden paths](GOLDEN_PATHS.md) (one section).
```

Use on Research / Reference pages, not on APPLICATIONS or GOLDEN_PATHS.

---

## mkdocs nav

Edit `mkdocs.yml` only when adding a **new Integrate or Research** page. Adopt tab should stay ≤6 entries.

---

## Pre-release check

```bash
python -m mkdocs build
python -m pytest tests/test_task_router.py -q
pmh-train route --list
```

Fix broken links to `GOLDEN_PATHS.md#g1` … `#g4` (use HTML `<a id="g1"></a>` anchors).
