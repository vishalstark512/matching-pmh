# Documentation guide (contributors)

How the docs site is organized after the **subtype-first cleanup** (fewer entry points, redirects for legacy URLs).

---

## User journey (sidebar)

| Order | Page | Purpose |
|-------|------|---------|
| 1 | `index.md` | Single hub + reading order |
| 2 | `WHAT_IS_PMH.md` | Concept, no paper |
| 3 | `WHEN_PMH_HELPS.md` | Honest expectations |
| 4 | `FIRST_HOUR.md` | Install, demo, wizard |
| 5 | `NUISANCE_SUBTYPES.md` | Pick D1–D7 |
| 6 | `GOLDEN_PATHS.md` | G1, **G1b** (Lightning), G2, G3, **G3b** (HF Trainer), G4 |
| 7 | `GETTING_STARTED.md` | Afternoon integration |
| 8 | `TROUBLESHOOTING.md` | Errors + glossary |

**Integrate** tab: CLI, Colab, API refs, hooks, sklearn/HF/Lightning, CORAL.

**Research** tab: paper alignment, fidelity, recipes, walkthrough index, benchmarks — not required for first adoption.

**Reference** tab: per-estimator pages, training API, theory.

---

## Redirect stubs (keep files, do not expand)

These URLs are linked from old READMEs, issues, and walkthroughs:

| File | Points to |
|------|-----------|
| `QUICKSTART.md` | `FIRST_HOUR.md` |
| `CHOOSE_YOUR_SETUP.md` | `NUISANCE_SUBTYPES.md` + `GOLDEN_PATHS.md` |
| `ADAPT_YOUR_PIPELINE.md` | `GETTING_STARTED.md` |
| `nuisance_types.md` | `NUISANCE_SUBTYPES.md` + `estimators/` |
| `getting-started.md` | `index.md` |

When adding links in new content, **prefer the target page**, not the stub.

---

## Removed from main nav (still in repo)

- Duplicate **Theory** top-level tab (use Reference → Theory)
- **19 numbered walkthroughs** in sidebar (listed only via `walkthroughs/index.md`)
- **Estimators** duplicate under Reference + old Nuisance cookbook
- **Contributors-only** plans: `NUISANCE_SUBTYPE_PLAN.md`, `DEVELOPER_ONBOARDING_PLAN.md`

Walkthrough files remain; link from index or walkthrough hub.

---

## Adding new docs

1. **User-facing feature** → update `GOLDEN_PATHS.md` or `api/developer.md`, not a new top-level page.
2. **New subtype / estimator** → `NUISANCE_SUBTYPES.md` + `estimators/dk.md` + optional recipe card.
3. **Paper block** → row in `PAPER_ALIGNMENT.md` + recipe or walkthrough, not a new “start here” page.
