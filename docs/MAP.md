# Documentation map

**Canonical adoption path** — everything else is optional or research.

```
APPLICATIONS.md  →  GOLDEN_PATHS.md (one section)  →  FIRST_HOUR  →  GETTING_STARTED
        ↑                      ↑
   pmh-train route      pmh-train wizard
```

---

## Adopt tab (read in order)

| # | File | Role |
|---|------|------|
| 1 | [APPLICATIONS.md](APPLICATIONS.md) | **Primary:** finder, nuisances, per-app walkthroughs |
| 2 | [START_HERE.md](START_HERE.md) | Three gates if still unsure |
| 3 | [GOLDEN_PATHS.md](GOLDEN_PATHS.md) | G1, G1b, G2, G3, G3b, G4 code |
| 4 | [WHEN_PMH_HELPS.md](WHEN_PMH_HELPS.md) | Honest expectations + decision flowchart |
| 5 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Errors, preflight, glossary |

**Optional background:** [WHAT_IS_PMH.md](WHAT_IS_PMH.md) (concept, no paper).

**Do not read first:** [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) (use `route` first), [walkthroughs/](walkthroughs/index.md), [THEORY.md](THEORY.md).

---

## Integrate tab (after step 1–2 work)

| Group | Files |
|-------|--------|
| **Onboarding** | [FIRST_HOUR.md](FIRST_HOUR.md), [GETTING_STARTED.md](GETTING_STARTED.md) |
| **CLI** | [cli.md](cli.md) — `route`, `wizard`, `doctor`, `validate`, `estimate` |
| **Shift reference** | [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md), [COMPARE_TO_CORAL.md](COMPARE_TO_CORAL.md) |
| **Data / ship** | [DATA_LAYOUT.md](DATA_LAYOUT.md), [DEPLOYMENT.md](DEPLOYMENT.md), [CUSTOM_GEOMETRY.md](CUSTOM_GEOMETRY.md) |
| **Stacks** | [hooks.md](hooks.md), [sklearn.md](sklearn.md), [integrations-lightning.md](integrations-lightning.md), [integrations-hf-trainer.md](integrations-hf-trainer.md), [integrations.md](integrations.md), [integrations-hf.md](integrations-hf.md) |
| **API** | [api/developer.md](api/developer.md), [api/pmh-trainer.md](api/pmh-trainer.md), [api/subtypes.md](api/subtypes.md), [api/custom.md](api/custom.md), [api/deployment.md](api/deployment.md), [training.md](training.md) |
| **Try without install** | [COLAB.md](COLAB.md), [DEMO_OUTPUT.md](DEMO_OUTPUT.md) |

---

## Gallery tab

| File | Maps to |
|------|---------|
| [gallery/README.md](gallery/README.md) | Index |
| [gallery/vision.md](gallery/vision.md) | G1 / pose / detection |
| [gallery/tabular.md](gallery/tabular.md) | G2 |
| [gallery/nlp.md](gallery/nlp.md) | G3 / G3b |

---

## Research tab (paper replication)

| File | Role |
|------|------|
| [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) | Block T1–T7 ↔ subtype D1–D7 |
| [FIDELITY_BY_SUBTYPE.md](FIDELITY_BY_SUBTYPE.md) | Default vs calibrator |
| [recipes/README.md](recipes/README.md) | Exemplar cards D1/D2/D4/D7 |
| [walkthroughs/index.md](walkthroughs/index.md) | 19 deep guides (tiered) |
| [walkthroughs/08-falsification-controls.md](walkthroughs/08-falsification-controls.md) | Controls (also adoption step 5) |
| [BENCHMARKS.md](BENCHMARKS.md), [CORRECT_USAGE.md](CORRECT_USAGE.md) | Benchmarks + contracts |

---

## Reference tab

| File | Role |
|------|------|
| [estimators/index.md](estimators/index.md) | D1–D7 lemma pages |
| [THEORY.md](THEORY.md) | Math |
| [datasets.md](datasets.md) | Bundled datasets |

---

## Legacy URLs (redirect stubs)

| Old path | Read instead |
|----------|--------------|
| [QUICKSTART.md](QUICKSTART.md) | [APPLICATIONS.md](APPLICATIONS.md) |
| [getting-started.md](getting-started.md) | [APPLICATIONS.md](APPLICATIONS.md) |
| [CHOOSE_YOUR_SETUP.md](CHOOSE_YOUR_SETUP.md) | [APPLICATIONS.md](APPLICATIONS.md) |
| [ADAPT_YOUR_PIPELINE.md](ADAPT_YOUR_PIPELINE.md) | [GETTING_STARTED.md](GETTING_STARTED.md) |
| [nuisance_types.md](nuisance_types.md) | [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) |

---

## Contributors

[DOCS_GUIDE.md](DOCS_GUIDE.md) — how to add docs without breaking the adoption path.
