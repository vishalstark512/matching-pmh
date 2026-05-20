# Nuisance-subtype product plan

**North star:** `matching-pmh` is a **pipeline for deployment nuisances by structural subtype (D1–D7)**.  
Paper blocks T1–T7 are **exemplars** that validated each subtype—not thirteen separate products.

**Uniform pipeline (every subtype):**

1. **Identify** \(\hat\Sigma_{\mathrm{task}}\) (or \(W\)) from data  
2. **Apply** PMH (Mode A: Jacobian on \(h\) · Mode B: feature projection)  
3. **Validate** with falsification controls (matched / wrong-W / task-appropriate negatives)

**Status legend:** `—` not started · `~` in progress · `✓` done

---

## Phase 0 — Principles and plan

| ID | Deliverable | Acceptance | Status |
|----|-------------|------------|--------|
| 0.1 | This plan (`NUISANCE_SUBTYPE_PLAN.md`) | Phases 1–6 defined with checklists | ✓ |
| 0.2 | One-sentence product rule in README | “Pick subtype D1–D7, not paper block” | ✓ (Phase 1) |

---

## Phase 1 — Subtype guide (front door)

| ID | Deliverable | Acceptance | Status |
|----|-------------|------------|--------|
| 1.1 | [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) | Decision tree + per-Dk “similar structure” + API + exemplar blocks + anti-patterns | ✓ |
| 1.2 | mkdocs nav: **Nuisance subtypes** under Start here | Above estimators / paper blocks | ✓ |
| 1.3 | `index.md`, `WHAT_IS_PMH.md`, `nuisance_types.md` | Link to subtype guide first | ✓ |
| 1.4 | [recipes/README.md](recipes/README.md) | Reframed subtype-first; paper blocks as exemplars | ✓ |

---

## Phase 2 — Routing (wizard + suggest)

| ID | Deliverable | Acceptance | Status |
|----|-------------|------------|--------|
| 2.1 | `pmh.subtypes` registry (D1–D7 metadata) | `list_subtypes()`, `get_subtype("D4")` | ✓ |
| 2.2 | Wizard: “What kind of deploy shift?” for PyTorch | Maps to `suggest_nuisance` flags | ✓ |
| 2.3 | `SetupRecommendation.lemma` + subtype doc link in output | Printed by `pmh-train wizard` | ✓ |
| 2.4 | `check_applicability` mentions suggested lemma | Optional `lemma=` in summary | ✓ |
| 2.5 | Tests: `test_subtypes.py`, wizard subtype paths | Non-interactive coverage | ✓ |

---

## Phase 3 — Identification refinements (within subtype)

| ID | Deliverable | Acceptance | Status |
|----|-------------|------------|--------|
| 3.1 | [FIDELITY_BY_SUBTYPE.md](FIDELITY_BY_SUBTYPE.md) | Per Dk: default estimator vs `pmh.calibrate` refinements vs paper exemplar | ✓ |
| 3.2 | Recipe cards retitled (subtype primary, T*k secondary) | All four cards + index | ✓ |
| 3.3 | `pmh-train list-methods` prints subtype one-liner | From registry | ✓ |

---

## Phase 4 — Developer API surface

| ID | Deliverable | Acceptance | Status |
|----|-------------|------------|--------|
| 4.1 | `suggest_subtype(**flags) -> SubtypeRecommendation` | Plain English + nuisance string + method | ✓ |
| 4.2 | `print_subtype_guide(method="D4")` | CLI / `python -m pmh.onboarding` | ✓ |
| 4.3 | Export from `import pmh` | Documented in GOLDEN_PATHS | ✓ |

---

## Phase 5 — CI fidelity (synthetic goldens)

| ID | Deliverable | Acceptance | Status |
|----|-------------|------------|--------|
| 5.1 | `tests/test_subtype_fidelity.py` | D1 W vs `numpy_api`; D4 Gram closed form | ✓ |
| 5.2 | D3 synthetic aug modes | Σ rank bounded | ✓ |
| 5.3 | Calibrator smoke: gradient, content-residual, style | Artifact PSD + shape | ✓ |
| 5.4 | Optional: compare to `Paper2/T1/common.py` on fixed seed | ‖W_lib − W_paper‖ / ‖W‖ < tol | ✓ |

---

## Phase 6 — Paper index (research tab only)

| ID | Deliverable | Acceptance | Status |
|----|-------------|------------|--------|
| 6.1 | Table in `PAPER_ALIGNMENT.md`: Block → **primary subtype** → refinement | No block as product name | ✓ |
| 6.2 | `LIBRARY.md` (Paper2) | Points to subtype guide, not block list | ✓ |

---

## Execution order

Work **one phase at a time**; do not start Phase 5 until Phase 1–2 docs and routing match.

```mermaid
flowchart LR
  P0[Phase 0 Plan] --> P1[Phase 1 Subtype guide]
  P1 --> P2[Phase 2 Wizard routing]
  P2 --> P3[Phase 3 Fidelity doc]
  P3 --> P4[Phase 4 API]
  P4 --> P5[Phase 5 CI goldens]
  P5 --> P6[Phase 6 Paper index]
```

---

## Out of scope (explicit)

- Re-implementing full `Paper2/` training trees inside `pmh`  
- Auto-classifying user problems without data flags  
- Promising accuracy wins on every benchmark  

---

## Changelog

| Date | Phase | Notes |
|------|-------|-------|
| 2026-05-19 | 0–1 | Plan + NUISANCE_SUBTYPES.md + nav |
| 2026-05-19 | 2 | `pmh.subtypes`, wizard questionnaire, tests |
| 2026-05-19 | 3–4 | FIDELITY doc, recipe retitles, `suggest_subtype` export |
| 2026-05-19 | 5–6 | `test_subtype_fidelity.py`, PAPER_ALIGNMENT subtype table |
| 2026-05-19 | docs | Nav cleanup, index hub, redirect stubs, `DOCS_GUIDE.md` |
