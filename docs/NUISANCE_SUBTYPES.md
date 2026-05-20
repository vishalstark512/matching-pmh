# Shift types D1–D7 (appendix)

**Step 1 technical appendix.** Start with [What is deployment shift?](WHAT_IS_DEPLOYMENT_SHIFT.md) — you do **not** need the word “nuisance” or lemma names to integrate.

Most users: `pmh-train route`, `pmh-train shifts`, or `suggest_nuisance` — open this page only for the full D1–D7 table.

```python
from pmh import suggest_subtype
print(suggest_subtype(has_target_domain=True, has_target_labels=False))
```

```bash
pmh-train list-methods
pmh-train wizard
```

Lemma detail: [estimators/index.md](estimators/index.md) · paper blocks: [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md)

---

## Quick picker

| Symptom | Dk | `nuisance` |
|---------|-----|------------|
| Site/camera look, deploy unlabeled OK | D4 | `domain_shift` |
| Same classes, labels on A and B | D1 | `subspace` |
| Known aug modes (blur, crop, …) | D3 | `augmentation` |
| Nuisance only on some indices of h | D5 | `compositional` |
| Drift along time / sequences | D6 | `temporal` |
| LLM format / style pairs | D7 | `style` |
| No preferred direction (noise) | D2 | `isotropic` |

---

### D1 — Subspace (cross-domain SVD) {#d1-cross-domain-subspace}

| | |
|--|--|
| **Assumption** | A1: low-rank subspace W |
| **Structure** | Same classes; different domains shift h along a low-rank subspace. |
| **Nuisance key** | `subspace` |
| **Needs** | source_features, source_labels, target_features, target_labels |
| **Mode** | either |
| **Exemplars** | T1 |

---

### D2 — Isotropic noise {#d2-isotropic}

| | |
|--|--|
| **Assumption** | A2: N(0, sigma^2 I) |
| **Structure** | Uniform σ²I nuisance; no learned directions. |
| **Nuisance key** | `isotropic` |
| **Needs** | (see config) |
| **Mode** | jacobian |
| **Exemplars** | T2A, T2B |

---

### D3 — Augmentation modes {#d3-augmentation-modes}

| | |
|--|--|
| **Assumption** | A3: finite aug coefficients |
| **Structure** | Finite known transforms; Σ from aug-induced deltas. |
| **Nuisance key** | `augmentation` |
| **Needs** | aug_deltas |
| **Mode** | jacobian |
| **Exemplars** | T3A, T3B |
| **Refinement** | Gradient-SVD: pmh.calibrate.gradient_subspace_numpy |

---

### D4 — Domain Gram {#d4-domain-gram}

| | |
|--|--|
| **Assumption** | A4: paired domain shift |
| **Structure** | Unlabeled domain difference; pooled source−target Gram. |
| **Nuisance key** | `domain_shift` |
| **Needs** | source_features, target_features |
| **Mode** | jacobian |
| **Exemplars** | T4A, T4B |

---

### D5 — Compositional block {#d5-compositional}

| | |
|--|--|
| **Assumption** | A5: nuisance coordinates |
| **Structure** | Nuisance on named coordinates of h (positions, tokens, nodes). |
| **Nuisance key** | `compositional` |
| **Needs** | features |
| **Mode** | jacobian |
| **Exemplars** | T5A, T5B |

---

### D6 — Temporal residual {#d6-temporal-sequence}

| | |
|--|--|
| **Assumption** | A6: label-constant drift |
| **Structure** | Label-constant drift along time or sensor trajectories. |
| **Nuisance key** | `temporal` |
| **Needs** | sequences |
| **Mode** | jacobian |
| **Exemplars** | T6A, T6B |
| **Refinement** | Content-residual: pmh.calibrate.content_residual_subspace |

---

### D7 — Style / alignment Gram {#d7-style-alignment}

| | |
|--|--|
| **Assumption** | A7: style pairs or PGD deltas |
| **Structure** | Same content, different surface form (format, tone, PGD δ). |
| **Nuisance key** | `style` |
| **Needs** | style_jsonl |
| **Mode** | jacobian |
| **Exemplars** | T7A, T7B |
| **Refinement** | PGD deltas: pmh.calibrate.subspace_artifact_from_deltas |

---

## Anti-patterns

| Mistake | Use instead |
|---------|-------------|
| D1 without target labels | D4 |
| D3 for pure site shift | D4 or D1 |
| Matched-only eval | [Falsification](walkthroughs/08-falsification-controls.md) |

## Next

[Golden paths](GOLDEN_PATHS.md) · [Integrate](INTEGRATE.md) · `pmh-train route --task …`
