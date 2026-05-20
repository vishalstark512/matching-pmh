"""Regenerate docs/NUISANCE_SUBTYPES.md (compact) from catalog + subtypes."""

from pathlib import Path

from pmh.catalog import METHODS
from pmh.subtypes import get_subtype, list_subtypes

ANCHORS = {
    "D1": "d1-cross-domain-subspace",
    "D2": "d2-isotropic",
    "D3": "d3-augmentation-modes",
    "D4": "d4-domain-gram",
    "D5": "d5-compositional",
    "D6": "d6-temporal-sequence",
    "D7": "d7-style-alignment",
}

lines = [
    "# Identify D1–D7 (nuisance families)",
    "",
    "**Step 1 of the [five-step recipe](FIVE_STEP_RECIPE.md).** Most users: "
    "`pmh-train route` or `suggest_subtype` — read this only when you need the mapping table.",
    "",
    "```python",
    "from pmh import suggest_subtype",
    "print(suggest_subtype(has_target_domain=True, has_target_labels=False))",
    "```",
    "",
    "```bash",
    "pmh-train list-methods",
    "pmh-train wizard",
    "```",
    "",
    "Lemma detail: [estimators/index.md](estimators/index.md) · paper blocks: [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md)",
    "",
    "---",
    "",
    "## Quick picker",
    "",
    "| Symptom | Dk | `nuisance` |",
    "|---------|-----|------------|",
    "| Site/camera look, deploy unlabeled OK | D4 | `domain_shift` |",
    "| Same classes, labels on A and B | D1 | `subspace` |",
    "| Known aug modes (blur, crop, …) | D3 | `augmentation` |",
    "| Nuisance only on some indices of h | D5 | `compositional` |",
    "| Drift along time / sequences | D6 | `temporal` |",
    "| LLM format / style pairs | D7 | `style` |",
    "| No preferred direction (noise) | D2 | `isotropic` |",
    "",
    "---",
    "",
]

for method in list_subtypes():
    spec = METHODS[method]
    info = get_subtype(method)
    anchor = ANCHORS[method]
    req = ", ".join(spec.required_data) or "(see config)"
    lines += [
        f"### {method} — {spec.name} {{#{anchor}}}",
        "",
        f"| | |",
        f"|--|--|",
        f"| **Assumption** | {spec.assumption} |",
        f"| **Structure** | {info.similar_structure} |",
        f"| **Nuisance key** | `{info.nuisance}` |",
        f"| **Needs** | {req} |",
        f"| **Mode** | {info.default_mode} |",
        f"| **Exemplars** | {', '.join(info.paper_exemplars)} |",
    ]
    if info.calibrator_note:
        lines += [f"| **Refinement** | {info.calibrator_note} |"]
    lines += ["", "---", ""]

lines += [
    "## Anti-patterns",
    "",
    "| Mistake | Use instead |",
    "|---------|-------------|",
    "| D1 without target labels | D4 |",
    "| D3 for pure site shift | D4 or D1 |",
    "| Matched-only eval | [Falsification](walkthroughs/08-falsification-controls.md) |",
    "",
    "## Next",
    "",
    "[Golden paths](GOLDEN_PATHS.md) · [Integrate](INTEGRATE.md) · `pmh-train route --task …`",
]

out = Path(__file__).resolve().parents[1] / "docs" / "NUISANCE_SUBTYPES.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(lines)} lines)")
