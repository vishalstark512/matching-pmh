"""Plain-language application catalog (what nuisance PMH adapts to)."""

from __future__ import annotations

from dataclasses import dataclass

from pmh.task_router import TASK_IDS, TaskRoute, explain_task, get_task, list_tasks


@dataclass(frozen=True)
class ShiftTypePlain:
    """Deploy shift described without paper vocabulary first."""

    id: str
    you_notice: str
    pmh_targets: str
    subtype: str
    nuisance_key: str
    need_data: str
    not_for: str


SHIFT_TYPES: tuple[ShiftTypePlain, ...] = (
    ShiftTypePlain(
        id="site_look",
        you_notice="Images, audio, or rows *look* different (camera, hospital, mic, cohort) but **the label means the same thing**.",
        pmh_targets="Sensitivity along directions that differ between train site A and deploy site B.",
        subtype="D4",
        nuisance_key="domain_shift",
        need_data="Batches from site B (labels optional).",
        not_for="New classes at deploy; label definition changes.",
    ),
    ShiftTypePlain(
        id="class_geometry",
        you_notice="Same disease/classes, and you have **labels on both** hospitals/cohorts — shift is mostly *how* classes appear in feature space.",
        pmh_targets="Low-rank subspace of cross-domain class-conditional differences.",
        subtype="D1",
        nuisance_key="subspace",
        need_data="Labeled source + labeled target (or strong class structure).",
        not_for="Unlabeled target only (use domain shift D4 instead).",
    ),
    ShiftTypePlain(
        id="llm_surface",
        you_notice="LLM outputs must keep the **same facts** but training vs deploy **tone, bullets, markdown, or template** differs.",
        pmh_targets="Directions in hidden states that move with formatting, not content.",
        subtype="D7",
        nuisance_key="style",
        need_data="Style-pair JSONL (same content, two surfaces) or matched corpora.",
        not_for="New knowledge, policy-only changes, factual drift.",
    ),
    ShiftTypePlain(
        id="known_transforms",
        you_notice="You can **name** the deploy sensitivities (blur, crop, color, rotation modes) and generate them.",
        pmh_targets="Gram from augmentation-induced representation deltas.",
        subtype="D3",
        nuisance_key="augmentation",
        need_data="Labeled data + enumerated aug modes.",
        not_for="Unknown site shift with no aug model.",
    ),
    ShiftTypePlain(
        id="coordinate_blocks",
        you_notice="Only **part** of the feature vector can change at deploy (joint indices, token blocks, molecular coords).",
        pmh_targets="Shift along known coordinate indices in h.",
        subtype="D5",
        nuisance_key="compositional",
        need_data="Feature matrix + `nuisance_indices`.",
        not_for="Global camera shift with no index structure.",
    ),
    ShiftTypePlain(
        id="time_drift",
        you_notice="Same patient/sequence label but **measurements drift over time** (longitudinal, sensor aging).",
        pmh_targets="Temporal difference directions in sequence features.",
        subtype="D6",
        nuisance_key="temporal",
        need_data="[N, T, d] sequences per entity.",
        not_for="Independent snapshots with no time axis.",
    ),
    ShiftTypePlain(
        id="isotropic_noise",
        you_notice="Deploy sensitivity has **no preferred direction** (e.g. isotropic sensor / representation noise).",
        pmh_targets="Uniform noise-level geometry in representation space (specialist cases).",
        subtype="D2",
        nuisance_key="isotropic",
        need_data="Known or assumed noise level; feature dim.",
        not_for="Clear site/camera shift (use domain_shift D4 first).",
    ),
)


def explain_nuisance_key(nuisance: str) -> str:
    """Plain English for a ``nuisance=`` API value (no paper vocabulary required)."""
    from pmh.nuisance import resolve_method

    key = nuisance.strip().lower().replace("-", "_")
    for s in SHIFT_TYPES:
        if s.nuisance_key == key:
            return (
                f"nuisance={s.nuisance_key!r} ({s.subtype})\n"
                f"  You notice: {s.you_notice}\n"
                f"  PMH targets: {s.pmh_targets}\n"
                f"  You need: {s.need_data}\n"
                f"  Not for: {s.not_for}"
            )
    method = resolve_method(nuisance)
    return (
        f"nuisance={nuisance!r} maps to estimator {method}.\n"
        f"  See docs/WHAT_IS_DEPLOYMENT_SHIFT.md or pmh-train shifts"
    )


def format_shift_types() -> str:
    lines = [
        "What kind of deploy shift is PMH for?",
        "(You do not need lemma names to decide — match what you notice.)",
        "",
    ]
    for s in SHIFT_TYPES:
        lines.append(f"• {s.you_notice}")
        lines.append(f"    PMH penalizes: {s.pmh_targets}")
        lines.append(f"    Usually: {s.subtype} ({s.nuisance_key}) · Need: {s.need_data}")
        lines.append(f"    Not for: {s.not_for}")
        lines.append("")
    return "\n".join(lines)


def format_application_finder() -> str:
    lines = [
        "Find your application (then run route for the full walkthrough):",
        "",
        f"{'#':<3} {'Application':<42} {'What changes':<28} {'Route'}",
        "-" * 95,
    ]
    for i, r in enumerate(list_tasks(), 1):
        tag = {"use_pmh": "YES", "maybe": "TRY", "skip_pmh": "NO"}[r.verdict]
        what = r.what_changes[:26] + ("…" if len(r.what_changes) > 27 else "")
        lines.append(f"{i:<3} {r.title[:40]:<42} {what:<28} {tag}")
        lines.append(f"    pmh-train route --task {r.task_id}")
    lines.append("")
    lines.append("Full walkthroughs: docs/APPLICATIONS.md")
    return "\n".join(lines)


def explain_application(task_id: str) -> str:
    """Extended walkthrough (CLI + notebooks)."""
    return explain_task(task_id)


def search_applications(query: str) -> list[TaskRoute]:
    from pmh.task_router import search_applications as _search

    return _search(query)


def format_search_results(query: str) -> str:
    from pmh.task_router import format_search_results as _fmt

    return _fmt(query)


__all__ = [
    "SHIFT_TYPES",
    "ShiftTypePlain",
    "explain_application",
    "format_application_finder",
    "format_search_results",
    "format_shift_types",
    "explain_nuisance_key",
    "search_applications",
    "get_task",
    "list_tasks",
]
