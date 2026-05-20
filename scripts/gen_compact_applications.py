"""Regenerate docs/APPLICATIONS.md (compact) from task_router."""

from pathlib import Path

from pmh.task_router import list_tasks

verdict_tag = {"use_pmh": "YES", "maybe": "TRY", "skip_pmh": "NO"}

lines = [
    "# Find your application",
    "",
    "**CLI is the detailed walkthrough** — this page is a finder + anchors. "
    "Spine: [FIVE_STEP_RECIPE](FIVE_STEP_RECIPE.md) · code: [GOLDEN_PATHS](GOLDEN_PATHS.md).",
    "",
    "```bash",
    "pmh-train route --search hospital",
    "pmh-train route --task pose_or_keypoints",
    "```",
    "",
    "```python",
    "from pmh import explain_task",
    'print(explain_task("pose_or_keypoints"))',
    "```",
    "",
    "---",
    "",
    "## Finder",
    "",
    '<a id="application-finder"></a>',
    "",
    "| Application | What changes | Fit | Details |",
    "|-------------|--------------|-----|---------|",
]
for t in list_tasks():
    tag = verdict_tag[t.verdict]
    short = t.what_changes[:52] + ("…" if len(t.what_changes) > 52 else "")
    lines.append(f"| {t.title} | {short} | **{tag}** | [↓](#{t.task_id}) |")

lines += [
    "",
    "**YES** = usual fit · **TRY** = validate on deploy metric first.",
    "",
    "---",
    "",
]

for t in list_tasks():
    tag = verdict_tag[t.verdict]
    gpath = "GOLDEN_PATHS.md"
    if "#g1" in t.golden_path.lower():
        gpath += "#g1"
    elif "#g1b" in t.golden_path.lower():
        gpath += "#g1b"
    elif "#g2" in t.golden_path.lower():
        gpath += "#g2"
    elif "#g3b" in t.golden_path.lower():
        gpath += "#g3b"
    elif "#g3" in t.golden_path.lower():
        gpath += "#g3"
    elif "#g4" in t.golden_path.lower():
        gpath += "#g4"
    lines += [
        f'<a id="{t.task_id}"></a>',
        "",
        f"## {t.title}",
        "",
        f"**Fit:** {tag} — {t.verdict_summary}",
        "",
        "| | |",
        "|--|--|",
        f"| **What changes** | {t.what_changes} |",
        f"| **Mapping** | {t.lemma} · `nuisance={t.nuisance!r}` |",
        f"| **Golden path** | [{t.golden_path}]({gpath}) |",
        f"| **Data** | {t.data_you_need} |",
        f"| **Not for** | {t.not_for} |",
        "",
        f"**CLI:** `pmh-train route --task {t.task_id}` · **Example:** `{t.example_script}`",
        "",
        "---",
        "",
    ]

lines += [
    "## Not PMH",
    "",
    "New classes at deploy or unrelated labels → [WHEN_PMH_HELPS](WHEN_PMH_HELPS.md).",
    "",
    "D1–D7 detail: [estimators/index.md](estimators/index.md) (when identification step requires it).",
]

out = Path(__file__).resolve().parents[1] / "docs" / "APPLICATIONS.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(lines)} lines)")
