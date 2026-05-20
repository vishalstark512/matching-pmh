"""HTML reports: deploy Step 5 + synthesized paper findings."""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

from pmh.adoption import ARM_PLAIN_NAMES, FALSIFICATION_ARM_ORDER
from pmh.paper_findings import list_paper_findings, synthesis_paragraphs

if TYPE_CHECKING:
    from pmh.developer import EvaluationReport

_CSS = """
:root {
  --bg: #f8f9fb;
  --card: #fff;
  --text: #1a1d26;
  --muted: #5c6370;
  --accent: #2563eb;
  --pass: #059669;
  --fail: #dc2626;
  --warn: #d97706;
  --border: #e2e8f0;
}
* { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.55;
  margin: 0;
  padding: 1.5rem;
  max-width: 52rem;
  margin-left: auto;
  margin-right: auto;
}
h1 { font-size: 1.65rem; margin: 0 0 0.5rem; }
h2 { font-size: 1.2rem; margin: 2rem 0 0.75rem; border-bottom: 1px solid var(--border); padding-bottom: 0.35rem; }
.meta { color: var(--muted); font-size: 0.92rem; margin-bottom: 1.5rem; }
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.25rem 1.5rem;
  margin: 1rem 0;
  box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
.badge {
  display: inline-block;
  font-size: 0.8rem;
  font-weight: 600;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  margin: 0.25rem 0;
}
.badge-pass { background: #d1fae5; color: #065f46; }
.badge-fail { background: #fee2e2; color: #991b1b; }
.badge-warn { background: #fef3c7; color: #92400e; }
.badge-partial { background: #e0e7ff; color: #3730a3; }
table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
th, td { text-align: left; padding: 0.55rem 0.65rem; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.03em; }
.metric-row td:first-child { color: var(--muted); width: 42%; }
.bar-wrap { background: #eef2f7; border-radius: 4px; height: 8px; margin-top: 4px; overflow: hidden; }
.bar { height: 100%; background: var(--accent); border-radius: 4px; }
.verdict { font-size: 1.05rem; font-weight: 600; margin-top: 1rem; }
.note {
  background: #fffbeb;
  border-left: 4px solid var(--warn);
  padding: 0.85rem 1rem;
  margin: 1.25rem 0;
  font-size: 0.92rem;
}
.note-lib {
  background: #eff6ff;
  border-left: 4px solid var(--accent);
}
ul { padding-left: 1.2rem; }
a { color: var(--accent); }
footer { margin-top: 2.5rem; color: var(--muted); font-size: 0.85rem; }
"""


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
{body}
<footer>matching-pmh · generated report</footer>
</body>
</html>"""


def _badge_for_verdict(verdict: str) -> str:
    v = verdict.upper()
    if "PASS" in v or "SHIP" in v:
        cls = "badge-pass"
    elif "FAIL" in v or "DO NOT" in v:
        cls = "badge-fail"
    elif "INCONCLUSIVE" in v:
        cls = "badge-warn"
    else:
        cls = "badge-warn"
    return f'<span class="badge {cls}">{html.escape(verdict)}</span>'


def _status_badge(status: str) -> str:
    m = {
        "pass": ("Pass", "badge-pass"),
        "partial": ("Partial / honest limit", "badge-partial"),
        "documented_failure": ("Documented limit", "badge-partial"),
    }
    label, cls = m.get(status, (status, "badge-warn"))
    return f'<span class="badge {cls}">{html.escape(label)}</span>'


def evaluation_report_html(
    report: EvaluationReport,
    *,
    title: str = "PMH deploy report",
    subtitle: str | None = None,
) -> str:
    """HTML one-pager for :class:`~pmh.developer.EvaluationReport`."""
    arms = report.falsification_arms
    max_val = max(
        [report.baseline_metric, report.pmh_metric, *arms.values()],
        default=1e-6,
    )
    rows = [
        ("ERM baseline (no PMH)", report.baseline_metric),
        ("Shift-matched PMH", report.pmh_metric),
    ]
    for key in FALSIFICATION_ARM_ORDER:
        if key in arms and key not in ("b0", "matched"):
            rows.append((ARM_PLAIN_NAMES.get(key, key), arms[key]))

    table_rows = []
    for label, val in rows:
        pct = 100.0 * val / max_val if max_val > 0 else 0
        table_rows.append(
            f"<tr class='metric-row'><td>{html.escape(label)}</td>"
            f"<td>{val:.4f} {html.escape(report.metric_name)}"
            f"<div class='bar-wrap'><div class='bar' style='width:{pct:.1f}%'></div></div></td></tr>"
        )

    preflight = ""
    if report.preflight:
        preflight = (
            f"<p><strong>Geometry check:</strong> {html.escape(str(report.preflight))} — "
            f"{html.escape(report.preflight_message)}</p>"
        )

    notes = "".join(f"<li>{html.escape(n)}</li>" for n in report.notes if n)

    body = f"""
<h1>{html.escape(title)}</h1>
<p class="meta">{html.escape(subtitle or "Deploy holdout · Step 5 falsification arms")}</p>
<div class="card">
  <table>
    <thead><tr><th>Arm</th><th>{html.escape(report.metric_name)}</th></tr></thead>
    <tbody>{"".join(table_rows)}</tbody>
  </table>
  {preflight}
  <p class="verdict">{_badge_for_verdict(report.ship_verdict())}</p>
</div>
"""
    if notes:
        body += f"<div class='card'><ul>{notes}</ul></div>"
    return _page(title, body)


def paper_findings_html(*, title: str = "PMH paper — block findings") -> str:
    """Synthesized outcomes from ``paper_code`` (not library demo numbers)."""
    blocks = list_paper_findings()
    n_pass = sum(1 for b in blocks if b.status == "pass")
    n_partial = sum(1 for b in blocks if b.status != "pass")

    rows = []
    for b in blocks:
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(b.block)}</strong><br/>"
            f"<span style='color:var(--muted);font-size:0.88rem'>{html.escape(b.task_id)}</span></td>"
            f"<td>{html.escape(b.title)}<br/>Lemma {html.escape(b.lemma)} · {html.escape(b.stack)}</td>"
            f"<td>{html.escape(b.headline)}</td>"
            f"<td>{_status_badge(b.status)}</td>"
            f"<td><code style='font-size:0.8rem'>{html.escape(b.final_path)}</code></td>"
            "</tr>"
        )

    synthesis = "".join(f"<p>{html.escape(p)}</p>" for p in synthesis_paragraphs())

    body = f"""
<h1>{html.escape(title)}</h1>
<p class="meta">Synthesized from <code>paper_code/T*/**/FINAL.md</code> · Train on A, deploy on B, same labels</p>

<div class="note note-lib">
  <strong>Library vs paper:</strong> These numbers come from block-specific reproduction code in
  <code>paper_code/</code>, not from <code>pip install matching-pmh</code> on demo loaders.
  The library implements the same recipe for <em>your</em> stack; expect iteration until Step 5 passes
  on your deploy holdout. See <a href="https://github.com/vishalstark512/matching-pmh/blob/main/docs/START.md">docs/START.md</a>.
</div>

<div class="card">
  <p><strong>Summary:</strong> <span class="badge badge-pass">{n_pass} blocks pass</span>
  <span class="badge badge-partial">{n_partial} partial / limit</span> (pre-registered criteria)</p>
  {synthesis}
</div>

<h2>Thirteen blocks (T1–T7)</h2>
<div class="card" style="overflow-x:auto">
  <table>
    <thead>
      <tr><th>Block</th><th>Task</th><th>Headline result (paper code)</th><th>Status</th><th>FINAL.md</th></tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>

<h2>Use the library</h2>
<div class="card">
  <ul>
    <li><strong>Golden path:</strong> <code>pmh-train try --quick</code> or <code>try_pmh(...)</code> → ship verdict</li>
    <li><strong>Your metrics:</strong> <code>report.save_html("deploy_report.html")</code> after <code>evaluate_robust_fit</code></li>
    <li><strong>Reproduce a block:</strong> run scripts under <code>paper_code/</code> for that task's FINAL.md</li>
  </ul>
</div>
"""
    return _page(title, body)


def write_html(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def save_evaluation_report_html(report: EvaluationReport, path: str | Path, **kwargs) -> Path:
    return write_html(path, evaluation_report_html(report, **kwargs))


def save_paper_findings_html(path: str | Path) -> Path:
    return write_html(path, paper_findings_html())
