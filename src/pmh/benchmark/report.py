"""Markdown/JSON reports for multi-arm training comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pmh.benchmark.arms import ARM_SPECS, STANDARD_ARMS, normalize_arm
from pmh.benchmark.protocol import BenchmarkResult


def _fmt(x: float | None, *, digits: int = 4) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def benchmark_to_markdown(
    result: BenchmarkResult | dict[str, Any],
    *,
    title: str = "PMH arm comparison (your pipeline)",
) -> str:
    """Human-readable comparison table."""
    if isinstance(result, BenchmarkResult):
        data = result.to_dict()
    else:
        data = result

    arms_data = data.get("arms", {})
    art = data.get("artifact", {})
    lines = [
        f"# {title}",
        "",
        f"Estimator: **{art.get('method', '?')}** · preflight: `{art.get('preflight')}` · "
        f"eigengap: {_fmt(art.get('eigengap'))}",
        "",
        "| Arm | Label | val metric | task loss | PMH loss |",
        "|-----|-------|------------|-----------|----------|",
    ]
    order = [normalize_arm(a) for a in STANDARD_ARMS if normalize_arm(a) in arms_data]
    for a in arms_data:
        if normalize_arm(a) not in order:
            order.append(normalize_arm(a))

    for arm in order:
        row = arms_data.get(arm, {})
        spec = ARM_SPECS.get(arm)  # type: ignore[arg-type]
        label = spec.label if spec else arm
        final = row.get("final") or {}
        lines.append(
            f"| `{arm}` | {label} | {_fmt(row.get('val_metric'))} | "
            f"{_fmt(final.get('task_loss'))} | {_fmt(final.get('pmh_loss'))} |"
        )

    lines.extend(["", "## How to read", ""])
    lines.append(
        "- **Matched** should beat **B0** on deployment-relevant `val_metric` when the nuisance story is correct."
    )
    lines.append("- **Wrong-W** and **isotropic** should not beat **matched** (Lemma C).")
    lines.append("- If only matched wins vs B0 but wrong-W also wins, the gain may be generic regularization.")

    notes = data.get("notes") or []
    if notes:
        lines.extend(["", "## Notes", ""])
        for n in notes:
            lines.append(f"- {n}")

    return "\n".join(lines) + "\n"


def write_benchmark_report(
    result: BenchmarkResult,
    output_dir: str | Path,
    *,
    stem: str = "benchmark",
) -> dict[str, Path]:
    """Write ``{stem}.json`` and ``{stem}.md`` under ``output_dir``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    payload = result.to_dict()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(benchmark_to_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def load_benchmark_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
