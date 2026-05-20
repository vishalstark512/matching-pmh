#!/usr/bin/env python3
"""Write docs/findings.html — synthesized paper block outcomes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pmh.report_html import save_paper_findings_html  # noqa: E402


def main() -> int:
    out = ROOT / "docs" / "findings.html"
    save_paper_findings_html(out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
