#!/usr/bin/env python3
"""Regenerate docs/tasks/index.md only. Task pages + notebooks: render_handcrafted_tasks.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "tasks"


def _render_index() -> str:
    sys.path.insert(0, str(REPO / "src"))
    path = REPO / "scripts" / "render_handcrafted_tasks.py"
    spec = importlib.util.spec_from_file_location("render_handcrafted_tasks", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.render_index()


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "index.md").write_text(_render_index(), encoding="utf-8")
    print("wrote docs/tasks/index.md")


if __name__ == "__main__":
    main()
