"""Keep task_router and docs/APPLICATIONS.md aligned."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pmh.task_router import TASK_IDS, list_tasks

REPO = Path(__file__).resolve().parents[1]
APPLICATIONS = REPO / "docs" / "APPLICATIONS.md"
GOLDEN_PATHS = REPO / "docs" / "GOLDEN_PATHS.md"


@pytest.fixture(scope="module")
def applications_text() -> str:
    assert APPLICATIONS.is_file(), f"missing {APPLICATIONS}"
    return APPLICATIONS.read_text(encoding="utf-8")


def test_task_ids_match_catalog():
    tasks = list_tasks()
    assert {t.task_id for t in tasks} == set(TASK_IDS)


def test_each_task_has_html_anchor(applications_text: str):
    for tid in TASK_IDS:
        assert f'<a id="{tid}"></a>' in applications_text, f"missing anchor for {tid}"


def test_doc_one_pager_points_to_applications_anchor(applications_text: str):
    for t in list_tasks():
        assert t.doc_one_pager.startswith("docs/APPLICATIONS.md#")
        anchor = t.doc_one_pager.split("#", 1)[1]
        assert anchor == t.task_id
        assert f'<a id="{anchor}"></a>' in applications_text


def test_application_finder_table_links(applications_text: str):
    for t in list_tasks():
        assert f"](#{t.task_id})" in applications_text, f"finder table missing link for {t.task_id}"


def test_example_scripts_exist():
    for t in list_tasks():
        path = REPO / t.example_script
        assert path.is_file(), f"{t.task_id}: missing {t.example_script}"


def test_golden_path_anchors_exist():
    text = GOLDEN_PATHS.read_text(encoding="utf-8")
    for t in list_tasks():
        m = re.search(r"G\d+b?", t.golden_path)
        assert m is not None, f"{t.task_id}: no G-path in golden_path={t.golden_path!r}"
        gid = m.group(0).lower()
        assert f'<a id="{gid}"></a>' in text or f'id="{gid}"' in text, (
            f"{t.task_id}: GOLDEN_PATHS missing anchor {gid}"
        )


def test_explain_task_mentions_walkthrough_and_golden_path():
    from pmh.task_router import explain_task

    for tid in TASK_IDS:
        text = explain_task(tid)
        assert "WALKTHROUGH" in text
        assert "G" in text  # golden path line
