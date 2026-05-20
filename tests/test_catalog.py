"""Paper task catalog: docs, notebooks, router mapping, paths on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmh.paper_tasks import PAPER_TASK_IDS, get_paper_task, list_paper_tasks
from pmh.task_router import list_tasks

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "tasks"
NB = REPO / "notebooks" / "tasks"

_ROUTER_TO_PAPER: dict[str, str] = {
    "pose_or_keypoints": "t03a-pose-gradient",
    "vision_classification": "t04a-vision-domain",
    "vision_detection": "t04b-multilayer-vision",
    "vision_segmentation": "t04b-multilayer-vision",
    "nlp_text_classification": "t07a-llm-style",
    "llm_style_or_format": "t07a-llm-style",
    "tabular_same_schema": "t01-classical",
    "speech_or_audio": "t06a-speech-whisper",
    "frozen_embeddings_sklearn": "t01-classical",
    "augmentation_robustness": "t03b-depth-augmentation",
    "temporal_drift": "t06b-temporal-har",
    "pytorch_lightning": "t04a-vision-domain",
    "compositional_coordinates": "t05a-qm9-molecule",
    "generic_pytorch": "t04a-vision-domain",
}


@pytest.fixture(scope="module")
def tasks_text() -> dict[str, str]:
    return {
        tid: (DOCS / f"{tid}.md").read_text(encoding="utf-8")
        for tid in PAPER_TASK_IDS
    }


def test_task_ids_match_catalog():
    assert {t.task_id for t in list_paper_tasks()} == set(PAPER_TASK_IDS)


@pytest.mark.parametrize("task_id", PAPER_TASK_IDS)
def test_task_page_exists(task_id: str) -> None:
    assert (DOCS / f"{task_id}.md").is_file()


@pytest.mark.parametrize("task_id", PAPER_TASK_IDS)
def test_task_notebook_exists(task_id: str) -> None:
    assert (NB / f"{task_id}.ipynb").is_file()


@pytest.mark.parametrize("task_id", PAPER_TASK_IDS)
def test_task_notebook_has_pmh_code(task_id: str) -> None:
    nb = json.loads((NB / f"{task_id}.ipynb").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "pmh" in code, task_id


# Keep in sync with scripts/render_handcrafted_tasks.py STANDARD_SECTIONS
_NOTEBOOK_STANDARD_SECTIONS = (
    "## 1 — Install",
    "## 2 — Config & imports",
    "## 3 — Load demo data",
    "## 4 — Scope (applicability)",
    "## 5 — Estimate $\\Sigma_{\\text{task}}$ + PMH train",
    "## 6 — Step 5 (deploy holdout)",
    "## 7 — Paper reproduction",
    "## 8 — Your pipeline",
)


@pytest.mark.parametrize("task_id", [t for t in PAPER_TASK_IDS if t != "t01-classical"])
def test_notebook_standard_sections(task_id: str) -> None:
    standard = _NOTEBOOK_STANDARD_SECTIONS

    nb = json.loads((NB / f"{task_id}.ipynb").read_text(encoding="utf-8"))
    text = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    for header in standard:
        assert header in text, f"{task_id} missing {header}"


@pytest.mark.parametrize("task_id", ["t02a-vit-isotropic", "t02b-chexpert-isotropic"])
def test_t02_notebook_uses_isotropic(task_id: str) -> None:
    nb = json.loads((NB / f"{task_id}.ipynb").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'nuisance="isotropic"' in code
    assert "pytorch_isotropic_demo_loaders" in code


def test_each_paper_task_has_anchor(tasks_text: dict[str, str]):
    for tid in PAPER_TASK_IDS:
        assert f'<a id="{tid}"></a>' in tasks_text[tid]


def test_router_maps_to_paper_page(tasks_text: dict[str, str]):
    for t in list_tasks():
        paper_id = _ROUTER_TO_PAPER.get(t.task_id)
        assert paper_id is not None, t.task_id
        assert f'<a id="{paper_id}"></a>' in tasks_text[paper_id]


def test_task_paths_exist():
    for t in list_paper_tasks():
        assert (REPO / t.notebook).is_file(), t.task_id
        if t.demo_script:
            assert (REPO / t.demo_script).is_file(), t.task_id
    for t in list_tasks():
        assert (REPO / t.example_script).is_file(), t.task_id
    final = REPO / "paper_code/T1/classical_pmh/FINAL.md"
    if final.is_file():
        assert (REPO / "paper_code/T1/classical_pmh/office31_pmh.py").is_file()


@pytest.mark.parametrize("task_id", PAPER_TASK_IDS)
def test_subtask_anchors(task_id: str, tasks_text: dict[str, str]) -> None:
    task = get_paper_task(task_id)
    if not task.subtasks:
        return
    page = tasks_text[task_id]
    for s in task.subtasks:
        assert f'<a id="{s.subtask_id}"></a>' in page, task_id


def test_t01_single_page_no_subfolder_docs():
    assert (REPO / "docs/tasks/t01-classical.md").is_file()
    assert not (REPO / "docs/tasks/t01").exists()


def test_tasks_index_lists_all():
    text = (DOCS / "index.md").read_text(encoding="utf-8")
    for tid in PAPER_TASK_IDS:
        assert tid in text
