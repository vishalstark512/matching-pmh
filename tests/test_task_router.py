"""Task-based routing for developer onboarding."""

from __future__ import annotations

import pytest

from pmh.task_router import explain_task, get_task, list_tasks, route_from_wizard_choice


def test_list_tasks_nonempty():
    tasks = list_tasks()
    assert len(tasks) >= 8
    assert any(t.task_id == "pose_or_keypoints" for t in tasks)


def test_pose_task_mentions_hook_and_task_doc():
    text = explain_task("pose_or_keypoints")
    assert "pose" in text.lower() or "keypoint" in text.lower()
    assert "WHAT CHANGES" in text
    assert "STEPS:" in text
    assert "t03a-pose-gradient" in text
    assert "backbone" in text.lower()
    assert "robust_fit" in text


def test_unknown_task_raises():
    with pytest.raises(KeyError, match="Unknown task"):
        get_task("not_a_real_task")


def test_wizard_choice_maps_to_id():
    assert route_from_wizard_choice("1") == "pose_or_keypoints"
    assert route_from_wizard_choice("pose_or_keypoints") == "pose_or_keypoints"


def test_search_hospital():
    from pmh.task_router import search_applications

    hits = search_applications("hospital")
    ids = {h.task_id for h in hits}
    assert "tabular_same_schema" in ids or "vision_classification" in ids


def test_augmentation_task():
    text = explain_task("augmentation_robustness")
    assert "D3" in text
    assert "augmentation" in text.lower()


def test_cli_route_pose(capsys):
    from pmh.cli.main import main

    assert main(["route", "--task", "pose_or_keypoints", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "WHAT CHANGES" in out
    assert "STEPS:" in out
