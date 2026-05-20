"""HTML deploy + paper findings reports."""

from __future__ import annotations

from pathlib import Path

from pmh.developer import EvaluationReport
from pmh.paper_findings import list_paper_findings, synthesis_paragraphs
from pmh.report_html import (
    evaluation_report_html,
    paper_findings_html,
    save_paper_findings_html,
)


def test_evaluation_report_html_contains_verdict_and_arms():
    report = EvaluationReport(
        baseline_metric=0.5,
        pmh_metric=0.7,
        metric_name="accuracy",
        falsification_arms={
            "b0": 0.5,
            "matched": 0.7,
            "wrong_w": 0.52,
            "isotropic": 0.51,
        },
        notes=["test note"],
    )
    html = report.to_html(title="Test deploy")
    assert "Test deploy" in html
    assert "PASS" in html or "FAIL" in html or "INCONCLUSIVE" in html
    assert "0.7000" in html or "0.7" in html
    assert "test note" in html


def test_evaluation_report_escapes_html_in_notes():
    report = EvaluationReport(
        baseline_metric=0.1,
        pmh_metric=0.2,
        metric_name="acc",
        falsification_arms={"b0": 0.1, "matched": 0.2},
        notes=["<script>alert(1)</script>"],
    )
    assert "<script>" not in report.to_html()
    assert "&lt;script&gt;" in report.to_html()


def test_save_html_tmp(tmp_path: Path):
    report = EvaluationReport(
        baseline_metric=0.4,
        pmh_metric=0.6,
        metric_name="f1",
        falsification_arms={"b0": 0.4, "matched": 0.6, "wrong_w": 0.41},
    )
    out = report.save_html(tmp_path / "r.html")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_paper_findings_html_has_blocks_and_disclaimer():
    html = paper_findings_html()
    assert "Library vs paper" in html
    assert "t04b-multilayer-vision" in html
    assert "Office-31" in html or "t01-classical" in html
    findings = list_paper_findings()
    assert len(findings) == 13
    assert any(b.status == "partial" for b in findings)
    assert len(synthesis_paragraphs()) >= 3


def test_save_paper_findings(tmp_path: Path):
    p = save_paper_findings_html(tmp_path / "findings.html")
    assert "12 of 13" in p.read_text(encoding="utf-8")
