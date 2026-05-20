"""Namespace subpackages (Phase 2) re-export flat modules."""

from __future__ import annotations


def test_guide_subpackage():
    from pmh.guide import explain_task, robust_fit

    assert "WALKTHROUGH" in explain_task("pose_or_keypoints")
    assert callable(robust_fit)


def test_core_subpackage():
    from pmh.core import PMHConfig, estimate_from_config

    assert PMHConfig.balanced() is not None
    assert callable(estimate_from_config)


def test_train_and_adapt():
    from pmh.adapt import PMHMatcher
    from pmh.train import PMHTrainer, PMHLoss

    assert PMHMatcher and PMHTrainer and PMHLoss


def test_research_lazy_and_direct():
    from pmh import research

    assert hasattr(research, "run_benchmark_protocol")
    assert callable(research.compare_arms)


def test_evidence_subpackage():
    from pmh import evidence

    assert hasattr(evidence, "compare_arms_sklearn")
    assert hasattr(evidence, "validate_falsification")


def test_protocol_subpackage():
    from pmh import protocol

    assert "matched" in protocol.control_modes()
