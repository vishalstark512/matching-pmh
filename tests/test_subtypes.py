"""Nuisance subtype registry and wizard routing."""

from __future__ import annotations

from pmh import suggest_subtype
from pmh.subtypes import apply_wizard_subtype_choice, get_subtype, list_subtypes
from pmh.suggest import suggest_nuisance


def test_list_subtypes():
    assert list_subtypes() == [f"D{k}" for k in range(1, 8)]


def test_get_subtype_d4():
    info = get_subtype("D4")
    assert info.nuisance == "domain_shift"
    assert "T4A" in info.paper_exemplars


def test_wizard_choice_d1():
    flags = apply_wizard_subtype_choice("1")
    sug = suggest_nuisance(**flags)
    assert sug.method == "D1"
    assert sug.nuisance == "subspace"


def test_wizard_choice_d3():
    flags = apply_wizard_subtype_choice("3")
    sug = suggest_nuisance(**flags)
    assert sug.method == "D3"


def test_suggest_subtype_export():
    rec = suggest_subtype(has_target_domain=True, has_target_labels=False)
    assert rec.method == "D4"
    assert rec.nuisance == "domain_shift"
    assert "Domain" in rec.title


def test_recommend_setup_carries_lemma():
    from pmh.onboarding import recommend_setup

    rec = recommend_setup(
        stack="pytorch",
        has_augmentation_modes=True,
    )
    assert rec.lemma == "D3"
    assert rec.nuisance == "augmentation"
    assert "docs/tasks/t03a-pose-gradient.md" in rec.subtype_doc
