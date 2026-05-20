"""Tier-0 public API contract."""

from __future__ import annotations

import pmh
from pmh._api import TIER_0, TIER_1, list_public_names, tier_of


def test_tier0_in_all():
    assert set(pmh.__all__) == set(TIER_0)


def test_tier0_importable():
    for name in TIER_0:
        if name == "__version__":
            assert pmh.__version__
            continue
        assert hasattr(pmh, name), name
        assert tier_of(name) == 0


def test_tier1_still_importable_from_pmh():
    for name in ("PMHLoss", "estimate_from_config", "compare_arms_sklearn"):
        assert hasattr(pmh, name)
        assert tier_of(name) == 1


def test_list_public_names():
    assert "robust_fit" in list_public_names(tier=0)
    assert "PMHLoss" in list_public_names(tier=1)
