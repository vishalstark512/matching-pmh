"""Protocol subpackage (step 4 + control modes)."""

from pmh.protocol import MultiPMHLoss, control_modes, default_protocol_config
from pmh.recipe import default_protocol_config as recipe_default


def test_control_modes():
    modes = control_modes()
    assert "matched" in modes
    assert "wrong_w" in modes
    assert "isotropic" in modes


def test_default_protocol_config():
    cfg = default_protocol_config(preset="balanced")
    assert cfg.cap_ratio > 0
    assert recipe_default(preset="balanced").cap_ratio == cfg.cap_ratio


def test_protocol_import_pmh_config():
    from pmh.protocol import PMHConfig, PMHLoss

    assert PMHConfig.balanced() is not None
    assert PMHLoss is not None
