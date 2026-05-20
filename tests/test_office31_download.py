"""Office-31 layout helpers (no network in default tests)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pmh.datasets.office31 import (
    DOMAIN_NAMES,
    download_office31,
    verify_office31_layout,
)


def _fake_domain_tree(root: Path, domain: str) -> None:
    d = root / domain / "class0"
    d.mkdir(parents=True)
    (d / "img.jpg").write_bytes(b"\xff\xd8\xff")


def test_verify_office31_layout_ok(tmp_path: Path) -> None:
    for dom in DOMAIN_NAMES:
        _fake_domain_tree(tmp_path, dom)
    verify_office31_layout(tmp_path)


def test_verify_office31_layout_missing(tmp_path: Path) -> None:
    _fake_domain_tree(tmp_path, "amazon")
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_office31_layout(tmp_path)


def test_download_skips_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for dom in DOMAIN_NAMES:
        _fake_domain_tree(tmp_path, dom)

    called = {"n": 0}

    def _no_download(*_a, **_k):
        called["n"] += 1

    monkeypatch.setattr("urllib.request.urlretrieve", _no_download)
    download_office31(tmp_path, force=False)
    assert called["n"] == 0
