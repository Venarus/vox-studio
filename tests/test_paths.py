"""core.paths ve core.constants testleri."""

from __future__ import annotations

import os
from pathlib import Path

from core import paths


def test_app_home_defaults_to_user_profile(monkeypatch):
    monkeypatch.delenv(paths.ENV_HOME, raising=False)
    assert paths.app_home() == Path.home() / ".vox_studio"


def test_app_home_respects_env(isolated_home):
    assert paths.app_home() == isolated_home


def test_ensure_dirs_creates_expected_tree(isolated_home):
    created = paths.ensure_dirs()
    for key in ("projects", "recordings", "models", "cache", "logs", "backups", "exports", "crash"):
        path = created[key]
        assert path.is_dir()
        assert path.parent == isolated_home
    assert paths.ensure_dirs()["recordings"] == created["recordings"]  # idempotent


def test_no_directories_created_on_import(isolated_home):
    """constants/paths import edildiğinde kullanıcı dizininde yan etki olmamalı."""
    import importlib

    import core.constants as constants

    importlib.reload(constants)
    assert not isolated_home.exists() or not any(isolated_home.iterdir())


def test_constants_legacy_aliases_exist():
    from core import constants

    assert constants.APP_NAME == "VOX Studio"
    assert constants.DEFAULT_SAMPLE_RATE == 48000
    assert "PCM_24" in constants.SUBTYPE_LABELS
    assert constants.COLOR_ACCENT.startswith("#")
