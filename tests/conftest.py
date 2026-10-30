"""
Test altyapısı.

Tüm testler `VOX_HOME` değişkenini geçici bir dizine yönlendirir; böylece
kullanıcının gerçek ayarları, kayıtları ve ~/.vox_studio içeriği asla
testler tarafından okunmaz veya değiştirilmez.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Test koleksiyonu sırasında core import edilmeden önce güvenli kök
_BOOTSTRAP_HOME = Path(tempfile.mkdtemp(prefix="vox-test-bootstrap-"))
os.environ.setdefault("VOX_HOME", str(_BOOTSTRAP_HOME))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Her test kendi izole uygulama kökünde çalışır."""
    root = tmp_path / "vox_home"
    monkeypatch.setenv("VOX_HOME", str(root))
    yield root


@pytest.fixture
def config(isolated_home):
    from core.config import ConfigManager

    return ConfigManager(root=isolated_home)


@pytest.fixture(scope="session")
def qapp():
    """Qt uygulaması (offscreen). Arayüz testleri için."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
