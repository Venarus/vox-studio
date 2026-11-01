"""
VOX Studio — Otomatik Gerçek Arayüz Ekran Görüntüsü Yakalama Aracı
(docs/screenshots/ ve docs/assets/ oluşturur)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless / offscreen Qt rendering
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from core import constants as C
from core import paths
from core.config import ConfigManager
from ui.context import AppContext
from ui.main_window import MainWindow
from ui.theme import apply_theme


def capture_all():
    docs_dir = root / "docs"
    screenshots_dir = docs_dir / "screenshots"
    assets_dir = docs_dir / "assets"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    apply_theme(app)

    cfg = ConfigManager(auto_load=False)
    ctx = AppContext(cfg)
    win = MainWindow(ctx)
    win.resize(1366, 850)
    win.show()
    app.processEvents()

    # Örnek ses dosyasını yükle
    sample_wav = root / "benchmarks_output" / "01_raw_recording_48k_24bit.wav"
    if sample_wav.exists():
        win.open_file(str(sample_wav))
        app.processEvents()

    # 1. Recording Studio Ekran Görüntüsü
    win._show_view("studio")
    app.processEvents()
    p1 = screenshots_dir / "01_recording_studio.png"
    win.grab().save(str(p1))
    print(f"[OK] Studio ekran görüntüsü kaydedildi: {p1}")

    # 2. Restoration Workstation (Simple Mode)
    if sample_wav.exists():
        win.processing.load_file(sample_wav)
    win._show_view("processing")
    win.processing._set_mode(simple=True)
    app.processEvents()
    p2 = screenshots_dir / "02_restoration_workstation.png"
    win.grab().save(str(p2))
    print(f"[OK] Restoration Workstation ekran görüntüsü kaydedildi: {p2}")

    # 3. Restoration Workstation (Pro Mode DSP Rack)
    win.processing._set_mode(simple=False)
    app.processEvents()
    p3 = screenshots_dir / "03_pro_dsp_rack.png"
    win.grab().save(str(p3))
    print(f"[OK] Pro DSP Rack ekran görüntüsü kaydedildi: {p3}")

    # 4. Take Library View
    win._show_view("library")
    app.processEvents()
    p4 = screenshots_dir / "04_audio_library.png"
    win.grab().save(str(p4))
    print(f"[OK] Library ekran görüntüsü kaydedildi: {p4}")

    # 5. Project Session Manager
    win._show_view("projects")
    app.processEvents()
    p5 = screenshots_dir / "05_project_manager.png"
    win.grab().save(str(p5))
    print(f"[OK] Project Manager ekran görüntüsü kaydedildi: {p5}")

    print("Tüm ekran görüntüleri başarıyla alındı!")


if __name__ == "__main__":
    capture_all()
