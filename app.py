"""
VOX Studio — Audio Engine V1 Uygulama Giriş Noktası

Sorumluluklar:
* Qt uygulamasını ve koyu grafit/kırmızı temayı kurmak,
* Kullanıcı veri dizinlerini oluşturmak,
* Yarım kalan kayıtları denetlemek,
* Ana pencereyi açmak.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from audio.recorder import recover_pending_recordings
from core import constants as C
from core import paths
from core.config import ConfigManager
from core.logger import get_logger
from ui.context import AppContext
from ui.icons import app_icon
from ui.main_window import MainWindow
from ui.theme import apply_theme

logger = get_logger("app")


def _pending_parts(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    try:
        return sorted(
            p
            for p in directory.glob("*.part.*")
            if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".ogg", ".aiff"}
        )
    except OSError:
        return []


def _ask_recovery(window_parent, directory: Path, parts: List[Path]) -> None:
    names = "\n".join(f"• {p.name}" for p in parts[:8])
    if len(parts) > 8:
        names += f"\n• ve {len(parts) - 8} dosya daha"
    box = QMessageBox(window_parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Yarım kalan kayıtlar")
    box.setText(
        f"Önceki oturumdan {len(parts)} yarım kayıt dosyası bulundu. "
        "Bunlar beklenmedik bir kapanmadan kalmış olabilir."
    )
    box.setInformativeText(
        f"{names}\n\nKurtarma, okunabilen dosyaları normal kayıt adına taşır. "
        "Okunamayan dosyalara dokunulmaz; elle inceleyebilirsiniz."
    )
    recover_btn = box.addButton("Kurtar", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Şimdi değil", QMessageBox.ButtonRole.RejectRole)
    box.exec()
    if box.clickedButton() is not recover_btn:
        logger.info("Kurtarma kullanıcı tarafından ertelendi (%d dosya)", len(parts))
        return
    try:
        recovered = recover_pending_recordings(directory)
    except Exception as exc:
        logger.exception("Kurtarma başarısız")
        QMessageBox.critical(
            window_parent,
            "Kurtarma başarısız",
            f"Yarım kayıtlar kurtarılırken hata oluştu:\n{exc}\n\n"
            f"Dosyalar şu klasörde duruyor: {directory}",
        )
        return
    if recovered:
        QMessageBox.information(
            window_parent,
            "Kurtarma tamamlandı",
            f"{len(recovered)} kayıt kurtarıldı:\n\n"
            + "\n".join(p.name for p in recovered[:10]),
        )


def _install_excepthook() -> None:
    def handle(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical("Yakalanmamış istisna:\n%s", text)
        try:
            crash_dir = paths.crash_dir()
            crash_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            (crash_dir / f"crash-{stamp}.log").write_text(text, encoding="utf-8")
        except Exception:
            logger.exception("Çökme raporu yazılamadı")

    sys.excepthook = handle


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vox-studio",
        description=f"{C.APP_NAME} — {C.APP_SUBTITLE}",
    )
    parser.add_argument("--version", action="store_true", help="Sürümü yazdır ve çık")
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Başlangıçta yarım kayıt kurtarma sorgusunu atla",
    )
    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="Açılış animasyonunu ve splash ekranını atla",
    )
    parser.add_argument(
        "--open",
        metavar="DOSYA",
        default=None,
        help="Açılışta belirtilen ses dosyasını aç",
    )
    return parser.parse_args(argv)


def run(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.version:
        print(f"{C.APP_NAME} {C.APP_VERSION} — {C.APP_SUBTITLE}")
        return 0

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv[:1] + ([] if argv is None else list(argv)))
    app.setApplicationName(C.APP_NAME)
    app.setApplicationDisplayName(C.APP_NAME)
    app.setApplicationVersion(C.APP_VERSION)
    app.setOrganizationName(C.APP_AUTHOR)
    app.setWindowIcon(app_icon())
    apply_theme(app)

    try:
        paths.ensure_dirs()
    except Exception as exc:
        QMessageBox.critical(
            None,
            "Başlatılamadı",
            f"Uygulama veri dizinleri oluşturulamadı:\n{paths.app_home()}\n\n{exc}",
        )
        return 1

    _install_excepthook()

    try:
        config = ConfigManager.get_instance()
    except Exception as exc:
        logger.exception("Ayar dosyası yüklenemedi")
        config = ConfigManager(auto_load=False)

    try:
        ctx = AppContext(config)
        window = MainWindow(ctx)
    except Exception as exc:
        logger.exception("Ana pencere oluşturulamadı")
        QMessageBox.critical(
            None,
            "Başlatılamadı",
            f"Arayüz başlatılamadı:\n{exc}\n\nAyrıntılar: {paths.logs_dir()}",
        )
        return 1

    if not args.no_recovery:
        parts = _pending_parts(ctx.recordings_dir())
        if parts:
            _ask_recovery(window, ctx.recordings_dir(), parts)
            window.studio.refresh_recent()

    if args.open:
        window.open_file(args.open)

    window.showMaximized()
    logger.info(
        "%s %s başladı — veri kökü: %s", C.APP_NAME, C.APP_VERSION, paths.app_home()
    )
    return app.exec()


def main(argv: Optional[List[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
