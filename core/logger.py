"""
VOX Studio Logging & Diagnostics

Tekil kök logger ("vox_studio") + konsol ve döner dosya handler'ları.
Dosya handler'ı oluşturulamazsa uygulama çalışmaya devam eder; yalnızca
konsola uyarı yazılır.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from . import paths

_configured = False
_ROOT_NAME = "vox_studio"

_CONSOLE_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
_FILE_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s"


def _configure() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(logging.DEBUG)
    root.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    try:
        log_dir = paths.logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "vox_studio.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)
    except Exception as exc:  # pragma: no cover - ortama bağlı
        print(f"Uyarı: dosya günlüğü kurulamadı ({exc}); yalnızca konsol kullanılacak.", file=sys.stderr)

    _configured = True


def get_logger(name: str = _ROOT_NAME) -> logging.Logger:
    _configure()

    if not name or name in (_ROOT_NAME, "__main__"):
        return logging.getLogger(_ROOT_NAME)
    if name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def log_directory() -> Optional[Path]:
    return paths.logs_dir()
