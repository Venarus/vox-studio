"""
VOX Studio Configuration Manager

JSON tabanlı, atomik yazımlı ve iş parçacığı güvenli yapılandırma yöneticisi.

Özellikler
----------
* Yazma işlemi geçici dosya + ``os.replace`` ile atomiktir; yazma sırasında
  oluşan çökme mevcut ayarları bozmaz.
* Bozuk JSON dosyası sessizce silinmez; ``config.json.corrupt-<zaman>``
  olarak saklanır ve varsayılanlarla devam edilir.
* ``VOX_HOME`` ortam değişkeni ile kök dizin değiştirilebilir (testler).
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from . import paths
from .constants import (
    DEFAULT_BIT_DEPTH,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CHANNELS,
    DEFAULT_RECORDING_SUBTYPE,
    DEFAULT_SAMPLE_RATE,
)
from .logger import get_logger

logger = get_logger("config")


def default_recording_dir() -> str:
    return str(paths.recordings_dir())


DEFAULT_CONFIG: Dict[str, Any] = {
    # Genel
    "language": "tr",  # geriye dönük uyumluluk
    "ui_language": "tr",  # 30 dilden biri (ör. tr, en, de, ar, ja, zh-CN...)
    "transcription_language": "auto",  # auto | 30 dilden biri
    "theme": "dark_crimson",
    "app_version": "1.0.0",
    # Ses cihazları (index + hostapi birlikte saklanır; index tek başına kalıcı değildir)
    "input_device_id": None,
    "input_device_name": "",
    "input_hostapi": None,
    "output_device_id": None,
    "output_device_name": "",
    "output_hostapi": None,
    # Kayıt ayarları
    "sample_rate": DEFAULT_SAMPLE_RATE,
    "channels": DEFAULT_CHANNELS,
    "block_size": DEFAULT_BLOCK_SIZE,
    "bit_depth": DEFAULT_BIT_DEPTH,
    "recording_subtype": DEFAULT_RECORDING_SUBTYPE,
    "recordings_dir": "",  # boş -> paths.recordings_dir()
    "countdown_sec": 0,
    # Proje / oturum
    "auto_save_interval_sec": 60,
    "last_project_path": "",
    "recent_projects": [],
    "recent_recordings": [],
    # AI
    "ai_model_mode": "balanced",  # fast | balanced | quality | advanced
    "whisper_model": "base",
    "cloud_ai_enabled": False,
    # Stem
    "stem_export_format": "wav",
    "stem_naming_template": "{project}_{stem}",
}


class ConfigManager:
    """Tekil (singleton) ayar yöneticisi.

    Not: ``get_instance()`` yalnızca varsayılan kök dizin için tekil örnek
    döndürür. Testler doğrudan ``ConfigManager(root=...)`` oluşturmalıdır.
    """

    _instance: Optional["ConfigManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self, root: Optional[Path] = None, auto_load: bool = True):
        self._lock = threading.RLock()
        self._root = Path(root) if root is not None else paths.app_home()
        self.config_path = self._root / "config.json"
        self._defaults: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
        self._data: Dict[str, Any] = copy.deepcopy(self._defaults)
        self.last_load_error: Optional[str] = None
        if auto_load:
            self.load()

    # ------------------------------------------------------------------ tekil
    @classmethod
    def get_instance(cls) -> "ConfigManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ConfigManager()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Tekil örneği bırakır (testler ve yeniden başlatma simülasyonu)."""
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------ disk
    def load(self) -> None:
        with self._lock:
            if not self.config_path.exists():
                self.save()
                return
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if not isinstance(saved, dict):
                    raise ValueError("kök nesne bir JSON sözlüğü değil")
            except Exception as exc:  # bozuk dosya: silme, yedekle
                self.last_load_error = str(exc)
                logger.error("Ayar dosyası okunamadı (%s): %s", self.config_path, exc)
                self._quarantine_corrupt_file()
                self._data = copy.deepcopy(self._defaults)
                self.save()
                return

            merged = copy.deepcopy(self._defaults)
            merged.update(saved)
            self._data = merged
            self.last_load_error = None
            logger.info("Ayar dosyası yüklendi: %s", self.config_path)

    def _quarantine_corrupt_file(self) -> None:
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            target = self.config_path.with_name(f"{self.config_path.name}.corrupt-{stamp}")
            os.replace(self.config_path, target)
            logger.warning("Bozuk ayar dosyası saklandı: %s", target)
        except Exception as exc:
            logger.error("Bozuk ayar dosyası taşınamadı: %s", exc)

    def save(self) -> bool:
        """Ayarları atomik olarak diske yazar. Başarı durumunu döndürür."""
        with self._lock:
            parent = self.config_path.parent
            try:
                parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(
                    dir=str(parent), prefix=".config-", suffix=".tmp"
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(self._data, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_name, self.config_path)
                except Exception:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                    raise
                logger.debug("Ayar dosyası kaydedildi: %s", self.config_path)
                return True
            except Exception as exc:
                logger.error("Ayar dosyası kaydedilemedi: %s", exc)
                return False

    # ------------------------------------------------------------- erişimciler
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def get_str(self, key: str, default: str = "") -> str:
        value = self.get(key, default)
        return default if value is None else str(value)

    def set(self, key: str, value: Any, auto_save: bool = True) -> None:
        with self._lock:
            self._data[key] = value
        if auto_save:
            self.save()

    def update(self, values: Dict[str, Any], auto_save: bool = True) -> None:
        with self._lock:
            self._data.update(values)
        if auto_save:
            self.save()

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def reset_to_defaults(self, auto_save: bool = True) -> None:
        with self._lock:
            self._data = copy.deepcopy(self._defaults)
        if auto_save:
            self.save()

    # ------------------------------------------------------------- kısayollar
    @property
    def recordings_dir(self) -> Path:
        raw = self.get_str("recordings_dir", "").strip()
        return Path(raw).expanduser() if raw else paths.recordings_dir()

    def set_recordings_dir(self, path: str | Path, auto_save: bool = True) -> None:
        self.set("recordings_dir", str(Path(path)), auto_save=auto_save)

    def add_recent_project(self, project_path: str) -> None:
        recent = list(self.get("recent_projects", []) or [])
        project_path = str(project_path)
        if project_path in recent:
            recent.remove(project_path)
        recent.insert(0, project_path)
        self.set("recent_projects", recent[:10])

    def remove_recent_project(self, project_path: str) -> None:
        recent = [p for p in (self.get("recent_projects", []) or []) if p != str(project_path)]
        self.set("recent_projects", recent)

    def add_recent_recording(self, recording_path: str) -> None:
        recent = list(self.get("recent_recordings", []) or [])
        recording_path = str(recording_path)
        if recording_path in recent:
            recent.remove(recording_path)
        recent.insert(0, recording_path)
        self.set("recent_recordings", recent[:30])
