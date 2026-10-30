"""
VOX Studio — Merkezi yol yönetimi.

Tüm kullanıcı verisi yolları bu modülden gelir. `VOX_HOME` ortam değişkeni
ile kök dizin tamamen değiştirilebilir; bu sayede testler kullanıcı profilinde
yan etki bırakmaz ve taşınabilir kurulum mümkün olur.

Tasarım notu: Bu modül import anında hiçbir dizin oluşturmaz. Dizin oluşturma
yalnızca `ensure_dirs()` çağrıldığında (uygulama başlangıcı, kayıt yazımı,
log kurulumu) yapılır.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

ENV_HOME = "VOX_HOME"

_SUBDIRS = (
    "projects",
    "recordings",
    "models",
    "cache",
    "logs",
    "backups",
    "exports",
    "crash",
)


def app_home() -> Path:
    """Uygulama kök dizini. `VOX_HOME` verilmişse onu, aksi halde ~/.vox_studio."""
    raw = os.environ.get(ENV_HOME, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".vox_studio"


def _sub(name: str) -> Path:
    return app_home() / name


def data_dir() -> Path:
    return app_home()


def projects_dir() -> Path:
    return _sub("projects")


def recordings_dir() -> Path:
    return _sub("recordings")


def models_dir() -> Path:
    return _sub("models")


def cache_dir() -> Path:
    return _sub("cache")


def logs_dir() -> Path:
    return _sub("logs")


def backups_dir() -> Path:
    return _sub("backups")


def exports_dir() -> Path:
    return _sub("exports")


def crash_dir() -> Path:
    return _sub("crash")


def config_file() -> Path:
    return app_home() / "config.json"


def all_dirs() -> Dict[str, Path]:
    return {name: _sub(name) for name in _SUBDIRS}


def ensure_dirs() -> Dict[str, Path]:
    """Tüm uygulama dizinlerini oluşturur (idempotent) ve yolları döndürür."""
    created = {"root": app_home()}
    for name, path in all_dirs().items():
        path.mkdir(parents=True, exist_ok=True)
        created[name] = path
    return created
