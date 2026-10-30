"""Biçimlendirme yardımcıları (süre, dB, dosya boyutu)."""

from __future__ import annotations

import math


def format_duration(seconds: float, millis: bool = True) -> str:
    """00:00:00.0 veya 00:00 biçiminde süre."""
    seconds = max(0.0, float(seconds or 0.0))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}" + (f".{int((secs % 1) * 10)}" if millis else "")
    if millis:
        return f"{minutes:02d}:{int(secs):02d}.{int((secs % 1) * 10)}"
    return f"{minutes:02d}:{int(secs):02d}"


def format_compact_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    if seconds < 60:
        return f"{seconds:.1f} sn"
    minutes = int(seconds // 60)
    return f"{minutes} dk {int(seconds % 60)} sn"


def format_db(db: float, digits: int = 1) -> str:
    if db is None or not math.isfinite(db):
        return "—"
    return f"{db:.{digits}f}"


def format_file_size(num_bytes: float) -> str:
    size = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_timestamp_short(iso_text: str) -> str:
    """2026-10-29T14:05:09 -> 29.10.2026 14:05"""
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(iso_text)
        return dt.strftime("%d.%m.%Y %H:%M")
    except Exception:
        return iso_text or "—"
