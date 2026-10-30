"""
Arayüz için arka plan işçileri.

Dosya tarama, küçük resim (thumbnail) üretimi ve ses analizi GUI iş parçacığında
yapılmaz; büyük dosyalarda arayüz donmaz. Her işçi tek kullanımlıktır ve sonucu
sinyalle bildirir.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import soundfile as sf
from PySide6.QtCore import QThread, Signal

from ai.media_extractor import MediaExtractor
from audio.player import PlaybackInfo, probe_audio_file
from core import constants as C
from core.logger import get_logger
from dsp.analysis import analyze_file
from ui.widgets.waveform import peaks_from_file

logger = get_logger("ui.workers")


def classify_media_category(path: Path) -> str:
    """Dosya yoluna ve uzantısına göre medya kategorisini belirler."""
    suffix = path.suffix.lower()
    name_lower = path.name.lower()
    parent_lower = path.parent.name.lower()

    if suffix in C.VIDEO_EXTENSIONS:
        return "video"
    if suffix in C.IMAGE_EXTENSIONS:
        return "image"
    if suffix in C.SUBTITLE_EXTENSIONS:
        return "subtitle"
    if "_stems" in parent_lower or name_lower.startswith("stem_") or any(
        tag in name_lower for tag in ("_vocals", "_instrumental", "_dialogue", "_ambient", "_music", "_drums", "_bass")
    ):
        return "stem"
    if "_processed" in name_lower or "ai_" in name_lower:
        return "ai"
    if any(tag in name_lower for tag in ("kayit", "take", "dublaj", "rec_")) or suffix == ".wav":
        return "recording"
    return "audio"


def _scan_entry(path: Path) -> dict:
    entry = {
        "path": str(path),
        "name": path.name,
        "size": 0,
        "mtime": 0.0,
        "duration": 0.0,
        "samplerate": 0,
        "channels": 0,
        "subtype": "",
        "format": "",
        "category": classify_media_category(path),
        "resolution": "",
        "thumbnail": "",
        "error": "",
    }
    try:
        stat = path.stat()
        entry["size"] = int(stat.st_size)
        entry["mtime"] = float(stat.st_mtime)
        suffix = path.suffix.lower()

        # Hızlı WAV/FLAC/OGG/MP3 yolu
        if suffix in (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"):
            try:
                info = sf.info(str(path))
                entry.update(
                    {
                        "duration": float(info.frames) / float(info.samplerate) if info.samplerate else 0.0,
                        "samplerate": int(info.samplerate),
                        "channels": int(info.channels),
                        "subtype": str(info.subtype or ""),
                        "format": str(info.format or ""),
                    }
                )
                return entry
            except Exception:
                pass

        # PyAV / PIL / SRT genel medya incelemesi
        m_info = MediaExtractor.probe(path)
        sr = m_info.streams[0].sample_rate if m_info.streams else 0
        ch = m_info.streams[0].channels if m_info.streams else 0
        sub = m_info.streams[0].codec_name.upper() if m_info.streams else (m_info.video_codec or "").upper()
        res_str = f"{m_info.video_resolution[0]}x{m_info.video_resolution[1]}" if m_info.video_resolution else ""
        if m_info.has_video:
            entry["category"] = "video"
        elif m_info.media_category in ("image", "subtitle"):
            entry["category"] = m_info.media_category

        thumb_path = ""
        if entry["category"] in ("video", "image"):
            tp = MediaExtractor.extract_thumbnail(path, size=(96, 54))
            if tp:
                thumb_path = str(tp)

        entry.update(
            {
                "duration": float(m_info.duration_sec),
                "samplerate": int(sr),
                "channels": int(ch),
                "subtype": sub,
                "format": str(m_info.format_name or suffix.lstrip(".")).upper(),
                "resolution": res_str,
                "thumbnail": thumb_path,
            }
        )
    except Exception as exc:
        entry["error"] = str(exc)
    return entry


class LibraryScanWorker(QThread):
    """Kayıt klasörünü, alt stem klasörlerini ve içe aktarılan medya dosyalarını tarar."""

    scanned = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        directory: Path,
        parent=None,
        extra_paths: Optional[Sequence[str | Path]] = None,
    ):
        super().__init__(parent)
        self._directory = Path(directory)
        self._extra_paths = [Path(p) for p in (extra_paths or [])]

    def run(self) -> None:  # noqa: D401
        try:
            seen = set()
            files: List[Path] = []

            if self._directory.exists():
                for path in self._directory.iterdir():
                    if path.is_file():
                        if path.suffix.lower() in C.ALL_MEDIA_EXTENSIONS and ".part." not in path.name:
                            resolved = str(path.resolve())
                            if resolved not in seen:
                                seen.add(resolved)
                                files.append(path)
                    elif path.is_dir() and path.name.endswith("_stems"):
                        for sub in path.iterdir():
                            if sub.is_file() and sub.suffix.lower() in C.ALL_MEDIA_EXTENSIONS:
                                resolved = str(sub.resolve())
                                if resolved not in seen:
                                    seen.add(resolved)
                                    files.append(sub)

            for ep in self._extra_paths:
                if ep.exists() and ep.is_file() and ep.suffix.lower() in C.ALL_MEDIA_EXTENSIONS:
                    resolved = str(ep.resolve())
                    if resolved not in seen:
                        seen.add(resolved)
                        files.append(ep)

            files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
            self.scanned.emit([_scan_entry(p) for p in files])
        except Exception as exc:
            logger.exception("Kütüphane taranamadı")
            self.failed.emit(f"Kayıt klasörü taranamadı: {exc}")


class AnalysisWorker(QThread):
    """Bir dosyayı arka planda analiz eder."""

    done = Signal(str, object)
    failed = Signal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = str(path)

    def run(self) -> None:  # noqa: D401
        try:
            analysis = analyze_file(self.path)
            self.done.emit(self.path, analysis)
        except Exception as exc:
            logger.error("Analiz başarısız (%s): %s", self.path, exc)
            self.failed.emit(f"{Path(self.path).name}: {exc}")


class PeaksWorker(QThread):
    """Bir dosya için waveform tepe özetlerini arka planda üretir."""

    done = Signal(str, object)
    failed = Signal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = str(path)

    def run(self) -> None:  # noqa: D401
        try:
            mins, maxs, frames, samplerate = peaks_from_file(self.path)
            duration = frames / samplerate if samplerate else 0.0
            self.done.emit(self.path, (mins, maxs, duration))
        except Exception as exc:
            logger.error("Waveform üretilemedi (%s): %s", self.path, exc)
            self.failed.emit(f"{Path(self.path).name}: {exc}")


class ProbeWorker(QThread):
    """Dosya bilgisini arka planda okur (bozuk dosyada anlaşılır hata)."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = str(path)

    def run(self) -> None:  # noqa: D401
        try:
            self.done.emit(probe_audio_file(self.path))
        except Exception as exc:
            self.failed.emit(str(exc))


def probe_inline(path: Path) -> Optional[PlaybackInfo]:
    """Küçük dosyalar için senkron bilgi okuma (UI testleri)."""
    try:
        return probe_audio_file(path)
    except Exception:
        return None
