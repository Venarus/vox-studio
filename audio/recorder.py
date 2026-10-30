"""
Gerçek zamanlı kayıt motoru.

Mimari
------
``PortAudio callback``  →  halka tampon (bellek ayırmaz)  →  yazıcı iş parçacığı  →  WAV/FLAC

* Callback içinde disk I/O, ağır hesaplama veya Qt çağrısı **yapılmaz**.
* Yazma işlemi ayrı iş parçacığındadır; arayüz donmaz.
* Kayıt önce ``*.part.wav`` dosyasına yazılır; düzgün bitirilirse son dosyaya
  taşınır. Beklenmedik kapanmada dosya korunur ve ``recover_pending_recordings``
  ile kurtarılabilir.
* Mikrofon kopması, taşma (overflow) ve kırpılma (clip) açıkça raporlanır.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from core import paths
from core.constants import (
    APP_VERSION,
    CLIP_THRESHOLD,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_RECORDING_SUBTYPE,
    DEFAULT_SAMPLE_RATE,
    RECORDING_FILENAME_TEMPLATE,
)
from core.logger import get_logger

from .level import LevelAnalyzer, LevelSnapshot
from .ringbuffer import RingBuffer
from .waveform import WaveformPeaks

logger = get_logger("audio.recorder")

_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}


class RecordingState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"
    FINISHED = "finished"
    ERROR = "error"
    CANCELLED = "cancelled"


class RecordingError(RuntimeError):
    """Kullanıcıya gösterilebilir kayıt hatası."""


def sanitize_filename(text: str, fallback: str = "kayit", max_length: int = 80) -> str:
    """Dosya adı için güvenli metin üretir (Türkçe karakterler korunur)."""
    text = unicodedata.normalize("NFC", str(text or "")).strip()
    text = _ILLEGAL_FILENAME_CHARS.sub("_", text)
    text = re.sub(r"[\s_]+", "_", text)
    text = text.strip(" ._")
    if not text:
        text = fallback
    if text.upper().split(".")[0] in _WINDOWS_RESERVED:
        text = f"_{text}"
    return text[:max_length]


def build_recording_filename(
    label: str = "",
    when: Optional[datetime] = None,
    extension: str = ".wav",
) -> str:
    when = when or datetime.now()
    stamp = when.strftime("%Y%m%d_%H%M%S")
    safe_label = sanitize_filename(label, fallback="") if label else ""
    name = RECORDING_FILENAME_TEMPLATE.format(timestamp=stamp, label=safe_label) if safe_label else stamp
    name = re.sub(r"_+", "_", name).strip("_")
    if not extension.startswith("."):
        extension = "." + extension
    return f"{name}{extension}"


@dataclass
class RecordingConfig:
    path: Path
    device_index: Optional[int] = None
    device_name: str = ""
    samplerate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1
    subtype: str = DEFAULT_RECORDING_SUBTYPE
    blocksize: int = DEFAULT_BLOCK_SIZE
    ring_seconds: float = 10.0
    label: str = ""


@dataclass
class RecordingResult:
    path: Optional[Any] = None
    frames: int = 0
    samplerate: int = 0
    channels: int = 0
    subtype: str = ""
    started_at: str = ""
    finished_at: str = ""
    peak: float = 0.0
    clip_count: int = 0
    overflow_frames: int = 0
    underrun_frames: int = 0
    device_name: str = ""
    recovered: bool = False
    cancelled: bool = False
    duration: float = 0.0
    peak_dbfs: Optional[float] = None
    clipped_samples: Optional[int] = None

    def __post_init__(self):
        if self.path is not None:
            self.path = str(self.path)
        if self.clipped_samples is not None and not self.clip_count:
            self.clip_count = self.clipped_samples
        if self.peak_dbfs is not None and not self.peak:
            import math
            self.peak = 10.0 ** (self.peak_dbfs / 20.0)
        if self.duration and not self.frames and self.samplerate:
            self.frames = int(self.duration * self.samplerate)
        elif self.frames and self.samplerate and not self.duration:
            self.duration = self.frames / self.samplerate

    @property
    def peak_db(self) -> float:
        import math

        return 20.0 * math.log10(self.peak) if self.peak > 0 else -math.inf


class _NullStream:
    """Testlerde gerçek cihaz açmadan motoru sınamak için."""

    def __init__(self) -> None:
        self.active = True

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False


class Recorder:
    """Kayıt motoru. UI'dan bağımsızdır; durum değişimlerini callback ile bildirir."""

    def __init__(
        self,
        on_state: Optional[Callable[[RecordingState], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        stream_factory: Optional[Callable[[RecordingConfig, Callable], object]] = None,
        writer_poll: float = 0.05,
    ) -> None:
        self._on_state = on_state
        self._on_error = on_error
        self._stream_factory = stream_factory
        self._writer_poll = writer_poll

        self._state = RecordingState.IDLE
        self._state_lock = threading.RLock()
        self._cfg: Optional[RecordingConfig] = None
        self._stream = None
        self._ring: Optional[RingBuffer] = None
        self._level = LevelAnalyzer()
        self._waveform: Optional[WaveformPeaks] = None
        self._writer: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._data_event = threading.Event()
        self._sfile: Optional[sf.SoundFile] = None

        self._frames_written = 0
        self._started_at: Optional[datetime] = None
        self._finished_at: Optional[datetime] = None
        self._part_path: Optional[Path] = None
        self._meta_path: Optional[Path] = None
        self._peak = 0.0
        self._error_message: Optional[str] = None
        self._last_flush = 0.0
        self._status_flags = 0
        self._device_lost = False

    # -------------------------------------------------------------- özellikler
    @property
    def state(self) -> RecordingState:
        return self._state

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def duration(self) -> float:
        cfg = self._cfg
        return self._frames_written / cfg.samplerate if cfg and cfg.samplerate else 0.0

    @property
    def is_active(self) -> bool:
        return self._state in (RecordingState.STARTING, RecordingState.RECORDING, RecordingState.PAUSED)

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    def level_snapshot(self) -> LevelSnapshot:
        return self._level.snapshot()

    def waveform_view(self, max_points: int = 2000):
        if self._waveform is None:
            import numpy as _np

            return _np.zeros(0, _np.float32), _np.zeros(0, _np.float32)
        return self._waveform.view(max_points)

    # ---------------------------------------------------------------- durum
    def _set_state(self, state: RecordingState) -> None:
        with self._state_lock:
            if self._state == state:
                return
            self._state = state
        if self._on_state:
            try:
                self._on_state(state)
            except Exception:  # UI hatası motoru durdurmamalı
                logger.exception("on_state geri çağrısı hata verdi")

    def _fail(self, message: str) -> None:
        self._error_message = message
        logger.error("Kayıt hatası: %s", message)
        self._set_state(RecordingState.ERROR)
        if self._on_error:
            try:
                self._on_error(message)
            except Exception:
                logger.exception("on_error geri çağrısı hata verdi")

    # ---------------------------------------------------------------- başlat
    def start(self, cfg: RecordingConfig) -> None:
        if self.is_active:
            raise RecordingError("Zaten bir kayıt sürüyor.")

        if cfg.samplerate <= 0 or cfg.channels <= 0:
            raise RecordingError("Geçersiz örnekleme hızı veya kanal sayısı.")

        cfg.path = Path(cfg.path)
        try:
            cfg.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RecordingError(f"Kayıt klasörü oluşturulamadı: {exc}") from exc

        self._reset_runtime()
        self._cfg = cfg
        self._set_state(RecordingState.STARTING)

        self._part_path = cfg.path.with_name(cfg.path.stem + ".part" + cfg.path.suffix)
        self._meta_path = cfg.path.with_suffix(".json")
        try:
            self._sfile = sf.SoundFile(
                str(self._part_path),
                mode="w",
                samplerate=cfg.samplerate,
                channels=cfg.channels,
                subtype=cfg.subtype,
            )
        except Exception as exc:
            self._set_state(RecordingState.ERROR)
            raise RecordingError(
                f"Kayıt dosyası oluşturulamadı ({cfg.subtype}): {exc}. "
                "Disk alanını ve klasör izinlerini kontrol edin."
            ) from exc

        self._ring = RingBuffer(int(cfg.ring_seconds * cfg.samplerate), cfg.channels)
        self._level = LevelAnalyzer()
        self._waveform = WaveformPeaks(channels=cfg.channels)
        self._started_at = datetime.now()
        self._write_meta(initial=True)

        try:
            self._stream = self._open_stream(cfg)
            self._stream.start()
        except Exception as exc:
            self._close_file()
            self._cleanup_partial()
            self._set_state(RecordingState.ERROR)
            raise RecordingError(
                f"Ses cihazı açılamadı: {exc}. Cihazın bağlı olduğunu, başka bir uygulama "
                "tarafından kullanılmadığını ve Windows mikrofon iznini kontrol edin."
            ) from exc

        self._stop_event.clear()
        self._data_event.clear()
        self._writer = threading.Thread(target=self._writer_loop, name="vox-recorder-writer", daemon=True)
        self._writer.start()
        self._set_state(RecordingState.RECORDING)
        logger.info(
            "Kayıt başladı: %s | %s | %d Hz | %d kanal | %s",
            cfg.path.name, cfg.device_name or cfg.device_index, cfg.samplerate, cfg.channels, cfg.subtype,
        )

    def _open_stream(self, cfg: RecordingConfig):
        if self._stream_factory is not None:
            return self._stream_factory(cfg, self._audio_callback)
        return sd.InputStream(
            device=cfg.device_index,
            samplerate=cfg.samplerate,
            channels=cfg.channels,
            dtype="float32",
            blocksize=cfg.blocksize,
            callback=self._audio_callback,
            finished_callback=self._on_stream_finished,
        )

    def _reset_runtime(self) -> None:
        self._frames_written = 0
        self._peak = 0.0
        self._error_message = None
        self._status_flags = 0
        self._device_lost = False
        self._started_at = None
        self._finished_at = None
        self._sfile = None
        self._ring = None
        self._waveform = None

    # ---------------------------------------------------------------- callback
    def _audio_callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        """PortAudio callback: bellek ayırmaz, bloklamaz, istisna sızdırmaz."""
        try:
            if status:
                self._status_flags |= 1
            if self._state is not RecordingState.RECORDING:
                return
            self._level.push_peak(indata)
            ring = self._ring
            if ring is not None and ring.write(indata) == 0:
                # Taşma zaten sayaçta; disk yavaşsa veri kaybı kullanıcıya bildirilir.
                pass
            self._data_event.set()
        except Exception as exc:  # callback'ten istisna çıkmamalı
            self._device_lost = True
            self._fail(f"Kayıt akışında hata: {exc}. Kayıt durduruluyor; dosya kurtarılabilir.")
            self._stop_event.set()

    def _on_stream_finished(self) -> None:
        if self._state in (RecordingState.RECORDING, RecordingState.PAUSED):
            self._device_lost = True
            self._fail(
                "Ses cihazı akışı beklenmedik şekilde durdu (bağlantı kesilmiş olabilir). "
                "Kayıt güvenli biçimde sonlandırılıyor."
            )
            self._stop_event.set()

    # ------------------------------------------------------------ yazıcı thread
    def _writer_loop(self) -> None:
        cfg = self._cfg
        assert cfg is not None and self._ring is not None
        staging = np.empty((cfg.blocksize * 8, cfg.channels), dtype=np.float32)
        last_activity = time.monotonic()

        while not self._stop_event.is_set():
            try:
                got = self._ring.read_into(staging)
                if got:
                    block = staging[:got]
                    self._write_block(block)
                    last_activity = time.monotonic()
                else:
                    self._data_event.wait(self._writer_poll)
                    self._data_event.clear()
                    if time.monotonic() - last_activity > 1.0 and not self._stream_active():
                        if not self._device_lost and self._state in (
                            RecordingState.RECORDING,
                            RecordingState.PAUSED,
                        ):
                            self._on_stream_finished()
            except Exception as exc:
                self._fail(f"Kayıt diske yazılamadı: {exc}. Kayıt durduruluyor.")
                self._stop_event.set()

        # Kalan veriyi boşalt
        try:
            while True:
                got = self._ring.read_into(staging)
                if not got:
                    break
                self._write_block(staging[:got])
        except Exception:
            logger.exception("Kalan tampon yazılamadı")

    def _stream_active(self) -> bool:
        stream = self._stream
        try:
            return bool(getattr(stream, "active", False))
        except Exception:
            return False

    def _write_block(self, block: np.ndarray) -> None:
        if self._sfile is None:
            return
        self._sfile.write(block)
        frames = block.shape[0]
        self._frames_written += frames
        self._level.push_block(block)
        if self._waveform is not None:
            self._waveform.add(block)
        local_peak = float(max(-block.min(), block.max()))
        if local_peak > self._peak:
            self._peak = local_peak

        now = time.monotonic()
        if now - self._last_flush >= 1.0:
            try:
                self._sfile.flush()
            except Exception:
                pass
            self._last_flush = now

    # ----------------------------------------------------------------- kontrol
    def pause(self) -> None:
        if self._state is not RecordingState.RECORDING:
            return
        self._set_state(RecordingState.PAUSED)
        logger.info("Kayıt duraklatıldı (%.1f sn)", self.duration)

    def resume(self) -> None:
        if self._state is not RecordingState.PAUSED:
            return
        self._set_state(RecordingState.RECORDING)
        logger.info("Kayda devam edildi")

    def stop(self) -> RecordingResult:
        with self._state_lock:
            if self._state in (RecordingState.IDLE, RecordingState.FINISHED, RecordingState.CANCELLED):
                raise RecordingError("Durdurulacak etkin bir kayıt yok.")
        self._set_state(RecordingState.STOPPING)
        self._stop_event.set()
        self._data_event.set()

        try:
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        finally:
            if self._writer is not None and self._writer.is_alive():
                self._writer.join(timeout=10.0)
                if self._writer.is_alive():
                    logger.error("Yazıcı iş parçacığı zamanında bitmedi")
            self._writer = None

        result = self._finalize(cancelled=False)
        self._set_state(RecordingState.ERROR if self._error_message else RecordingState.FINISHED)
        return result

    def cancel(self) -> None:
        """Kaydı iptal eder ve yarım dosyayı siler (kullanıcı onayı UI'da alınır)."""
        if self.is_active:
            try:
                self._set_state(RecordingState.STOPPING)
                self._stop_event.set()
                self._data_event.set()
                if self._stream is not None:
                    try:
                        self._stream.stop()
                    except Exception:
                        pass
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                if self._writer is not None:
                    self._writer.join(timeout=10.0)
                self._writer = None
            finally:
                self._close_file()
                self._cleanup_partial()
                self._frames_written = 0
                self._set_state(RecordingState.CANCELLED)
                logger.info("Kayıt iptal edildi")

    # --------------------------------------------------------------- bitirme
    def _finalize(self, cancelled: bool) -> RecordingResult:
        cfg = self._cfg
        if cfg is None:
            raise RecordingError("Etkin kayıt yapılandırması yok.")

        self._close_file()

        final_path = cfg.path
        path_str: Optional[str] = None
        if self._part_path and self._part_path.exists():
            if cancelled:
                self._cleanup_partial()
            else:
                try:
                    if final_path.exists():
                        # Var olan kaydın üzerine asla sessizce yazma
                        final_path = final_path.with_name(
                            f"{final_path.stem}-{datetime.now().strftime('%H%M%S')}{final_path.suffix}"
                        )
                    os.replace(self._part_path, final_path)
                    path_str = str(final_path)
                except Exception as exc:
                    self._fail(f"Kayıt dosyası taşınamadı: {exc}")
                    path_str = str(self._part_path)

        self._finished_at = datetime.now()
        result = RecordingResult(
            path=path_str,
            frames=self._frames_written,
            samplerate=cfg.samplerate,
            channels=cfg.channels,
            subtype=cfg.subtype,
            started_at=self._started_at.isoformat(timespec="seconds") if self._started_at else "",
            finished_at=self._finished_at.isoformat(timespec="seconds"),
            peak=self._peak,
            clip_count=self._level.snapshot().clip_count,
            overflow_frames=self._ring.overflow_frames if self._ring else 0,
            underrun_frames=self._ring.underrun_frames if self._ring else 0,
            device_name=cfg.device_name,
            cancelled=cancelled,
        )
        self._write_meta(
            initial=False,
            result=result,
            target=Path(path_str) if path_str else None,
        )
        logger.info(
            "Kayıt bitti: %s | %.2f sn | %d frame | tepe %.4f | clip %d | taşma %d",
            path_str, result.duration, result.frames, result.peak, result.clip_count, result.overflow_frames,
        )
        return result

    def _close_file(self) -> None:
        if self._sfile is not None:
            try:
                if self._waveform is not None:
                    self._waveform.flush_partial()
                self._sfile.flush()
                self._sfile.close()
            except Exception:
                logger.exception("Kayıt dosyası kapatılırken hata")
            finally:
                self._sfile = None

    def _cleanup_partial(self) -> None:
        for p in (self._part_path, self._meta_path):
            try:
                if p and p.exists():
                    p.unlink()
            except Exception:
                logger.warning("Geçici dosya silinemedi: %s", p)

    def _write_meta(
        self,
        initial: bool,
        result: Optional[RecordingResult] = None,
        target: Optional[Path] = None,
    ) -> None:
        meta_path = target.with_suffix(".json") if target is not None else self._meta_path
        if meta_path is None or self._cfg is None:
            return
        payload = {
            "app_version": APP_VERSION,
            "device_name": self._cfg.device_name,
            "device_index": self._cfg.device_index,
            "samplerate": self._cfg.samplerate,
            "channels": self._cfg.channels,
            "subtype": self._cfg.subtype,
            "started_at": self._started_at.isoformat(timespec="seconds") if self._started_at else "",
            "status": "recording" if initial else "finished",
            "label": self._cfg.label,
            "target_path": str(self._cfg.path),
        }
        if result is not None:
            payload.update(asdict(result))
            payload["status"] = "finished"
        try:
            meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.warning("Kayıt meta dosyası yazılamadı: %s", meta_path)


# ---------------------------------------------------------------------------
# Kurtarma
# ---------------------------------------------------------------------------
def recover_pending_recordings(directory: Optional[Path] = None) -> List[Path]:
    """Beklenmedik kapanmadan kalan ``*.part.wav`` dosyalarını kurtarır.

    Dosya okunabiliyorsa son adına taşınır; okunamıyorsa dokunulmaz.
    Dönen liste kurtarılan dosyaların yollarıdır.
    """
    directory = Path(directory) if directory is not None else paths.recordings_dir()
    if not directory.exists():
        return []

    recovered: List[Path] = []
    parts = [
        p
        for p in sorted(directory.glob("*.part.*"))
        if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".ogg", ".aiff"}
    ]
    for part in parts:
        try:
            info = sf.info(str(part))
            if info.frames <= 0:
                logger.warning("Kurtarma: boş dosya atlandı: %s", part)
                continue
        except Exception as exc:
            logger.warning("Kurtarma: okunamayan dosya atlandı (%s): %s", part, exc)
            continue

        final = part.with_name(part.name.replace(".part", ""))
        if final.exists():
            final = final.with_name(f"{final.stem}-kurtarilan{final.suffix}")
        try:
            os.replace(part, final)
        except Exception as exc:
            logger.error("Kurtarma: dosya taşınamadı (%s): %s", part, exc)
            continue

        meta_src = part.with_name(part.name.replace(".part", "")).with_suffix(".json")
        if meta_src.exists():
            try:
                data = json.loads(meta_src.read_text(encoding="utf-8"))
                data.update({"status": "recovered", "recovered": True, "frames": info.frames})
                final.with_suffix(".json").write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                if meta_src != final.with_suffix(".json"):
                    meta_src.unlink()
            except Exception:
                logger.warning("Kurtarma: meta dosyası güncellenemedi: %s", meta_src)
        recovered.append(final)
        logger.info("Kurtarılan kayıt: %s (%d frame)", final, info.frames)
    return recovered
