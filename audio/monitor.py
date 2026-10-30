"""
Giriş izleme (monitor) — kayıt başlatmadan önce mikrofonu doğrulamak için.

Ses kaydedilmez, diske yazılmaz; yalnızca seviye ölçülür. Kullanıcının doğru
cihazı seçtiğini ve sinyal geldiğini kayıttan önce görmesini sağlar.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import sounddevice as sd

from core.logger import get_logger

from .level import LevelAnalyzer, LevelSnapshot

logger = get_logger("audio.monitor")


class MonitorError(RuntimeError):
    """Kullanıcıya gösterilebilir izleme hatası."""


class InputMonitor:
    def __init__(
        self,
        on_error: Optional[Callable[[str], None]] = None,
        stream_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        self._on_error = on_error
        self._stream_factory = stream_factory
        self._stream = None
        self._level = LevelAnalyzer()
        self._lock = threading.Lock()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(
        self,
        device_index: Optional[int],
        samplerate: int,
        channels: int,
        blocksize: int = 1024,
    ) -> None:
        if self._active:
            return
        self._level.reset()

        def callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            try:
                self._level.push_peak(indata)
                self._level.push_block(indata)
            except Exception:
                pass

        try:
            if self._stream_factory is not None:
                stream = self._stream_factory(
                    device=device_index,
                    samplerate=samplerate,
                    channels=channels,
                    dtype="float32",
                    blocksize=blocksize,
                    callback=callback,
                )
            else:
                stream = sd.InputStream(
                    device=device_index,
                    samplerate=samplerate,
                    channels=channels,
                    dtype="float32",
                    blocksize=blocksize,
                    callback=callback,
                )
            stream.start()
        except Exception as exc:
            message = (
                f"Giriş izleme başlatılamadı: {exc}. Cihazın bağlı olduğunu, başka bir uygulama "
                "tarafından kullanılmadığını ve Windows mikrofon iznini kontrol edin."
            )
            logger.error(message)
            if self._on_error:
                self._on_error(message)
            raise MonitorError(message) from exc

        with self._lock:
            self._stream = stream
            self._active = True

    def stop(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            self._active = False
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def snapshot(self) -> LevelSnapshot:
        return self._level.snapshot()
