"""
Ses dosyası oynatıcı.

Dosya belleğe tamamen yüklenmez: okuma iş parçacığı dosyayı blok blok okur,
halka tampona yazar; PortAudio callback'i yalnızca tampondan kopyalar.
Bu sayede uzun dosyalarda bellek kullanımı sabit kalır.

Oynatma konumu gerçek tüketilen frame sayısından hesaplanır.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from core.logger import get_logger

from .ringbuffer import RingBuffer

logger = get_logger("audio.player")


class PlaybackState(str, Enum):
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"


class PlaybackError(RuntimeError):
    """Kullanıcıya gösterilebilir oynatma hatası."""


@dataclass(frozen=True)
class PlaybackInfo:
    path: str
    samplerate: int
    channels: int
    frames: int
    subtype: str
    format: str
    decoded_path: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.frames / self.samplerate if self.samplerate else 0.0


def probe_audio_file(path: str | Path) -> PlaybackInfo:
    """Dosyayı açmadan bilgi verir; libsndfile dışı kapsayıcılarda (M4A, AAC, Opus, WMA, MP4 vb.)
    PyAV/FFmpeg ile önbelleğe çözerek oynatılabilir hale getirir; bozuk dosyada anlaşılır hata üretir."""
    p = Path(path)
    if not p.exists():
        raise PlaybackError(f"Dosya bulunamadı: {p}")
    try:
        info = sf.info(str(p))
        return PlaybackInfo(
            path=str(p),
            samplerate=int(info.samplerate),
            channels=int(info.channels),
            frames=int(info.frames),
            subtype=str(info.subtype or ""),
            format=str(info.format or ""),
        )
    except Exception as exc:
        try:
            import hashlib
            from ai.media_extractor import MediaExtractor
            from core import paths

            m_info = MediaExtractor.probe(p)
            if not m_info.streams:
                raise PlaybackError(f"Dosyada ses akışı bulunamadı: {p.name}")
            stat = p.stat()
            digest = hashlib.sha1(f"{p.resolve()}|{stat.st_size}|{stat.st_mtime}".encode("utf-8", errors="ignore")).hexdigest()[:20]
            cache_wav = paths.cache_dir() / "decoded_audio" / f"dec_{digest}.wav"
            if not cache_wav.exists() or cache_wav.stat().st_size == 0:
                sr = m_info.streams[0].sample_rate or 48000
                ch = min(2, max(1, m_info.streams[0].channels))
                MediaExtractor.extract_audio(
                    p, output_wav_path=cache_wav, target_sample_rate=sr, target_channels=ch
                )
            c_info = sf.info(str(cache_wav))
            return PlaybackInfo(
                path=str(p),
                samplerate=int(c_info.samplerate),
                channels=int(c_info.channels),
                frames=int(c_info.frames),
                subtype=str(m_info.streams[0].codec_name.upper()),
                format=str(m_info.format_name.upper()),
                decoded_path=str(cache_wav),
            )
        except Exception:
            raise PlaybackError(
                f"Ses dosyası okunamadı: {p.name} ({exc}). Dosya bozuk olabilir veya "
                "desteklenmeyen bir kodlayıcı kullanıyor olabilir."
            ) from exc


class Player:
    """Basit, güvenilir oynatıcı (tek dosya). UI'dan bağımsızdır."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        on_state: Optional[Callable[[PlaybackState], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        output_factory: Optional[Callable[..., object]] = None,
        gain: float = 1.0,
    ) -> None:
        self._device = device_index
        self._on_state = on_state
        self._on_error = on_error
        self._output_factory = output_factory
        self._state = PlaybackState.IDLE
        self._info: Optional[PlaybackInfo] = None
        self._gain = float(gain)

        self._stream = None
        self._reader: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ring: Optional[RingBuffer] = None
        self._sfile: Optional[sf.SoundFile] = None
        self._frames_consumed = 0
        self._start_frame = 0
        self._out_channels = 0
        self._eof = False
        self._lock = threading.RLock()
        self._cb_tmp = np.zeros((0, 1), dtype=np.float32)

    # ------------------------------------------------------------------ meta
    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def info(self) -> Optional[PlaybackInfo]:
        return self._info

    @property
    def duration(self) -> float:
        return self._info.duration if self._info else 0.0

    @property
    def position(self) -> float:
        if not self._info or not self._info.samplerate:
            return 0.0
        return (self._start_frame + self._frames_consumed) / self._info.samplerate

    @property
    def is_playing(self) -> bool:
        return self._state is PlaybackState.PLAYING

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = max(0.0, min(2.0, float(value)))

    def set_device(self, device_index: Optional[int]) -> None:
        self._device = device_index

    def _set_state(self, state: PlaybackState) -> None:
        if self._state == state:
            return
        self._state = state
        if self._on_state:
            try:
                self._on_state(state)
            except Exception:
                logger.exception("on_state geri çağrısı hata verdi")

    def _fail(self, message: str) -> None:
        logger.error("Oynatma hatası: %s", message)
        if self._on_error:
            try:
                self._on_error(message)
            except Exception:
                logger.exception("on_error geri çağrısı hata verdi")

    # ------------------------------------------------------------------ yükle
    def load(self, path: str | Path) -> PlaybackInfo:
        self.stop()
        info = probe_audio_file(path)
        self._info = info
        self._frames_consumed = 0
        self._start_frame = 0
        return info

    # ------------------------------------------------------------------ kontrol
    def play(self, start_seconds: float = 0.0) -> None:
        if self._info is None:
            raise PlaybackError("Önce bir ses dosyası yükleyin.")
        if self._state is PlaybackState.PLAYING:
            return

        info = self._info
        start_frame = int(max(0.0, min(start_seconds, info.duration)) * info.samplerate)
        if start_frame >= info.frames:
            start_frame = 0

        self._stop_stream()
        self._start_frame = start_frame
        self._frames_consumed = 0
        self._eof = False

        try:
            self._sfile = sf.SoundFile(str(info.decoded_path or info.path), mode="r")
            self._sfile.seek(start_frame)
        except Exception as exc:
            raise PlaybackError(f"Dosya açılamadı: {exc}") from exc

        self._out_channels = info.channels
        self._ring = RingBuffer(int(info.samplerate * 2.0), info.channels)
        self._cb_tmp = np.zeros((4096, info.channels), dtype=np.float32)

        try:
            self._stream = self._open_output(info)
            self._stream.start()
        except Exception as exc:
            self._close_file()
            raise PlaybackError(
                f"Çıkış cihazı açılamadı: {exc}. Farklı bir çıkış cihazı seçmeyi deneyin."
            ) from exc

        self._stop_event.clear()
        self._reader = threading.Thread(target=self._reader_loop, name="vox-player-reader", daemon=True)
        self._reader.start()
        self._set_state(PlaybackState.PLAYING)

    def _open_output(self, info: PlaybackInfo):
        def factory(channels: int, sr: int):
            if self._output_factory is not None:
                return self._output_factory(
                    samplerate=sr,
                    channels=channels,
                    dtype="float32",
                    blocksize=1024,
                    device=self._device,
                    callback=self._output_callback,
                )
            return sd.OutputStream(
                samplerate=sr,
                channels=channels,
                dtype="float32",
                blocksize=1024,
                device=self._device,
                callback=self._output_callback,
            )

        rates_to_try = [info.samplerate]
        for candidate_sr in (48000, 44100):
            if candidate_sr not in rates_to_try:
                rates_to_try.append(candidate_sr)

        channels_to_try = [1, 2] if info.channels == 1 else [info.channels]
        if 2 not in channels_to_try:
            channels_to_try.append(2)

        last_exc = None
        for sr in rates_to_try:
            for ch in channels_to_try:
                try:
                    stream = factory(ch, sr)
                    self._out_channels = ch
                    return stream
                except Exception as exc:
                    last_exc = exc
                    continue

        if last_exc:
            raise last_exc
        raise RuntimeError("Hiçbir çıkış yapılandırması açılamadı.")

    def _output_callback(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        try:
            ring = self._ring
            if ring is None:
                outdata.fill(0)
                return
            if self._out_channels == ring.channels:
                got = ring.read_into(outdata)
            else:
                tmp = self._cb_tmp
                if tmp.shape[0] < frames:
                    tmp = np.zeros((frames, ring.channels), dtype=np.float32)
                got = ring.read_into(tmp[:frames])
                if self._out_channels == 2 and ring.channels == 1:
                    outdata[:got, 0] = tmp[:got, 0]
                    outdata[:got, 1] = tmp[:got, 0]
                else:
                    outdata[:got] = tmp[:got, : self._out_channels]
            if got < frames:
                outdata[got:].fill(0)
            self._frames_consumed += got
        except Exception:
            outdata.fill(0)

    def _reader_loop(self) -> None:
        info = self._info
        ring = self._ring
        sfile = self._sfile
        assert info is not None and ring is not None and sfile is not None
        chunk = 16384
        empty_since: Optional[float] = None

        try:
            while not self._stop_event.is_set():
                if ring.free_space() < chunk:
                    time.sleep(0.01)
                    continue
                data = sfile.read(chunk, dtype="float32", always_2d=True)
                if data.shape[0] == 0:
                    self._eof = True
                    break
                if self._gain != 1.0:
                    data = data * np.float32(self._gain)
                ring.write(data)
        except Exception as exc:
            self._fail(f"Dosya okunurken hata: {exc}")
            self._stop_event.set()
            return

        # Tampon boşalınca oynatmayı bitir
        while not self._stop_event.is_set():
            if ring.available_read() == 0:
                if empty_since is None:
                    empty_since = time.monotonic()
                elif time.monotonic() - empty_since > 0.08:
                    break
            else:
                empty_since = None
            time.sleep(0.01)

        if not self._stop_event.is_set():
            self._finish_natural()

    def _finish_natural(self) -> None:
        with self._lock:
            self._stop_stream()
            if self._info is not None:
                self._frames_consumed = max(0, self._info.frames - self._start_frame)
            self._set_state(PlaybackState.IDLE)

    def resume(self) -> None:
        """Duraklatılmış oynatmayı kaldığı yerden sürdürür."""
        if self._state is not PlaybackState.PAUSED:
            return
        pos = self.position
        self._set_state(PlaybackState.IDLE)
        self.play(pos)

    def pause(self) -> None:
        if self._state is not PlaybackState.PLAYING:
            return
        self._stop_stream()
        self._set_state(PlaybackState.PAUSED)

    def stop(self) -> None:
        if self._state is PlaybackState.IDLE and self._stream is None and self._reader is None:
            return
        self._stop_stream()
        self._set_state(PlaybackState.IDLE)

    def seek(self, seconds: float) -> None:
        """Konumu değiştirir; çalıyorsa yeni konumdan devam eder."""
        if self._info is None:
            return
        was_playing = self._state is PlaybackState.PLAYING
        target = max(0.0, min(float(seconds), self._info.duration))
        self._stop_stream()
        self._start_frame = int(target * self._info.samplerate)
        self._frames_consumed = 0
        if was_playing:
            self._set_state(PlaybackState.IDLE)
            self.play(target)

    # ------------------------------------------------------------------ temizlik
    def _stop_stream(self) -> None:
        self._stop_event.set()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        reader = self._reader
        self._reader = None
        if reader is not None and reader.is_alive() and reader is not threading.current_thread():
            reader.join(timeout=2.0)
        self._close_file()

    def _close_file(self) -> None:
        if self._sfile is not None:
            try:
                self._sfile.close()
            except Exception:
                pass
            self._sfile = None

    def close(self) -> None:
        self.stop()
        self._info = None
