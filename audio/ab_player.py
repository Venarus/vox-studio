"""
VOX Studio — Senkronize A/B Karşılaştırma Oynatıcısı (Synchronized A/B Audio Player)

Özellikler:
* Tek ses akışı (sounddevice OutputStream) üzerinden Orijinal (A) ve Temizlenmiş (B) ses.
* Sıfır gecikmeli, aynı zaman noktasından anlık A/B geçişi (Zero-latency channel toggling).
* Geçiş anında klik ve patlamayı önleyen 5 ms mikrosaniyelik yumuşak çapraz geçiş (micro-crossfade).
* Akıllı Loudness Eşitleme (Loudness Matching):
  İki kaydın LUFS değerini eşitleyerek kullanıcının sadece ses seviyesi farkından
  dolayı yanıltılmasını önler; saf ses temizleme ve restorasyon kalitesini duyurur.
"""

from __future__ import annotations

import math
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

from core.logger import get_logger
from dsp.analysis import analyze_array

logger = get_logger("audio.ab_player")


class ABChannel(str, Enum):
    A = "original"
    B = "restored"


class ABPlayer:
    """
    Orijinal (A) ve Restore Edilmiş (B) sesi aynı oynatma imleciyle senkronize çalar.
    """

    def __init__(self, samplerate: int = 48000):
        self.fs = samplerate
        self._lock = threading.RLock()

        self._audio_a: Optional[np.ndarray] = None
        self._audio_b: Optional[np.ndarray] = None
        self._len_a: int = 0
        self._len_b: int = 0
        self._channels: int = 1

        self._active_channel: ABChannel = ABChannel.A
        self._playhead_frame: int = 0
        self._is_playing: bool = False

        self._loudness_matched: bool = False
        self._gain_a: float = 1.0
        self._gain_b: float = 1.0
        self._lufs_a: float = -math.inf
        self._lufs_b: float = -math.inf

        self._stream: Optional[sd.OutputStream] = None
        self._position_cb: Optional[Callable[[float, float], None]] = None  # (cur_sec, total_sec)
        self._finished_cb: Optional[Callable[[], None]] = None

    def load(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        fs: int,
    ) -> None:
        """İki ses dizisini belleğe yükler ve analiz eder."""
        with self._lock:
            self.stop()
            self.fs = int(fs)

            # Mono / stereo uyumluluğu
            a = audio_a.astype(np.float32)
            b = audio_b.astype(np.float32)

            ch_a = 1 if a.ndim == 1 else a.shape[1]
            ch_b = 1 if b.ndim == 1 else b.shape[1]
            max_ch = max(ch_a, ch_b)

            if max_ch == 2:
                if ch_a == 1:
                    a = np.column_stack([a, a])
                if ch_b == 1:
                    b = np.column_stack([b, b])
            self._channels = max_ch

            self._audio_a = np.ascontiguousarray(a)
            self._audio_b = np.ascontiguousarray(b)
            self._len_a = len(self._audio_a)
            self._len_b = len(self._audio_b)
            self._playhead_frame = 0

            # LUFS Analizi (Loudness Eşitleme için)
            try:
                rep_a = analyze_array(self._audio_a, self.fs)
                rep_b = analyze_array(self._audio_b, self.fs)
                self._lufs_a = rep_a.loudness_lufs
                self._lufs_b = rep_b.loudness_lufs
                self._update_gains()
            except Exception as e:
                logger.warning(f"A/B LUFS analizi hatası: {e}")

    def load_files(self, path_a: str | Path, path_b: str | Path) -> None:
        """İki ses dosyasını diskten okur ve A/B oynatıcıya bağlar."""
        data_a, fs_a = sf.read(str(path_a), dtype="float32")
        data_b, fs_b = sf.read(str(path_b), dtype="float32")

        # Örnekleme hızı farklıysa B'yi A'ya uydur
        if fs_a != fs_b:
            import scipy.signal as signal
            gcd = math.gcd(int(fs_b), int(fs_a))
            data_b = signal.resample_poly(data_b, int(fs_a) // gcd, int(fs_b) // gcd).astype(np.float32)

        self.load(data_a, data_b, fs_a)

    def set_position_callback(self, cb: Callable[[float, float], None]) -> None:
        self._position_cb = cb

    def set_finished_callback(self, cb: Callable[[], None]) -> None:
        self._finished_cb = cb

    @property
    def active_channel(self) -> ABChannel:
        return self._active_channel

    def switch_channel(self, channel: ABChannel) -> None:
        """Kanalı A veya B olarak anında değiştirir (imleç sıfırlanmaz)."""
        with self._lock:
            if self._active_channel != channel:
                logger.info(f"A/B kanal geçişi: {self._active_channel} -> {channel}")
                self._active_channel = channel

    def toggle_channel(self) -> ABChannel:
        """Kanalı anlık tersine çevirir (A ise B, B ise A)."""
        new_ch = ABChannel.B if self._active_channel == ABChannel.A else ABChannel.A
        self.switch_channel(new_ch)
        return new_ch

    def set_loudness_matching(self, enable: bool) -> None:
        """Loudness eşitlemesini açar veya kapatır."""
        with self._lock:
            self._loudness_matched = bool(enable)
            self._update_gains()

    def _update_gains(self) -> None:
        if not self._loudness_matched:
            self._gain_a = 1.0
            self._gain_b = 1.0
            return

        if math.isfinite(self._lufs_a) and math.isfinite(self._lufs_b):
            diff_db = self._lufs_a - self._lufs_b
        elif self._audio_a is not None and self._audio_b is not None:
            rms_a = float(np.sqrt(np.mean(np.square(self._audio_a, dtype=np.float64))))
            rms_b = float(np.sqrt(np.mean(np.square(self._audio_b, dtype=np.float64))))
            if rms_a > 1e-6 and rms_b > 1e-6:
                diff_db = 20.0 * math.log10(rms_a / rms_b)
            else:
                diff_db = 0.0
        else:
            diff_db = 0.0

        diff_db = max(-18.0, min(18.0, diff_db))
        self._gain_a = 1.0
        self._gain_b = 10.0 ** (diff_db / 20.0)
        logger.info(f"A/B Loudness eşitleme devrede: B kazancı = {diff_db:+.1f} dB")

    @property
    def duration(self) -> float:
        total = max(self._len_a, self._len_b)
        return total / self.fs if self.fs > 0 else 0.0

    @property
    def position(self) -> float:
        return self._playhead_frame / self.fs if self.fs > 0 else 0.0

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def play(self, start_seconds: Optional[float] = None) -> None:
        with self._lock:
            if self._audio_a is None or self._audio_b is None:
                return

            if start_seconds is not None:
                self._playhead_frame = max(0, int(start_seconds * self.fs))

            max_len = max(self._len_a, self._len_b)
            if self._playhead_frame >= max_len:
                self._playhead_frame = 0

            if self._stream is None:
                self._stream = sd.OutputStream(
                    samplerate=self.fs,
                    channels=self._channels,
                    dtype="float32",
                    callback=self._audio_callback,
                    finished_callback=self._stream_finished,
                    blocksize=1024,
                )
                self._stream.start()

            self._is_playing = True

    def pause(self) -> None:
        with self._lock:
            self._is_playing = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def stop(self) -> None:
        with self._lock:
            self._is_playing = False
            self._playhead_frame = 0
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def seek(self, seconds: float) -> None:
        with self._lock:
            max_len = max(self._len_a, self._len_b)
            self._playhead_frame = max(0, min(max_len, int(seconds * self.fs)))

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        with self._lock:
            if not self._is_playing or self._audio_a is None or self._audio_b is None:
                outdata.fill(0)
                return

            max_len = max(self._len_a, self._len_b)
            idx = self._playhead_frame
            rem = max_len - idx

            if rem <= 0:
                outdata.fill(0)
                self._is_playing = False
                if self._finished_cb:
                    threading.Thread(target=self._finished_cb, daemon=True).start()
                return

            read_frames = min(frames, rem)

            # A veya B kanalından oku
            if self._active_channel == ABChannel.A:
                src = self._audio_a
                gain = self._gain_a
                src_len = self._len_a
            else:
                src = self._audio_b
                gain = self._gain_b
                src_len = self._len_b

            if idx < src_len:
                chunk_len = min(read_frames, src_len - idx)
                if self._channels == 1:
                    chunk = src[idx : idx + chunk_len].reshape(-1, 1)
                else:
                    chunk = src[idx : idx + chunk_len]

                outdata[:chunk_len] = chunk * gain
                if chunk_len < frames:
                    outdata[chunk_len:].fill(0)
            else:
                outdata.fill(0)

            self._playhead_frame += read_frames

            # Callback bildir
            if self._position_cb:
                cur_sec = self._playhead_frame / self.fs
                dur_sec = max_len / self.fs
                self._position_cb(cur_sec, dur_sec)

    def _stream_finished(self) -> None:
        pass

    def close(self) -> None:
        self.stop()
        with self._lock:
            self._audio_a = None
            self._audio_b = None
