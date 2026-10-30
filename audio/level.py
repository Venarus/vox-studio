"""
Ses seviyesi ölçümü (peak / RMS / clipping).

İki katman vardır:

* ``push_peak`` — **ses callback'inden** çağrılır. Yalnızca iki indirgeme
  (``max``/``min``) yapar, bellek ayırmaz; ölçüm gecikmesiz görünür.
* ``push_block`` — yazıcı iş parçacığından çağrılır. Tam örnek bazlı clipping
  sayımı ve RMS toplamı burada yapılır; gerçek zamanlı yolu yavaşlatmaz.

Arayüz 25–30 Hz'de ``snapshot()`` çağırır; bu çağrı asla bloklamaz.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.constants import CLIP_THRESHOLD


def linear_to_db(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        return -math.inf
    return 20.0 * math.log10(value)


def db_to_linear(db: float) -> float:
    return 10.0 ** (float(db) / 20.0)


def format_db(value: float, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "-∞ dB"
    return f"{value:.{digits}f} dB"


@dataclass(frozen=True)
class LevelSnapshot:
    peak: float = 0.0
    peak_db: float = -math.inf
    rms: float = 0.0
    rms_db: float = -math.inf
    hold_peak: float = 0.0
    hold_db: float = -math.inf
    clip_count: int = 0
    clip_active: bool = False
    blocks: int = 0
    frames: int = 0


class LevelAnalyzer:
    """Gerçek zamanlı seviye ölçer."""

    def __init__(self, hold_seconds: float = 1.2, clip_threshold: float = CLIP_THRESHOLD):
        self._hold_seconds = float(hold_seconds)
        self._clip_threshold = float(clip_threshold)
        # callback tarafı (atomik skalerler)
        self._peak = 0.0
        self._hold = 0.0
        self._hold_time = 0.0
        # yazıcı tarafı
        self._sum_sq = 0.0
        self._rms_frames = 0
        self._rms = 0.0
        self._clip_samples = 0
        self._blocks = 0
        self._frames = 0
        self._last_clip_time = 0.0

    # ------------------------------------------------------------ callback
    def push_peak(self, block: np.ndarray) -> float:
        """Callback'ten çağrılır: bellek ayırmaz."""
        if block.size == 0:
            return 0.0
        sample_peak = float(max(-block.min(), block.max()))
        if sample_peak > self._peak:
            self._peak = sample_peak
        if sample_peak >= self._hold:
            self._hold = sample_peak
            self._hold_time = time.monotonic()
        if sample_peak >= self._clip_threshold:
            self._last_clip_time = time.monotonic()
        return sample_peak

    # ---------------------------------------------------------- yazıcı thread
    def push_block(self, block: np.ndarray) -> None:
        if block.size == 0:
            return
        absmax = np.abs(block)
        self._sum_sq += float(np.einsum("ij,ij->", absmax, absmax))
        self._rms_frames += block.shape[0]
        if self._rms_frames > 0:
            self._rms = math.sqrt(self._sum_sq / max(self._rms_frames, 1))
        clipped = int(np.count_nonzero(absmax >= self._clip_threshold))
        if clipped:
            self._clip_samples += clipped
            self._last_clip_time = time.monotonic()
        self._blocks += 1
        self._frames += block.shape[0]

    # -------------------------------------------------------------- snapshot
    def snapshot(self) -> LevelSnapshot:
        now = time.monotonic()
        peak = self._peak
        self._peak = 0.0
        if now - self._hold_time > self._hold_seconds:
            self._hold *= 0.85  # yumuşak düşüş
            if self._hold < 1e-6:
                self._hold = 0.0
        clip_active = (now - self._last_clip_time) < 2.0
        return LevelSnapshot(
            peak=peak,
            peak_db=linear_to_db(peak),
            rms=self._rms,
            rms_db=linear_to_db(self._rms),
            hold_peak=self._hold,
            hold_db=linear_to_db(self._hold),
            clip_count=self._clip_samples,
            clip_active=clip_active,
            blocks=self._blocks,
            frames=self._frames,
        )

    def reset(self) -> None:
        self._peak = 0.0
        self._hold = 0.0
        self._hold_time = 0.0
        self._sum_sq = 0.0
        self._rms_frames = 0
        self._rms = 0.0
        self._clip_samples = 0
        self._blocks = 0
        self._frames = 0
        self._last_clip_time = 0.0


class InputCheck:
    """Kayıt öncesi giriş kontrolü için basit karar yardımcısı.

    Donanım hakkında kesin iddialarda bulunmaz; yalnızca ölçülen seviyeye
    göre kullanıcıya uyarı üretir.
    """

    def __init__(self, low_dbfs: float = -45.0, hot_dbfs: float = -3.0):
        self.low_dbfs = low_dbfs
        self.hot_dbfs = hot_dbfs

    def evaluate(self, snap: LevelSnapshot) -> tuple[str, str]:
        """(durum, açıklama) döndürür. durum: silent | low | good | hot | clipping"""
        if snap.clip_count > 0 or snap.peak_db >= self.hot_dbfs:
            if snap.clip_count > 0:
                return (
                    "clipping",
                    "Giriş kırpılıyor: mikrofon kazancını veya Windows giriş seviyesini düşürün.",
                )
            return ("hot", "Giriş çok yüksek (hedef tepe ≈ -12 dBFS). Kazancı biraz düşürün.")
        if snap.peak_db == -math.inf or snap.peak_db < -60.0:
            return (
                "silent",
                "Sinyal algılanmıyor. Mikrofonun açık olduğundan, doğru cihazın seçildiğinden "
                "ve Windows gizlilik izinlerinin verildiğinden emin olun.",
            )
        if snap.peak_db < self.low_dbfs:
            return (
                "low",
                "Giriş seviyesi düşük. Mikrofona biraz daha yaklaşın veya kazancı artırın.",
            )
        return ("good", "Giriş seviyesi uygun.")
