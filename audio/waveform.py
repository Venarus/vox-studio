"""
Waveform tepe (peak) özetleyici.

Kayıt sürerken her blok için min/max tepe değerleri biriktirilir. Böylece
saatlerce süren kayıtlar için bile tam çözünürlüklü örnek tutmadan waveform
çizilebilir (bellek: ~6 MB / saat).
"""

from __future__ import annotations

from array import array
from typing import Tuple

import numpy as np

from core.constants import WAVEFORM_PEAK_BUCKET


class WaveformPeaks:
    def __init__(self, bucket_frames: int = WAVEFORM_PEAK_BUCKET, channels: int = 1):
        if bucket_frames <= 0:
            raise ValueError("bucket_frames pozitif olmalı")
        self.bucket_frames = int(bucket_frames)
        self.channels = int(channels)
        self._mins = array("f")
        self._maxs = array("f")
        self._cur_min = 0.0
        self._cur_max = 0.0
        self._cur_n = 0
        self._frames = 0

    # ------------------------------------------------------------------ ekle
    def add(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32)
        if block.size == 0:
            return
        if block.ndim == 1:
            mono = block
        else:
            mono = block[:, 0] if block.shape[1] == 1 else block.mean(axis=1)

        self._frames += mono.shape[0]
        i = 0
        n = mono.shape[0]
        while i < n:
            take = min(self.bucket_frames - self._cur_n, n - i)
            seg = mono[i:i + take]
            seg_min = float(seg.min())
            seg_max = float(seg.max())
            if self._cur_n == 0:
                self._cur_min = seg_min
                self._cur_max = seg_max
            else:
                if seg_min < self._cur_min:
                    self._cur_min = seg_min
                if seg_max > self._cur_max:
                    self._cur_max = seg_max
            self._cur_n += take
            i += take
            if self._cur_n >= self.bucket_frames:
                self._mins.append(self._cur_min)
                self._maxs.append(self._cur_max)
                self._cur_min = 0.0
                self._cur_max = 0.0
                self._cur_n = 0

    def flush_partial(self) -> None:
        """Yarım kalan kovayı yayımlar (kayıt durdurulurken)."""
        if self._cur_n > 0:
            self._mins.append(self._cur_min)
            self._maxs.append(self._cur_max)
            self._cur_min = 0.0
            self._cur_max = 0.0
            self._cur_n = 0

    # ------------------------------------------------------------------ okuma
    @property
    def frames(self) -> int:
        return self._frames

    @property
    def bucket_count(self) -> int:
        return len(self._mins)

    def extent(self) -> Tuple[np.ndarray, np.ndarray]:
        mins = np.frombuffer(self._mins, dtype=np.float32) if len(self._mins) else np.zeros(0, np.float32)
        maxs = np.frombuffer(self._maxs, dtype=np.float32) if len(self._maxs) else np.zeros(0, np.float32)
        return np.array(mins, copy=True), np.array(maxs, copy=True)

    def view(self, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Grafik için en fazla ``max_points`` noktaya indirger."""
        mins, maxs = self.extent()
        count = mins.shape[0]
        if count == 0 or max_points <= 0 or count <= max_points:
            return mins, maxs
        group = int(np.ceil(count / max_points))
        starts = np.arange(0, count, group)
        vmin = np.minimum.reduceat(mins, starts)
        vmax = np.maximum.reduceat(maxs, starts)
        return vmin, vmax

    def reset(self) -> None:
        del self._mins[:]
        del self._maxs[:]
        self._cur_min = 0.0
        self._cur_max = 0.0
        self._cur_n = 0
        self._frames = 0
