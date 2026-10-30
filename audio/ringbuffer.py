"""
Gerçek zamanlı ses için halka tampon (ring buffer).

Tasarım kuralları
-----------------
* Ses callback'i **bellek ayırmaz**; yalnızca önceden ayrılmış diziye kopyalar.
* Tek üretici (audio callback) / tek tüketici (yazıcı veya okuyucu iş parçacığı)
  varsayımı vardır.
* Kilit yalnızca sayaç güncellemesi için, çok kısa süreli tutulur; numpy
  kopyalama kilidin dışında yapılır.
* Taşma (overflow) ve boşta kalma (underrun) sayaçları raporlanır; sessizce
  veri kaybı yaşanmaz.
* Üretici, veriyi kopyaladıktan sonra "yayımlar". Tüketici de veriyi
  kopyaladıktan sonra konumunu ilerletir. Böylece üzerine yazma yarışı olmaz.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """float32, (frames, channels) biçiminde halka tampon."""

    def __init__(self, capacity_frames: int, channels: int = 1):
        if capacity_frames <= 0:
            raise ValueError("capacity_frames pozitif olmalı")
        if channels <= 0:
            raise ValueError("channels pozitif olmalı")
        self._cap = int(capacity_frames)
        self._channels = int(channels)
        self._buf = np.zeros((self._cap, self._channels), dtype=np.float32)
        self._lock = threading.Lock()
        self._reserved = 0    # üreticinin ayırdığı (kopyalanmakta olan) toplam
        self._published = 0   # tüketicinin görebildiği toplam
        self._read = 0        # tüketilmiş toplam
        self.overflow_frames = 0
        self.underrun_frames = 0

    # ------------------------------------------------------------------ meta
    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def channels(self) -> int:
        return self._channels

    def available_read(self) -> int:
        with self._lock:
            return self._published - self._read

    def free_space(self) -> int:
        with self._lock:
            return self._cap - (self._reserved - self._read)

    def clear(self) -> None:
        with self._lock:
            self._reserved = 0
            self._published = 0
            self._read = 0
            self.overflow_frames = 0
            self.underrun_frames = 0

    # -------------------------------------------------------------- üretici
    def write(self, data: np.ndarray) -> int:
        """Veriyi tampona yazar; yazılan frame sayısını döndürür.

        Yer yoksa hiçbir şey yazılmaz (kısmi yazma yok) ve
        ``overflow_frames`` artırılır; böylece kanal düzeni bozulmaz.
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim != 2 or data.shape[1] != self._channels:
            raise ValueError(f"beklenen (n, {self._channels}) veri, gelen {data.shape}")
        n = data.shape[0]
        if n == 0:
            return 0

        with self._lock:
            free = self._cap - (self._reserved - self._read)
            if free < n:
                self.overflow_frames += n
                return 0
            start = self._reserved % self._cap
            self._reserved += n

        first = min(n, self._cap - start)
        self._buf[start:start + first] = data[:first]
        if n > first:
            self._buf[: n - first] = data[first:]

        with self._lock:
            self._published = self._reserved
        return n

    # -------------------------------------------------------------- tüketici
    def read_into(self, out: np.ndarray) -> int:
        """``out`` (n, kanal) dizisini doldurur; okunan frame sayısını döndürür."""
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if out.ndim != 2 or out.shape[1] != self._channels:
            raise ValueError(f"beklenen (n, {self._channels}) hedef, gelen {out.shape}")

        with self._lock:
            avail = self._published - self._read
            n = min(avail, out.shape[0])
            if n <= 0:
                self.underrun_frames += out.shape[0]
                return 0
            start = self._read % self._cap

        first = min(n, self._cap - start)
        out[:first] = self._buf[start:start + first]
        if n > first:
            out[first:n] = self._buf[: n - first]

        with self._lock:
            self._read += n
        return n
