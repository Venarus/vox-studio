"""
VOX Studio — DSP Filtreleri ve Parametrik EQ
Robert Bristow-Johnson Audio EQ Cookbook biquad filtreleri (SOS formatı).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class EQBand:
    band_type: str  # 'highpass', 'lowpass', 'peaking', 'highshelf', 'lowshelf', 'notch', 'bandpass'
    frequency: float  # Hz
    gain_db: float = 0.0  # dB (peaking and shelf only)
    q: float = 0.707  # Q factor
    enabled: bool = True


@dataclass
class HumFilterSettings:
    base_freq_hz: float = 50.0  # 50 Hz (Avrupa/TR) veya 60 Hz (ABD)
    harmonics: int = 3          # Kaç harmonik bastırılacak (1..5)
    q: float = 30.0             # Dar çentik (notch) keskinliği
    enabled: bool = False


def make_biquad(band: EQBand, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Audio EQ Cookbook biquad katsayıları üretir (b, a)."""
    f0 = np.clip(band.frequency, 10.0, fs * 0.499)
    w0 = 2.0 * math.pi * f0 / fs
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * max(band.q, 0.01))
    A = 10.0 ** (band.gain_db / 40.0)

    b = np.zeros(3, dtype=np.float64)
    a = np.zeros(3, dtype=np.float64)

    btype = band.band_type.lower()
    if btype == "highpass":
        b[0] = (1.0 + cos_w0) / 2.0
        b[1] = -(1.0 + cos_w0)
        b[2] = (1.0 + cos_w0) / 2.0
        a[0] = 1.0 + alpha
        a[1] = -2.0 * cos_w0
        a[2] = 1.0 - alpha
    elif btype == "lowpass":
        b[0] = (1.0 - cos_w0) / 2.0
        b[1] = 1.0 - cos_w0
        b[2] = (1.0 - cos_w0) / 2.0
        a[0] = 1.0 + alpha
        a[1] = -2.0 * cos_w0
        a[2] = 1.0 - alpha
    elif btype == "peaking":
        b[0] = 1.0 + alpha * A
        b[1] = -2.0 * cos_w0
        b[2] = 1.0 - alpha * A
        a[0] = 1.0 + alpha / A
        a[1] = -2.0 * cos_w0
        a[2] = 1.0 - alpha / A
    elif btype == "highshelf":
        sqrtA = math.sqrt(A)
        b[0] = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha)
        b[1] = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        b[2] = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha)
        a[0] = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha
        a[1] = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        a[2] = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha
    elif btype == "lowshelf":
        sqrtA = math.sqrt(A)
        b[0] = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha)
        b[1] = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        b[2] = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha)
        a[0] = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha
        a[1] = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        a[2] = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha
    elif btype == "notch":
        b[0] = 1.0
        b[1] = -2.0 * cos_w0
        b[2] = 1.0
        a[0] = 1.0 + alpha
        a[1] = -2.0 * cos_w0
        a[2] = 1.0 - alpha
    elif btype == "bandpass":
        b[0] = alpha
        b[1] = 0.0
        b[2] = -alpha
        a[0] = 1.0 + alpha
        a[1] = -2.0 * cos_w0
        a[2] = 1.0 - alpha
    else:
        # Bypass
        b[0] = 1.0
        a[0] = 1.0

    # a[0]'a göre normalizasyon
    a0 = a[0]
    b /= a0
    a /= a0
    return b, a


def apply_eq(audio: np.ndarray, fs: float, bands: List[EQBand]) -> np.ndarray:
    """Audio matrisine (1D mono veya 2D frames x channels) parametrik EQ uygular."""
    active_bands = [b for b in bands if b.enabled]
    if not active_bands:
        return audio.copy()

    sos_list = []
    for band in active_bands:
        b, a = make_biquad(band, fs)
        # scipy sos formatı: [b0, b1, b2, a0, a1, a2] where a0 is 1.0
        sos_list.append([b[0], b[1], b[2], 1.0, a[1], a[2]])

    sos = np.array(sos_list, dtype=np.float64)

    if audio.ndim == 1:
        filtered = signal.sosfilt(sos, audio.astype(np.float64))
        return filtered.astype(audio.dtype)
    else:
        # Çok kanallı: eksen 0 boyunca filtrele (Sol/Sağ stereo ayrımını korur)
        filtered = signal.sosfilt(sos, audio.astype(np.float64), axis=0)
        return filtered.astype(audio.dtype)


def apply_hum_filter(audio: np.ndarray, fs: float, settings: HumFilterSettings) -> np.ndarray:
    """50 Hz veya 60 Hz şebeke uğultusunu (hum) ve harmoniklerini dar çentik (notch) filtreyle temizler."""
    if not settings.enabled or settings.base_freq_hz <= 0:
        return audio.copy()
    bands: List[EQBand] = []
    n_harm = max(1, min(6, int(settings.harmonics)))
    for h in range(1, n_harm + 1):
        freq = settings.base_freq_hz * h
        if freq < fs * 0.48:
            bands.append(EQBand(band_type="notch", frequency=freq, q=max(5.0, settings.q), enabled=True))
    return apply_eq(audio, fs, bands)
