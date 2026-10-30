"""
VOX Studio — DSP Oda Yankısı Azaltma (Dereverberation) Motoru

Yankılı odalarda, boş salonlarda veya mikrofona uzak mesafeden yapılan kayıtlarda
oluşan geç oda yansımalarını (late reverberation) ve akustik kuyrukları bastırır.

Algoritma:
* STFT spektral güç tabanlı geç yansıma izleme (reverberant tail tracking).
* RT60 sönümlenme süresi ve ilk yansıma marjı (early reflection boundary ~30-50ms).
* Doğal oda ambiyansını tamamen yok etmeyip boğuklaşmayı önleyen spectral floor kontrolü.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.signal as signal

from core.logger import get_logger

logger = get_logger("dsp.dereverb")


@dataclass
class DereverbSettings:
    enabled: bool = True
    rt60: float = 0.45             # Tahmini sönümlenme süresi (0.1 .. 1.5 saniye)
    amount: float = 0.65           # Bastırma derinliği (0.0 .. 1.0)
    delay_ms: float = 40.0         # İlk yansıma gecikme koruması (ms)
    spectral_floor: float = 0.20   # Metalikliği önleyen asgari spektral taban (0.05 .. 0.5)


def dereverb_spectral(
    audio: np.ndarray,
    fs: float,
    settings: Optional[DereverbSettings] = None,
) -> np.ndarray:
    """
    Ses sinyaline spektral oda yankısı azaltma (Dereverberation) uygular.
    """
    if settings is None:
        settings = DereverbSettings()

    if not settings.enabled or settings.amount <= 0.01:
        return audio.copy()

    orig_dtype = audio.dtype
    is_1d = (audio.ndim == 1)
    channels = [audio] if is_1d else [audio[:, ch] for ch in range(audio.shape[1])]
    processed_channels = []

    # STFT parametreleri
    nperseg = 1024 if fs >= 32000 else 512
    hop = nperseg // 4
    window = signal.windows.hann(nperseg, sym=False)

    dt = hop / fs
    rt60 = max(0.08, float(settings.rt60))
    # Üstel sönümlenme katsayısı: gamma = exp(-6 * ln(10) * dt / RT60)
    gamma = math.exp(-6.0 * math.log(10.0) * dt / rt60)

    # İlk yansıma koruma gecikmesi (frame sayısı cinsinden)
    delay_frames = max(2, int((settings.delay_ms / 1000.0) * fs / hop))
    amount = max(0.0, min(1.0, float(settings.amount)))
    floor = max(0.05, min(0.6, float(settings.spectral_floor)))

    for ch_data in channels:
        ch_len = len(ch_data)
        if ch_len < nperseg:
            processed_channels.append(ch_data.copy())
            continue

        f, t_stft, Z = signal.stft(
            ch_data.astype(np.float32),
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=nperseg - hop,
            boundary="zeros",
            padded=True,
        )

        mag = np.abs(Z)
        phase = np.angle(Z)
        power = mag ** 2
        n_bins, n_frames = power.shape

        # Geç yansıma güç tahmini (R_late)
        r_late = np.zeros_like(power)
        for i in range(delay_frames, n_frames):
            past_frame = power[:, i - delay_frames]
            r_late[:, i] = (gamma * r_late[:, i - 1]) + ((1.0 - gamma) * past_frame)

        # Spektral zayıflatma kazancı: G = sqrt(max(1 - amount * R_late / (P + eps), 0))
        # Spectral floor ile sınırlanır
        gain_sq = 1.0 - (amount * (r_late / (power + 1e-10)))
        gain_sq = np.maximum(gain_sq, floor ** 2)
        gain = np.sqrt(gain_sq)

        Z_clean = (mag * gain) * np.exp(1j * phase)

        _, clean_ch = signal.istft(
            Z_clean,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=nperseg - hop,
            boundary=True,
        )

        if len(clean_ch) > ch_len:
            clean_ch = clean_ch[:ch_len]
        elif len(clean_ch) < ch_len:
            clean_ch = np.pad(clean_ch, (0, ch_len - len(clean_ch)))

        processed_channels.append(clean_ch)

    if is_1d:
        out = processed_channels[0]
    else:
        out = np.column_stack(processed_channels)

    return out.astype(orig_dtype)
