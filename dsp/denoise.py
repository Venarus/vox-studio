"""
VOX Studio — DSP Spektral Gürültü Azaltma
Doğal insan sesini koruyan, spektral tabanlı (oversubtraction + spectral floor) gürültü filtresi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class DenoiseSettings:
    amount_db: float = 12.0  # Azaltma miktarı (6-24 dB)
    noise_profile_duration_sec: float = 0.25  # Profil süresi
    spectral_floor: float = 0.05  # Müzikal artefaktları önleyen taban (0.01 - 0.2)
    enabled: bool = True
    reduction_db: Optional[float] = None

    def __post_init__(self):
        if self.reduction_db is not None:
            self.amount_db = float(self.reduction_db)


def reduce_noise_spectral(
    audio: np.ndarray,
    fs: float,
    settings: DenoiseSettings,
    custom_noise_profile: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    STFT tabanlı spektral çıkarma ile gürültü azaltma uygular.
    """
    if not settings.enabled or settings.amount_db <= 0.5:
        return audio.copy()

    orig_dtype = audio.dtype
    is_1d = (audio.ndim == 1)
    channels = [audio] if is_1d else [audio[:, ch] for ch in range(audio.shape[1])]
    processed_channels = []

    nperseg = 1024
    noverlap = nperseg // 2

    # İndirgeme faktörü
    reduction_linear = 10.0 ** (-settings.amount_db / 20.0)
    # Oversubtraction faktörü
    alpha = 1.0 + (settings.amount_db / 10.0)
    beta = max(settings.spectral_floor, 0.01)

    for ch_data in channels:
        ch_len = len(ch_data)
        if ch_len < nperseg:
            processed_channels.append(ch_data.copy())
            continue

        # STFT
        f, t, Zxx = signal.stft(ch_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Gürültü profilini belirle (varsayılan: baştaki n saniye)
        profile_frames = max(2, int((settings.noise_profile_duration_sec * fs) / noverlap))
        profile_frames = min(profile_frames, mag.shape[1])

        if custom_noise_profile is not None and len(custom_noise_profile) == mag.shape[0]:
            noise_est = np.asarray(custom_noise_profile).reshape(-1, 1)
        else:
            noise_est = np.mean(mag[:, :profile_frames], axis=1, keepdims=True)

        # Spektral çıkarma
        # Subtracted = max(Mag - alpha * Noise, beta * Mag)
        subtracted = mag - (alpha * noise_est)
        floor = beta * mag
        clean_mag = np.maximum(subtracted, floor)

        # iSTFT
        Zxx_clean = clean_mag * np.exp(1j * phase)
        _, clean_audio = signal.istft(Zxx_clean, fs=fs, nperseg=nperseg, noverlap=noverlap)

        # Orijinal uzunluğa getir
        if len(clean_audio) > ch_len:
            clean_audio = clean_audio[:ch_len]
        elif len(clean_audio) < ch_len:
            clean_audio = np.pad(clean_audio, (0, ch_len - len(clean_audio)))

        processed_channels.append(clean_audio)

    if is_1d:
        out = processed_channels[0]
    else:
        out = np.column_stack(processed_channels)
    return out.astype(orig_dtype)
