"""
VOX Studio — DSP Dinamik İşlemciler (Compressor & Limiter)
Düşük gecikmeli, soft-knee RMS kompresör ve lookahead brickwall limiter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CompressorSettings:
    threshold_db: float = -18.0  # dBFS
    ratio: float = 3.0  # 1.0 - 20.0
    attack_ms: float = 10.0  # ms
    release_ms: float = 100.0  # ms
    knee_db: float = 4.0  # soft knee genişliği dB
    makeup_gain_db: float = 0.0  # dB
    enabled: bool = True


@dataclass
class LimiterSettings:
    ceiling_db: float = -0.5  # dBFS tavan
    threshold_db: float = -1.0  # tetiklenme eşiği dBFS
    release_ms: float = 50.0  # ms
    enabled: bool = True
    ceiling_dbfs: Optional[float] = None

    def __post_init__(self):
        if self.ceiling_dbfs is not None:
            self.ceiling_db = float(self.ceiling_dbfs)


@dataclass
class DeEsserSettings:
    frequency_hz: float = 6500.0  # Sibilans merkez frekansı (4000 - 10000 Hz)
    threshold_db: float = -24.0   # Tetiklenme eşiği dBFS
    max_reduction_db: float = 6.0 # Maksimum bastırma (0 - 18 dB)
    q: float = 2.0
    enabled: bool = False
    center_freq_hz: Optional[float] = None
    reduction_db: Optional[float] = None

    def __post_init__(self):
        if self.center_freq_hz is not None:
            self.frequency_hz = float(self.center_freq_hz)
        if self.reduction_db is not None:
            self.max_reduction_db = float(self.reduction_db)


def apply_compressor(
    audio: np.ndarray,
    fs: float,
    settings: CompressorSettings,
) -> Tuple[np.ndarray, float]:
    """
    Ses sinyaline soft-knee kompresör uygular.
    Dönüş: (işlenmiş_ses, maksimum_gain_reduction_dB)
    """
    if not settings.enabled or settings.ratio <= 1.0:
        return audio.copy(), 0.0

    orig_dtype = audio.dtype
    is_1d = (audio.ndim == 1)
    data = audio.reshape(-1, 1) if is_1d else audio.copy()
    n_samples, n_channels = data.shape

    # Katsayılar
    alpha_attack = math.exp(-1.0 / (max(settings.attack_ms, 0.1) * 0.001 * fs))
    alpha_release = math.exp(-1.0 / (max(settings.release_ms, 1.0) * 0.001 * fs))
    t = settings.threshold_db
    r = settings.ratio
    w = max(settings.knee_db, 0.0)

    # Dedektör için kanal ortalaması büyüklüğü
    detector_in = np.max(np.abs(data), axis=1)
    env = 0.0
    gains = np.ones(n_samples, dtype=np.float64)

    max_gr = 0.0

    for i in range(n_samples):
        x = float(detector_in[i])
        # Logaritmik seviye
        x_db = 20.0 * math.log10(x) if x > 1e-6 else -120.0

        # Statik karakteristik hesaplama (gain computer with soft knee)
        if 2.0 * (x_db - t) < -w:
            y_db = x_db
        elif 2.0 * abs(x_db - t) <= w:
            y_db = x_db + (1.0 / r - 1.0) * ((x_db - t + w / 2.0) ** 2) / (2.0 * w)
        else:
            y_db = t + (x_db - t) / r

        control_db = y_db - x_db  # <= 0

        # Zarf izleyici (ballistics)
        target = control_db
        if target < env:
            env = alpha_attack * env + (1.0 - alpha_attack) * target
        else:
            env = alpha_release * env + (1.0 - alpha_release) * target

        gr = -env
        if gr > max_gr:
            max_gr = gr

        gains[i] = 10.0 ** (env / 20.0)

    # Makeup gain uygula
    makeup = 10.0 ** (settings.makeup_gain_db / 20.0)
    out = data * gains[:, np.newaxis] * makeup

    if is_1d:
        out = out.reshape(-1)
    return out.astype(orig_dtype), max_gr


def apply_limiter(
    audio: np.ndarray,
    fs: float,
    settings: LimiterSettings,
) -> np.ndarray:
    """
    Brickwall / Peak Limiter.
    Ceiling seviyesini kesinlikle aşmayacak şekilde tepe kontrolü ve soft limiting yapar.
    """
    if not settings.enabled:
        return audio.copy()

    orig_dtype = audio.dtype
    is_1d = (audio.ndim == 1)
    data = audio.reshape(-1, 1) if is_1d else audio.copy()
    n_samples, n_channels = data.shape

    ceiling = 10.0 ** (settings.ceiling_db / 20.0)
    threshold = 10.0 ** (settings.threshold_db / 20.0)
    alpha_rel = math.exp(-1.0 / (max(settings.release_ms, 1.0) * 0.001 * fs))

    # Lookahead (3 ms)
    lookahead_samples = int(fs * 0.003)
    padded = np.pad(data, ((0, lookahead_samples), (0, 0)), mode="edge")

    env = 1.0
    gains = np.ones(n_samples, dtype=np.float64)

    detector = np.max(np.abs(padded), axis=1)

    for i in range(n_samples):
        # Lookahead penceresindeki tepe
        window_peak = np.max(detector[i : i + lookahead_samples + 1])
        if window_peak > threshold:
            target_gain = min(1.0, ceiling / max(window_peak, 1e-6))
        else:
            target_gain = 1.0

        if target_gain < env:
            env = target_gain  # ani düşüş (zero attack)
        else:
            env = alpha_rel * env + (1.0 - alpha_rel) * target_gain

        gains[i] = env

    out = data * gains[:, np.newaxis]

    # Kesin tepe koruması (tavanı aşmayı engelle)
    np.clip(out, -ceiling, ceiling, out=out)

    if is_1d:
        out = out.reshape(-1)
    return out.astype(orig_dtype)


def apply_deesser(
    audio: np.ndarray,
    fs: float,
    settings: DeEsserSettings,
) -> Tuple[np.ndarray, float]:
    """
    Sert 's/ş/ç' (sibilans) frekanslarını algılayıp yalnızca sibilans bandını
    dinamik olarak bastıran split-band De-Esser.
    Dönüş: (işlenmiş_ses, maksimum_bastırma_dB)
    """
    if not settings.enabled or settings.max_reduction_db <= 0.1:
        return audio.copy(), 0.0

    from dsp.filters import EQBand, apply_eq

    orig_dtype = audio.dtype
    is_1d = audio.ndim == 1
    data = audio.reshape(-1, 1).astype(np.float64) if is_1d else audio.astype(np.float64).copy()

    # Sibilans bandını ayır (bandpass)
    sib_band = apply_eq(
        data,
        fs,
        [EQBand(band_type="bandpass", frequency=settings.frequency_hz, q=max(0.7, settings.q), enabled=True)],
    )
    low_band = data - sib_band

    sib_env = np.max(np.abs(sib_band), axis=1)
    thresh_lin = 10.0 ** (settings.threshold_db / 20.0)
    min_gain = 10.0 ** (-abs(settings.max_reduction_db) / 20.0)

    # Yumuşak zarf (10 ms pencere ortalaması)
    win = max(1, int(fs * 0.008))
    if win > 1 and len(sib_env) > win:
        kernel = np.ones(win, dtype=np.float64) / win
        smooth_env = np.convolve(sib_env, kernel, mode="same")
    else:
        smooth_env = sib_env

    gains = np.ones_like(smooth_env)
    over = smooth_env > thresh_lin
    if np.any(over):
        ratio_scale = thresh_lin / np.maximum(smooth_env[over], 1e-9)
        gains[over] = np.clip(ratio_scale, min_gain, 1.0)

    max_red_db = float(-20.0 * math.log10(max(float(np.min(gains)), 1e-6))) if len(gains) else 0.0
    out = low_band + (sib_band * gains[:, np.newaxis])

    if is_1d:
        out = out.reshape(-1)
    return out.astype(orig_dtype), round(max_red_db, 2)
