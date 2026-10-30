"""
VOX Studio — DSP Normalizasyon ve Kırpma Araçları
Peak normalizasyon, loudness normalizasyon ve sessizlik kırpma.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def normalize_peak(
    audio: np.ndarray,
    target_dbfs: float = -1.0,
) -> Tuple[np.ndarray, float]:
    """
    Sinyali hedef tepe dBFS değerine normalize eder.
    Dönüş: (normalize_ses, uygulanan_kazanc_dB)
    """
    current_peak = float(np.max(np.abs(audio)))
    if current_peak < 1e-6:
        return audio.copy(), 0.0

    target_linear = 10.0 ** (target_dbfs / 20.0)
    gain_linear = target_linear / current_peak
    gain_db = 20.0 * math.log10(gain_linear)

    normalized = audio * gain_linear
    return normalized.astype(audio.dtype), gain_db


def trim_silence(
    audio: np.ndarray,
    fs: float,
    threshold_dbfs: float = -50.0,
    margin_sec: float = 0.05,
) -> Tuple[np.ndarray, int, int]:
    """
    Sesin başındaki ve sonundaki sessizliği kırpar.
    Dönüş: (kırpılmış_ses, başlangıç_kesim_örnekleri, bitiş_kesim_örnekleri)
    """
    threshold_linear = 10.0 ** (threshold_dbfs / 20.0)
    mag = np.abs(audio) if audio.ndim == 1 else np.max(np.abs(audio), axis=1)

    above = np.where(mag > threshold_linear)[0]
    if len(above) == 0:
        return audio.copy(), 0, 0

    margin_samples = int(margin_sec * fs)
    start_idx = max(0, above[0] - margin_samples)
    end_idx = min(len(audio), above[-1] + margin_samples + 1)

    trimmed = audio[start_idx:end_idx]
    return trimmed.copy(), start_idx, len(audio) - end_idx


def apply_gain_pan_fade(
    audio: np.ndarray,
    fs: float,
    gain_db: float = 0.0,
    pan: float = 0.0,
    fade_in_sec: float = 0.0,
    fade_out_sec: float = 0.0,
) -> np.ndarray:
    """
    Ses sinyaline kazanç (dB), stereo Pan (-1.0 sol .. +1.0 sağ) ve
    yumuşak Fade-In / Fade-Out zarfı uygular. Stereo kanalları asla mono'ya düşürmez.
    """
    orig_dtype = audio.dtype
    out = audio.astype(np.float64).copy()
    if len(out) == 0:
        return audio.copy()

    if abs(gain_db) > 1e-4:
        out *= 10.0 ** (gain_db / 20.0)

    # Stereo Pan kontrolü (sabit güç - constant power pan law)
    if out.ndim == 2 and out.shape[1] >= 2 and abs(pan) > 1e-4:
        p = max(-1.0, min(1.0, float(pan)))
        angle = (p + 1.0) * (math.pi / 4.0)  # 0 .. pi/2
        gain_l = math.cos(angle) * math.sqrt(2.0)
        gain_r = math.sin(angle) * math.sqrt(2.0)
        out[:, 0] *= gain_l
        out[:, 1] *= gain_r

    n_samples = out.shape[0]
    if fade_in_sec > 0.005 and fs > 0:
        fi_samples = min(n_samples, max(1, int(fade_in_sec * fs)))
        curve_in = np.linspace(0.0, 1.0, fi_samples, dtype=np.float64)
        if out.ndim == 1:
            out[:fi_samples] *= curve_in
        else:
            out[:fi_samples, :] *= curve_in[:, np.newaxis]

    if fade_out_sec > 0.005 and fs > 0:
        fo_samples = min(n_samples, max(1, int(fade_out_sec * fs)))
        curve_out = np.linspace(1.0, 0.0, fo_samples, dtype=np.float64)
        if out.ndim == 1:
            out[-fo_samples:] *= curve_out
        else:
            out[-fo_samples:, :] *= curve_out[:, np.newaxis]

    return out.astype(orig_dtype)


def normalize_lufs(
    audio: np.ndarray,
    fs: float,
    target_lufs: float = -16.0,
) -> Tuple[np.ndarray, float]:
    """
    Sinyali ITU-R BS.1770-4 Entegre LUFS hedefine (örn. -16 LUFS dublaj, -14 LUFS yayın)
    göre normalize eder. Dönüş: (normalize_ses, uygulanan_kazanc_dB).
    """
    from dsp.analysis import analyze_array

    rep = analyze_array(audio.astype(np.float32, copy=False), int(fs))
    curr_lufs = rep.loudness_lufs
    if not math.isfinite(curr_lufs) or curr_lufs < -70.0:
        return audio.copy(), 0.0
    gain_db = max(-24.0, min(24.0, float(target_lufs) - float(curr_lufs)))
    gain_lin = 10.0 ** (gain_db / 20.0)
    out = (audio.astype(np.float64) * gain_lin).astype(audio.dtype)
    return out, round(gain_db, 2)
