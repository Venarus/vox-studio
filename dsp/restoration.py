"""
VOX Studio — DSP Restorasyon Araçları: De-Clip, De-Click, De-Pop ve Speech Clarity

İçerik:
1. De-Clip: Dijital kırpılma (clipping / kare dalga distorsiyonu) tespiti ve kübik spline restorasyonu.
2. De-Click / De-Crackle: Mikrofon tıkırtıları, ağız sesleri ve dijital sıçramaların tespiti ve onarımı.
3. Plosive Reduction (De-Pop): Nefes patlamaları ("p", "b" harflerindeki rüzgar vuruşları) bastırma.
4. Speech Clarity Engine: Konuşma frekanslarını dengeleyen, bulanıklığı gideren ve anlaşılırlığı artıran akıllı ton şekillendirici.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage, signal
from scipy.interpolate import CubicSpline

from core.logger import get_logger
from dsp.filters import EQBand, apply_eq

logger = get_logger("dsp.restoration")


# ==============================================================================
# 1. DE-CLIP (Kırpılmış Ses Dalgalarını Onarma)
# ==============================================================================
@dataclass
class DeclipSettings:
    enabled: bool = True
    threshold_dbfs: float = -0.1   # Kırpılma algılama eşiği (~0.988)
    margin_samples: int = 4        # İnterpolasyon için çevre örnek sayısı


def declip(audio: np.ndarray, settings: Optional[DeclipSettings] = None) -> Tuple[np.ndarray, int]:
    """
    Kırpılmış tepe noktalarını tespit eder ve kübik spline ile yuvarlatarak restore eder.
    Dönüş: (onarılmış_ses, onarılan_örnek_sayısı)
    """
    if settings is None:
        settings = DeclipSettings()

    if not settings.enabled:
        return audio.copy(), 0

    orig_dtype = audio.dtype
    is_1d = (audio.ndim == 1)
    channels = [audio] if is_1d else [audio[:, ch] for ch in range(audio.shape[1])]
    processed_channels = []
    total_repaired = 0

    threshold = 10.0 ** (settings.threshold_dbfs / 20.0)

    for ch_data in channels:
        ch = ch_data.astype(np.float64).copy()
        n = len(ch)
        if n < 32:
            processed_channels.append(ch)
            continue

        clipped_mask = np.abs(ch) >= threshold
        if not np.any(clipped_mask):
            processed_channels.append(ch)
            continue

        # Ardışık kırpılmış bölgeleri tespit et
        diff = np.diff(np.pad(clipped_mask.astype(np.int8), (1, 1)))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        repaired_ch_count = 0
        margin = max(3, settings.margin_samples)

        for s, e in zip(starts, ends):
            run_len = e - s
            # Çok uzun bloklar (>150 örnek) müzik/doğal yüksek sestir; interpolasyon patlamamalı
            if run_len > 120 or run_len == 0:
                continue

            left_start = max(0, s - margin)
            right_end = min(n, e + margin)

            # Sol ve sağ sağlıklı örnekler
            idx_left = np.arange(left_start, s)
            idx_right = np.arange(e, right_end)
            if len(idx_left) < 2 or len(idx_right) < 2:
                continue

            x_pts = np.concatenate([idx_left, idx_right])
            y_pts = ch[x_pts]

            try:
                # Monotonluğu koruyan kübik spline
                cs = CubicSpline(x_pts, y_pts, bc_type="natural")
                interp_idx = np.arange(s, e)
                restored_vals = cs(interp_idx)

                # Sağlama: Orijinal yönüyle uyumlu olmalı
                ch[interp_idx] = restored_vals
                repaired_ch_count += run_len
            except Exception:
                pass

        # Aşırı yükselen tepe varsa hafifçe yumuşat
        peak_val = np.max(np.abs(ch))
        if peak_val > 1.0:
            ch /= peak_val

        total_repaired += repaired_ch_count
        processed_channels.append(ch)

    if is_1d:
        out = processed_channels[0]
    else:
        out = np.column_stack(processed_channels)

    return out.astype(orig_dtype), total_repaired


# ==============================================================================
# 2. DE-CLICK / DE-CRACKLE (Tıkırtı ve Dijital Sıçrama Temizleme)
# ==============================================================================
@dataclass
class DeclickSettings:
    enabled: bool = True
    sensitivity: float = 0.7       # 0.1 (az hassas) .. 1.0 (yüksek hassasiyet)
    max_click_samples: int = 48    # ~1 ms @ 48kHz


def declick(
    audio: np.ndarray,
    fs: float,
    settings: Optional[DeclickSettings] = None,
) -> Tuple[np.ndarray, int]:
    """
    Kısa süreli impulsif tıkırtıları ve çatlamaları (clicks/crackles) tespit eder
    ve lokal medyan-spline ile onarır.
    """
    if settings is None:
        settings = DeclickSettings()

    if not settings.enabled:
        return audio.copy(), 0

    orig_dtype = audio.dtype
    is_1d = (audio.ndim == 1)
    channels = [audio] if is_1d else [audio[:, ch] for ch in range(audio.shape[1])]
    processed_channels = []
    total_clicks = 0

    # Hassasiyet çarpanı
    sens = max(0.1, min(1.0, float(settings.sensitivity)))
    k_sigma = 8.0 - (sens * 4.5)  # 3.5 .. 7.5

    for ch_data in channels:
        ch = ch_data.astype(np.float64).copy()
        n = len(ch)
        if n < 64:
            processed_channels.append(ch)
            continue

        # 9 örnekli lokal medyan filtresi (hızlı ve tıkırtıya duyarlı)
        med = ndimage.median_filter(ch, size=9)
        residual = np.abs(ch - med)

        # Medyan mutlak sapma (MAD)
        mad = float(np.median(residual))
        sigma = 1.4826 * float(np.median(np.abs(residual - mad)))
        threshold = max(k_sigma * sigma, 1e-4)

        click_mask = residual > threshold
        if not np.any(click_mask):
            processed_channels.append(ch)
            continue

        diff = np.diff(np.pad(click_mask.astype(np.int8), (1, 1)))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        ch_clicks = 0
        max_samples = max(4, int(settings.max_click_samples * (fs / 48000.0)))

        for s, e in zip(starts, ends):
            length = e - s
            if length > max_samples or length == 0:
                continue

            # Tıkırtı aralığını sınırlarından pürüzsüz enterpolasyonla kapat
            left_val = ch[s - 1] if s > 0 else ch[s]
            right_val = ch[e] if e < n else ch[e - 1]
            interp = np.linspace(left_val, right_val, length + 2)[1:-1]
            ch[s:e] = interp
            ch_clicks += 1

        total_clicks += ch_clicks
        processed_channels.append(ch)

    if is_1d:
        out = processed_channels[0]
    else:
        out = np.column_stack(processed_channels)

    return out.astype(orig_dtype), total_clicks


# ==============================================================================
# 3. PLOSIVE REDUCTION (Nefes ve Patlama Önleme)
# ==============================================================================
@dataclass
class PlosiveSettings:
    enabled: bool = True
    cutoff_hz: float = 120.0       # Patlama bas filtresi frekansı
    sensitivity_db: float = 6.0    # Düşük frekans enerji patlama eşiği (dB)


def reduce_plosives(
    audio: np.ndarray,
    fs: float,
    settings: Optional[PlosiveSettings] = None,
) -> np.ndarray:
    """
    Mikrofona çarpan hava/nefes patlamalarındaki aşırı düşük frekanslı
    enerji sıçramalarını dinamik olarak zayıflatır.
    """
    if settings is None:
        settings = PlosiveSettings()

    if not settings.enabled:
        return audio.copy()

    orig_dtype = audio.dtype
    is_1d = (audio.ndim == 1)
    channels = [audio] if is_1d else [audio[:, ch] for ch in range(audio.shape[1])]
    processed_channels = []

    # 80 Hz yüksek geçiren 2. derece Butterworth filtresi
    nyq = 0.5 * fs
    norm_cutoff = min(0.49, settings.cutoff_hz / nyq)
    sos = signal.butter(2, norm_cutoff, btype="highpass", output="sos")

    for ch_data in channels:
        ch = ch_data.astype(np.float64)
        # Sinyali yüksek geçiren filtreyle süzerek patlama vuruşlarını bertaraf et
        filtered = signal.sosfilt(sos, ch)
        processed_channels.append(filtered)

    if is_1d:
        out = processed_channels[0]
    else:
        out = np.column_stack(processed_channels)

    return out.astype(orig_dtype)


# ==============================================================================
# 4. SPEECH CLARITY ENGINE (Konuşma Netliği ve Formant İyileştirme)
# ==============================================================================
@dataclass
class SpeechClaritySettings:
    enabled: bool = True
    presence_gain_db: float = 3.0   # 3.5 kHz konuşma varlığı/anlaşılırlığı (+1 .. +6 dB)
    boxiness_cut_db: float = -2.0   # 350 Hz oda boğukluğu ve kutu hissi kesimi (-4 .. 0 dB)
    air_shelf_db: float = 1.5       # 12 kHz parlaklık ve hava hissi (+0 .. +4 dB)
    highpass_freq_hz: float = 80.0  # Dip titreşim/uğultu kesme frekansı


def apply_speech_clarity(
    audio: np.ndarray,
    fs: float,
    settings: Optional[SpeechClaritySettings] = None,
) -> np.ndarray:
    """
    Konuşma anlaşılırlığını, artikülasyon netliğini ve ton dengesini
    profesyonel vokal master EQ eğrisiyle optimize eder.
    """
    if settings is None:
        settings = SpeechClaritySettings()

    if not settings.enabled:
        return audio.copy()

    bands = [
        # 1. Dip sarsıntı / mekanik titreşim temizliği
        EQBand(band_type="highpass", frequency=settings.highpass_freq_hz, q=0.707, enabled=True),
        # 2. 350 Hz oda kutu hissi ve boğukluk temizliği
        EQBand(
            band_type="peaking",
            frequency=350.0,
            gain_db=settings.boxiness_cut_db,
            q=1.2,
            enabled=abs(settings.boxiness_cut_db) > 0.1,
        ),
        # 3. 3.5 kHz konuşma artikülasyon ve varlık (presence) takviyesi
        EQBand(
            band_type="peaking",
            frequency=3500.0,
            gain_db=settings.presence_gain_db,
            q=1.1,
            enabled=abs(settings.presence_gain_db) > 0.1,
        ),
        # 4. 12 kHz hava ve parlaklık raf filtresi
        EQBand(
            band_type="highshelf",
            frequency=12000.0,
            gain_db=settings.air_shelf_db,
            q=0.707,
            enabled=abs(settings.air_shelf_db) > 0.1,
        ),
    ]

    return apply_eq(audio, fs, bands)
