"""
VOX Studio — DSP Modülleri Testi
"""

import numpy as np
import pytest

from dsp.chain import DSPProcessor, ProcessingChainConfig
from dsp.denoise import DenoiseSettings, reduce_noise_spectral
from dsp.dynamics import CompressorSettings, LimiterSettings, apply_compressor, apply_limiter
from dsp.filters import EQBand, apply_eq
from dsp.normalize import normalize_peak, trim_silence


def test_eq_highpass_and_peaking():
    fs = 48000.0
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    # 50 Hz sinyal (sub-bass) + 1000 Hz sinyal
    low_freq = np.sin(2 * np.pi * 50 * t)
    mid_freq = np.sin(2 * np.pi * 1000 * t)
    signal_in = (low_freq + mid_freq).astype(np.float32)

    # 120 Hz High-pass filtresi
    bands = [EQBand(band_type="highpass", frequency=120.0, enabled=True)]
    filtered = apply_eq(signal_in, fs, bands)

    # 50 Hz bileşeni belirgin şekilde zayıflamalı
    assert np.max(np.abs(filtered)) < np.max(np.abs(signal_in))


def test_compressor_gain_reduction():
    fs = 48000.0
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    # Yüksek seviyeli sinüs (0 dBFS civarı tepe)
    signal_in = (0.95 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    settings = CompressorSettings(
        threshold_db=-20.0,
        ratio=4.0,
        attack_ms=5.0,
        release_ms=50.0,
        makeup_gain_db=0.0,
    )
    compressed, max_gr = apply_compressor(signal_in, fs, settings)

    assert max_gr > 0.0
    assert np.max(np.abs(compressed)) < np.max(np.abs(signal_in))


def test_limiter_ceiling_protection():
    fs = 48000.0
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    # Çok yüksek genlikli aşırı sinyal (1.5 peak = +3.5 dBFS)
    overshoot = (1.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    settings = LimiterSettings(ceiling_db=-1.0, threshold_db=-2.0)
    limited = apply_limiter(overshoot, fs, settings)

    ceiling_linear = 10.0 ** (-1.0 / 20.0)
    # Tavanı hiçbir örnek aşmamalı (+ tolerans)
    assert np.max(np.abs(limited)) <= ceiling_linear + 1e-4


def test_denoise_reduces_noise_floor():
    fs = 48000.0
    np.random.seed(42)
    # Beyaz gürültü
    noise = (0.05 * np.random.randn(int(fs * 0.5))).astype(np.float32)

    settings = DenoiseSettings(amount_db=15.0, noise_profile_duration_sec=0.1)
    denoised = reduce_noise_spectral(noise, fs, settings)

    # Gürültü gücü belirgin şekilde düşmeli
    power_orig = np.mean(noise ** 2)
    power_denoised = np.mean(denoised ** 2)
    assert power_denoised < power_orig


def test_normalize_peak():
    signal_in = np.array([0.1, -0.2, 0.5, -0.4], dtype=np.float32)
    normalized, gain_db = normalize_peak(signal_in, target_dbfs=-0.5)

    target_linear = 10.0 ** (-0.5 / 20.0)
    assert abs(np.max(np.abs(normalized)) - target_linear) < 1e-5
    assert gain_db > 0.0


def test_dsp_chain_pipeline():
    fs = 48000.0
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    signal_in = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    config = ProcessingChainConfig()
    config.normalize_target_dbfs = -1.0
    out, stats = DSPProcessor.process(signal_in, fs, config)

    assert out.shape == signal_in.shape
    assert "limited" in stats or "normalization_gain_db" in stats
