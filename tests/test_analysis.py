"""Ses analizi testleri (sentetik sinyaller + gerçek dosya)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf

from dsp.analysis import LoudnessMeter, analyze_array, analyze_file, k_weighting_sos


def sine(amplitude: float, freq: float = 1000.0, seconds: float = 3.0, sr: int = 48000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32).reshape(-1, 1)


def test_peak_rms_crest_of_sine():
    a = analyze_array(sine(0.5), 48000)
    assert a.frames == 144000
    assert a.duration == pytest.approx(3.0)
    assert a.peak_dbfs == pytest.approx(-6.02, abs=0.05)
    assert a.rms_dbfs == pytest.approx(-9.03, abs=0.05)
    assert a.crest_db == pytest.approx(3.01, abs=0.1)
    assert a.clipped_samples == 0
    assert a.silence_ratio == pytest.approx(0.0, abs=1e-3)


def test_silence_ratio_and_warning():
    a = analyze_array(np.zeros((48000 * 2, 1), np.float32), 48000)
    assert a.silence_ratio == pytest.approx(1.0)
    assert a.rms_dbfs == -math.inf
    assert a.score is not None and a.score < 80
    assert any("sessiz" in w for w in a.warnings)


def test_clipping_detected_and_penalised():
    clean = analyze_array(sine(0.5), 48000)
    clipped_signal = np.clip(sine(1.5), -1.0, 1.0)
    clipped = analyze_array(clipped_signal, 48000)
    assert clipped.clipped_samples > 0
    assert clipped.score < clean.score
    assert any("kırp" in w.lower() for w in clipped.warnings)


def test_loudness_reference_sine():
    # 1 kHz, tepe genliği 0.1 (-20 dBFS peak) sinüs: RMS -23.0 dBFS,
    # BS.1770 K-ağırlık 1 kHz'de +0.70 dB ve -0.691 düzeltmesi -> ~ -23.0 LUFS
    a = analyze_array(sine(0.1, seconds=5.0), 48000)
    assert math.isfinite(a.loudness_lufs)
    assert a.loudness_lufs == pytest.approx(-23.0, abs=0.5)
    assert a.loudness_blocks > 0


def test_loudness_is_rate_independent():
    at_48 = analyze_array(sine(0.1, seconds=4.0, sr=48000), 48000)
    at_44 = analyze_array(sine(0.1, seconds=4.0, sr=44100), 44100)
    assert at_44.loudness_lufs == pytest.approx(at_48.loudness_lufs, abs=0.5)


def test_stereo_imbalance_flagged():
    left = sine(0.5, seconds=2.0)
    right = sine(0.05, seconds=2.0)
    stereo = np.hstack([left, right])
    a = analyze_array(stereo, 48000)
    assert a.channels == 2
    assert a.channel_imbalance_db == pytest.approx(20.0, abs=0.5)
    assert any("dengesiz" in w.lower() for w in a.warnings)


def test_dc_offset_flagged():
    signal = sine(0.3, seconds=2.0) + 0.05
    a = analyze_array(signal.astype(np.float32), 48000)
    assert a.dc_offset > 0.02
    assert "dc_offset" in a.score_breakdown


def test_k_weighting_filter_shape():
    sos = k_weighting_sos()
    assert sos.shape == (2, 6)
    # 1 kHz'de K-ağırlık toplam kazancı BS.1770'e göre +0.7 dB civarındadır
    w, h = __import__("scipy.signal", fromlist=["sosfreqz"]).sosfreqz(sos, worN=[1000], fs=48000)
    assert abs(20 * math.log10(abs(h[0]))) < 1.0
    # 20 Hz'de yüksek geçiren baskın: belirgin zayıflama beklenir
    _, h_low = __import__("scipy.signal", fromlist=["sosfreqz"]).sosfreqz(sos, worN=[20], fs=48000)
    assert 20 * math.log10(abs(h_low[0])) < -6.0
    # 3 kHz'de raf filtresi nedeniyle hafif yükselti
    _, h_high = __import__("scipy.signal", fromlist=["sosfreqz"]).sosfreqz(sos, worN=[3000], fs=48000)
    assert 20 * math.log10(abs(h_high[0])) > 2.0


def test_loudness_meter_gating_ignores_silence():
    meter = LoudnessMeter()
    meter.push(sine(0.1, seconds=4.0))          # konuşma benzeri bölüm
    meter.push(np.zeros((48000 * 4, 1), np.float32))  # uzun sessizlik
    lufs, blocks = meter.integrated_lufs()
    assert math.isfinite(lufs)
    assert blocks > 0
    assert lufs == pytest.approx(-23.0, abs=1.0)


def test_analyze_file_roundtrip(tmp_path):
    path = tmp_path / "test_kayit.wav"
    data = sine(0.25, seconds=2.0)
    sf.write(str(path), data, 48000, subtype="PCM_24")
    a = analyze_file(path)
    assert a.frames == 96000
    assert a.samplerate == 48000
    assert a.channels == 1
    assert a.subtype == "PCM_24"
    assert a.file_format == "WAV"
    assert a.source == str(path)
    assert a.peak_dbfs == pytest.approx(-12.04, abs=0.1)


def test_analyze_file_rejects_corrupt(tmp_path):
    bad = tmp_path / "bozuk.wav"
    bad.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    with pytest.raises(ValueError) as exc:
        analyze_file(bad)
    assert "bozuk" in str(exc.value).lower() or "desteklenmiyor" in str(exc.value).lower()


def test_analyze_file_missing(tmp_path):
    with pytest.raises(ValueError):
        analyze_file(tmp_path / "yok.wav")


def test_score_bounds_and_disclaimer():
    a = analyze_array(sine(0.3), 48000)
    assert 0 <= a.score <= 100
    assert "mutlak" in a.score_disclaimer
    assert isinstance(a.score_breakdown, dict)
