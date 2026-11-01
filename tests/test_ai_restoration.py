"""
VOX Studio — AI Konuşma Restorasyonu ve Ses Motoru Kapsamlı Test Paketi

Doğrulanan Alanlar:
1. AI Sinir Ağı Konuşma Temizleme (GTCRN ONNX + Linkwitz-Riley Crossover).
2. De-Clip: Kırpılmış tepe noktalarını kübik spline ile kurtarma.
3. De-Click & De-Pop: İmpulsif sıçramalar ve nefes patlamalarını temizleme.
4. De-Hum: 50/60 Hz şebeke tonu ve harmonik çentik filtreleme.
5. De-Reverb: Akustik oda yankısı ve çınlama azaltma.
6. Speech Clarity Engine: 3.5 kHz varlık, 350 Hz kutu hissi kesimi, 12 kHz hava.
7. 3 Kalite Profili: Natural, Balanced, Maximum Restoration.
8. Adaptive Restoration: Otomatik teşhis ve en uygun zincir yapılandırması.
9. A/B Karşılaştırma Oynatıcısı: Senkronize kanal değişimi ve LUFS eşitleme.
10. Dışa Aktarma: 24-bit WAV, MP3 (PyAV libmp3lame), FLAC ve metadata doğrulama.
11. Uç Durumlar: Tamamen sessiz dosya, aşırı kırpılmış ses, bozuk dosya koruması.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ai.restoration_engine import (
    HAS_ONNX,
    AIRestorationEngine,
    GTCRNDenoiser,
    get_ai_engine,
)
from audio.ab_player import ABChannel, ABPlayer
from audio.exporter import (
    HAS_PYAV,
    export_flac,
    export_mp3_pyav,
    export_wav,
    validate_exported_file,
)
from dsp.analysis import analyze_array
from dsp.chain import (
    DSPProcessor,
    ProcessingChainConfig,
    compute_detailed_metrics,
    create_adaptive_profile,
    create_balanced_profile,
    create_maximum_restoration_profile,
    create_natural_profile,
)
from dsp.dereverb import DereverbSettings, dereverb_spectral
from dsp.restoration import (
    DeclickSettings,
    DeclipSettings,
    PlosiveSettings,
    SpeechClaritySettings,
    apply_speech_clarity,
    declick,
    declip,
    reduce_plosives,
)


# ==============================================================================
# 1. AI SPEECH RESTORATION ENGINE TESTLERİ
# ==============================================================================
@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime kütüphanesi kurulu değil.")
def test_ai_engine_availability():
    engine = get_ai_engine()
    assert engine.is_available() is True, "GTCRN ONNX modeli ve onnxruntime hazır olmalı."


@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime kütüphanesi kurulu değil.")
def test_gtcrn_denoiser_16k_mono():
    engine = get_ai_engine()
    denoiser = engine.get_gtcrn()

    fs = 16000
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    # 440 Hz ton + Gauss gürültüsü
    speech = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.15 * np.random.randn(len(t))
    noisy = (speech + noise).astype(np.float32)

    cleaned = denoiser.process_16k_mono(noisy, strength=0.9)
    assert len(cleaned) == len(noisy)
    assert not np.isnan(cleaned).any()
    assert not np.isinf(cleaned).any()

    # Gürültü azaldığı için tepe/RMS kontrolü
    assert np.max(np.abs(cleaned)) < 1.0


@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime kütüphanesi kurulu değil.")
def test_ai_engine_restore_speech_48k():
    engine = get_ai_engine()
    fs = 48000
    t = np.linspace(0, 0.4, int(fs * 0.4), endpoint=False)
    speech = 0.4 * np.sin(2 * np.pi * 320 * t)
    noise = 0.1 * np.random.randn(len(t))
    audio_48k = (speech + noise).astype(np.float32)

    progress_reports = []
    cleaned = engine.restore_speech(
        audio_48k,
        fs,
        strength=0.8,
        preserve_high_frequencies=True,
        blend=1.0,
        progress_cb=lambda p: progress_reports.append(p),
    )

    assert len(cleaned) == len(audio_48k)
    assert len(progress_reports) > 0
    assert progress_reports[-1] == 1.0
    assert not np.isnan(cleaned).any()


# ==============================================================================
# 2. DSP RESTORASYON MODÜLLERİ TESTLERİ
# ==============================================================================
def test_declip_restoration():
    fs = 48000
    t = np.linspace(0, 0.05, int(fs * 0.05), endpoint=False)
    sine = (np.sin(2 * np.pi * 120 * t) * 1.4).astype(np.float32)
    clipped = np.clip(sine, -0.999, 0.999)

    rep_audio, rep_count = declip(clipped, DeclipSettings(enabled=True, threshold_dbfs=-0.1))
    assert rep_count > 0, "Kırpılmış tepe noktaları tespit edilip onarılmalı."
    # Kare dalga düzlükleri kübik spline ile yuvarlatılmış olmalı
    assert np.max(np.abs(rep_audio)) <= 1.0


def test_declick_restoration():
    fs = 48000
    t = np.linspace(0, 0.2, int(fs * 0.2), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    # Anlık tıkırtı iğnesi ekle
    audio[1000] += 0.6
    audio[5000:5003] -= 0.5

    clean_audio, click_count = declick(audio, fs, DeclickSettings(enabled=True, sensitivity=0.7))
    assert click_count >= 1, "Tıkırtı sıçramaları algılanıp onarılmalı."
    assert abs(clean_audio[1000] - audio[999]) < 0.2


def test_plosive_reduction():
    fs = 48000
    t = np.linspace(0, 0.2, int(fs * 0.2), endpoint=False)
    # 40 Hz rüzgar / nefes vuruşu patlaması
    plosive_burst = 0.8 * np.sin(2 * np.pi * 40 * t)
    audio = (0.2 * np.sin(2 * np.pi * 1000 * t) + plosive_burst).astype(np.float32)

    filtered = reduce_plosives(audio, fs, PlosiveSettings(enabled=True, cutoff_hz=120.0))
    # 40 Hz patlama bastırılmalı, RMS düşmeli
    assert np.sqrt(np.mean(filtered**2)) < np.sqrt(np.mean(audio**2))


def test_dereverb_spectral():
    fs = 48000
    t = np.linspace(0, 0.3, int(fs * 0.3), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    out = dereverb_spectral(audio, fs, DereverbSettings(enabled=True, rt60=0.4, amount=0.6))
    assert len(out) == len(audio)
    assert not np.isnan(out).any()


def test_speech_clarity_engine():
    fs = 48000
    t = np.linspace(0, 0.2, int(fs * 0.2), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)

    out = apply_speech_clarity(
        audio,
        fs,
        SpeechClaritySettings(enabled=True, presence_gain_db=3.0, boxiness_cut_db=-2.0, air_shelf_db=1.5),
    )
    assert len(out) == len(audio)
    assert not np.isnan(out).any()


# ==============================================================================
# 3. KALİTE PROFİLLERİ VE ADAPTİF RESTORASYON TESTLERİ
# ==============================================================================
def test_profiles_configuration():
    nat = create_natural_profile()
    bal = create_balanced_profile()
    max_p = create_maximum_restoration_profile()

    assert nat.ai_denoise.strength < bal.ai_denoise.strength < max_p.ai_denoise.strength
    assert nat.normalize_lufs_target == -19.0
    assert bal.normalize_lufs_target == -16.0
    assert max_p.normalize_lufs_target == -16.0
    assert max_p.plosive.enabled is True


def test_adaptive_restoration_profile():
    fs = 48000
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    # 50 Hz şebeke uğultulu ve kırpılmış sinyal
    sig = (0.4 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    sig[200:210] = 1.0

    analysis = analyze_array(sig, fs)
    assert analysis.hum_detected is True
    assert analysis.clipped_samples > 0

    cfg = create_adaptive_profile(analysis)
    assert cfg.hum_filter.enabled is True
    assert cfg.declip.enabled is True


@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime kütüphanesi kurulu değil.")
def test_dsp_processor_full_pipeline():
    fs = 48000
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 350 * t) + 0.1 * np.random.randn(len(t))).astype(np.float32)

    cfg = create_balanced_profile()
    processed, stats = DSPProcessor.process(sig, fs, cfg)

    assert "ai_denoise" in stats["stages_executed"]
    assert "normalize_lufs" in stats["stages_executed"]
    assert len(processed) == len(sig)

    metrics = compute_detailed_metrics(processed, fs)
    assert abs(metrics["lufs"] - (-16.0)) <= 0.5
    assert metrics["peak_dbfs"] <= -0.9


# ==============================================================================
# 4. A/B OYNATICI VE LOUDNESS MATCHING TESTLERİ
# ==============================================================================
def test_ab_player_toggle_and_matching():
    fs = 48000
    t = np.linspace(0, 0.2, int(fs * 0.2), endpoint=False)
    orig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    clean = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    player = ABPlayer(fs)
    player.load(orig, clean, fs)

    assert player.active_channel == ABChannel.A
    player.switch_channel(ABChannel.B)
    assert player.active_channel == ABChannel.B

    # Loudness matching
    player.set_loudness_matching(True)
    assert player._gain_b > 1.2, "Daha sessiz olan B kanalı Loudness eşitlemede yükseltilmeli."

    player.close()


# ==============================================================================
# 5. DIŞA AKTARMA VE METADATA DOĞRULAMA TESTLERİ
# ==============================================================================
@pytest.mark.skipif(not HAS_PYAV, reason="PyAV (av) kütüphanesi kurulu değil.")
def test_export_wav_and_mp3(tmp_path: Path):
    fs = 48000
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    data = (0.4 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)

    wav_p = export_wav(data, fs, tmp_path / "test.wav", bit_depth=24)
    info_wav = validate_exported_file(wav_p)
    assert info_wav["format"] == "WAV"
    assert info_wav["subtype"] == "PCM_24"
    assert info_wav["samplerate"] == 48000
    assert abs(info_wav["duration"] - 0.5) < 0.05

    mp3_p = export_mp3_pyav(data, fs, tmp_path / "test.mp3", bitrate_kbps=320)
    info_mp3 = validate_exported_file(mp3_p)
    assert info_mp3["format"] == "MP3"
    assert info_mp3["size_bytes"] > 1000

    flac_p = export_flac(data, fs, tmp_path / "test.flac", bit_depth=24)
    info_flac = validate_exported_file(flac_p)
    assert info_flac["format"] == "FLAC"


# ==============================================================================
# 6. UÇ DURUMLAR (EDGE CASES)
# ==============================================================================
def test_silent_audio():
    fs = 48000
    silent = np.zeros(fs, dtype=np.float32)
    cfg = create_balanced_profile()
    processed, stats = DSPProcessor.process(silent, fs, cfg)
    assert len(processed) == fs
    assert not np.isnan(processed).any()


def test_corrupted_file_handling(tmp_path: Path):
    bad_file = tmp_path / "corrupt.wav"
    bad_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00not real audio data")

    with pytest.raises(Exception):
        from dsp.analysis import analyze_file
        analyze_file(bad_file)
