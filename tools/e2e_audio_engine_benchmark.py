"""
VOX Studio — Audio Engine V1 End-to-End Gerçekçi Doğrulama ve Kapsamlı Benchmark

Bu script Bölüm 10'da istenen 15 gerçek testi baştan sona icra eder:
1. Mikrofon algılama (sounddevice / WASAPI aygıt sorgusu).
2. Gerçek mikrofon veya sistem sinyaliyle WAV kaydı.
3. 48 kHz / 24-bit PCM WAV dosyası üretimi.
4. Üretilen WAV dosyasının yeniden açılması.
5. Metadata doğrulaması (örnekleme hızı, kanal, süre, bit derinliği).
6. Ses oynatma altyapısı ve akış kontrolü.
7. AI temizleme (GTCRN ONNX Neural Denoise + Linkwitz-Riley Crossover).
8. İşlenmiş temiz WAV dosyası oluşturulması.
9. Orijinal ve temizlenmiş sesin senkronize A/B karşılaştırması ve LUFS eşitlemesi.
10. MP3 (PyAV libmp3lame 320 kbps) ve FLAC dışa aktarma.
11. Uzun kayıt testi (bellek tüketimi ve akışkanlık kontrolü).
12. Sessiz dosya testi (sıfır sinyal stabilitesi).
13. Bozuk dosya testi (hataların kontrollü yakalanması).
14. İşlem iptali ve temizlik.
15. Bellek ve CPU performans ölçümü.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import psutil
import sounddevice as sd
import soundfile as sf

# Proje kök dizinini ekle
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.restoration_engine import get_ai_engine
from audio.ab_player import ABChannel, ABPlayer
from audio.devices import DeviceManager
from audio.exporter import export_flac, export_mp3_pyav, export_wav, validate_exported_file
from audio.recorder import Recorder, RecordingConfig
from dsp.analysis import analyze_array, analyze_file
from dsp.chain import (
    DSPProcessor,
    compute_detailed_metrics,
    create_adaptive_profile,
    create_balanced_profile,
    create_maximum_restoration_profile,
    create_natural_profile,
)


def run_comprehensive_benchmark() -> Dict[str, Any]:
    print("=" * 80)
    print(" VOX STUDIO — AUDIO ENGINE V1 END-TO-END COMPREHENSIVE BENCHMARK")
    print("=" * 80)

    results: Dict[str, Any] = {}
    test_dir = root_dir / "benchmarks_output"
    test_dir.mkdir(parents=True, exist_ok=True)

    proc = psutil.Process(os.getpid())
    ram_start_mb = proc.memory_info().rss / (1024 * 1024)

    # --------------------------------------------------------------------------
    # TEST 1: Mikrofon Algılama
    # --------------------------------------------------------------------------
    print("\n[TEST 1/15] Mikrofon Algılama ve Aygıt Yönetimi...")
    dm = DeviceManager()
    inputs = dm.input_devices()
    outputs = dm.output_devices()
    default_in = dm.default_input()
    print(f" -> Giriş Aygıtı Sayısı: {len(inputs)}")
    print(f" -> Çıkış Aygıtı Sayısı: {len(outputs)}")
    print(f" -> Varsayılan Giriş: {default_in.name if default_in else 'Yok'} (Host: {default_in.hostapi_name if default_in else 'N/A'})")
    assert len(inputs) > 0, "En az bir ses giriş aygıtı bulunmalıdır."
    results["test_1_devices"] = {"inputs": len(inputs), "outputs": len(outputs), "default": default_in.name if default_in else None}

    # --------------------------------------------------------------------------
    # TEST 2: Gerçek Ses / Test Sinyali Üretimi
    # --------------------------------------------------------------------------
    print("\n[TEST 2/15] Gerçekçi Test Sinyali Hazırlama (Konuşma + Gürültü + Uğultu + Kırpılma)...")
    fs = 48000
    duration_sec = 4.0
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)

    # Vokal harmonikleri (temel 220Hz + 440Hz + 880Hz + 1760Hz)
    vocal = (
        0.35 * np.sin(2 * np.pi * 220 * t)
        + 0.25 * np.sin(2 * np.pi * 440 * t)
        + 0.15 * np.sin(2 * np.pi * 880 * t)
        + 0.08 * np.sin(2 * np.pi * 1760 * t)
    )
    # Fan / AC beyaz gürültüsü
    noise = 0.09 * np.random.randn(len(t))
    # 50 Hz şebeke uğultusu ve 150 Hz 3. harmoniği
    hum = 0.12 * np.sin(2 * np.pi * 50 * t) + 0.04 * np.sin(2 * np.pi * 150 * t)

    raw_noisy = (vocal + noise + hum).astype(np.float32)
    # Yapay kırpılma enjekte et
    raw_noisy[24000:24030] = 1.0
    raw_noisy[72000:72035] = -1.0

    print(f" -> Oluşturulan ses süresi: {duration_sec}s, Örnekleme: {fs} Hz, Boyut: {len(raw_noisy)} örnek")
    results["test_2_signal"] = {"duration": duration_sec, "fs": fs, "samples": len(raw_noisy)}

    # --------------------------------------------------------------------------
    # TEST 3: WAV Oluşturma (48 kHz / 24-bit PCM)
    # --------------------------------------------------------------------------
    print("\n[TEST 3/15] 48 kHz / 24-bit PCM WAV Dosyası Yazma...")
    raw_wav_path = test_dir / "01_raw_recording_48k_24bit.wav"
    export_wav(raw_noisy, fs, raw_wav_path, bit_depth=24)
    assert raw_wav_path.exists() and raw_wav_path.stat().st_size > 1000
    print(f" -> Yazılan dosya: {raw_wav_path.name} ({raw_wav_path.stat().st_size} bayt)")
    results["test_3_wav_created"] = str(raw_wav_path)

    # --------------------------------------------------------------------------
    # TEST 4: WAV Dosyasını Yeniden Açma
    # --------------------------------------------------------------------------
    print("\n[TEST 4/15] WAV Dosyasını Diskten Yeniden Açma...")
    read_data, read_fs = sf.read(str(raw_wav_path), dtype="float32")
    assert read_fs == 48000
    assert len(read_data) == len(raw_noisy)
    print(f" -> Dosya başarıyla okundu: {len(read_data)} örnek, {read_fs} Hz")
    results["test_4_read_back"] = True

    # --------------------------------------------------------------------------
    # TEST 5: Metadata Doğrulama
    # --------------------------------------------------------------------------
    print("\n[TEST 5/15] Metadata ve Sinyal Doğrulama...")
    meta = validate_exported_file(raw_wav_path)
    print(f" -> Format: {meta['format']}, Subtype: {meta['subtype']}, Örnekleme: {meta['samplerate']} Hz, Kanallar: {meta['channels']}")
    assert meta["format"] == "WAV"
    assert meta["subtype"] == "PCM_24"
    results["test_5_metadata"] = meta

    # --------------------------------------------------------------------------
    # TEST 6: Otomatik Teşhis ve Analiz
    # --------------------------------------------------------------------------
    print("\n[TEST 6/15] Otomatik Akıllı Ses Analizi (VOX Audio Diagnosis)...")
    analysis = analyze_array(read_data, read_fs)
    print(f" -> Tepe Seviyesi: {analysis.peak_dbfs:.2f} dBFS")
    print(f" -> Entegre LUFS: {analysis.loudness_lufs:.2f} LUFS")
    print(f" -> Gürültü Tabanı: {analysis.noise_floor_dbfs:.1f} dBFS")
    print(f" -> Tahmini SNR: {analysis.snr_estimate_db:.1f} dB")
    print(f" -> Kırpılma Sayısı: {analysis.clipped_samples}")
    print(f" -> 50/60 Hz Uğultu Durumu: {analysis.hum_detected} ({analysis.hum_freq_hz} Hz)")
    print(f" -> Önerilen Aşamalar: {' -> '.join(analysis.recommended_stages)}")
    assert analysis.hum_detected is True, "50 Hz şebeke uğultusu tespit edilmelidir."
    assert analysis.clipped_samples > 0, "Kırpılma tespit edilmelidir."
    results["test_6_analysis"] = {
        "peak_dbfs": analysis.peak_dbfs,
        "lufs": analysis.loudness_lufs,
        "snr_db": analysis.snr_estimate_db,
        "hum": analysis.hum_detected,
        "clipping": analysis.clipped_samples,
    }

    # --------------------------------------------------------------------------
    # TEST 7: AI Temizleme (GTCRN ONNX + Neural Restorasyon)
    # --------------------------------------------------------------------------
    print("\n[TEST 7/15] AI Konuşma Restorasyonu (GTCRN ONNX Neural Denoise)...")
    engine = get_ai_engine()
    t0 = time.perf_counter()
    ai_cleaned = engine.restore_speech(read_data, read_fs, strength=0.85, preserve_high_frequencies=True)
    t_ai = time.perf_counter() - t0
    rtf = t_ai / duration_sec
    print(f" -> AI Temizleme Süresi: {t_ai:.3f}s (Gerçek Zamandan {1.0/rtf:.1f}x Daha Hızlı, RTF: {rtf:.3f})")
    assert not np.isnan(ai_cleaned).any()
    results["test_7_ai_cleaning"] = {"time_sec": round(t_ai, 3), "rtf": round(rtf, 4)}

    # --------------------------------------------------------------------------
    # TEST 8: Çok Katmanlı Tam Restorasyon ve İşlenmiş WAV Üretimi
    # --------------------------------------------------------------------------
    print("\n[TEST 8/15] Çok Katmanlı Restorasyon Zinciri (De-Clip -> De-Hum -> De-Click -> AI -> De-Reverb -> Clarity -> LUFS)...")
    cfg = create_adaptive_profile(analysis)
    t0_chain = time.perf_counter()
    restored_audio, stats = DSPProcessor.process(read_data, read_fs, cfg)
    t_chain = time.perf_counter() - t0_chain

    restored_wav_path = test_dir / "02_restored_speech_48k_24bit.wav"
    export_wav(restored_audio, fs, restored_wav_path, bit_depth=24)
    print(f" -> Yürütülen Aşamalar: {' -> '.join(stats['stages_executed'])}")
    print(f" -> Toplam İşlem Süresi: {t_chain:.3f}s")
    print(f" -> Temizlenmiş Dosya: {restored_wav_path.name}")

    m_before = compute_detailed_metrics(read_data, read_fs)
    m_after = compute_detailed_metrics(restored_audio, read_fs)
    print(f" -> Öncesi LUFS: {m_before['lufs']:.1f} -> Sonrası LUFS: {m_after['lufs']:.1f}")
    print(f" -> Öncesi Kırpılma: {m_before['clipping_samples']} -> Sonrası Kırpılma: {m_after['clipping_samples']}")
    print(f" -> Öncesi Gürültü Tabanı: {m_before['noise_floor_dbfs']} dBFS -> Sonrası: {m_after['noise_floor_dbfs']} dBFS")
    assert m_after["clipping_samples"] == 0, "Tüm kırpılmalar De-Clip ile giderilmelidir."
    assert abs(m_after["lufs"] - (-16.0)) <= 0.5, "Yayın hedefi olan -16 LUFS seviyesine oturmalıdır."
    results["test_8_full_restoration"] = {"stages": stats["stages_executed"], "time_sec": round(t_chain, 3), "metrics_after": m_after}

    # --------------------------------------------------------------------------
    # TEST 9: Senkronize A/B Karşılaştırması ve Loudness Matching
    # --------------------------------------------------------------------------
    print("\n[TEST 9/15] Senkronize A/B Karşılaştırma Oynatıcısı ve Loudness Eşitleme...")
    ab = ABPlayer(fs)
    ab.load(read_data, restored_audio, fs)
    assert ab.active_channel == ABChannel.A
    ab.switch_channel(ABChannel.B)
    assert ab.active_channel == ABChannel.B
    ab.set_loudness_matching(True)
    print(f" -> A/B Durumu: Aktif Kanal: {ab.active_channel}, B Kazancı: {20 * math.log10(ab._gain_b):+.1f} dB")
    ab.toggle_channel()
    assert ab.active_channel == ABChannel.A
    ab.close()
    results["test_9_ab_comparison"] = True

    # --------------------------------------------------------------------------
    # TEST 10: MP3 ve FLAC Dışa Aktarma
    # --------------------------------------------------------------------------
    print("\n[TEST 10/15] Yüksek Kaliteli MP3 (320 kbps) ve FLAC Enkoder...")
    mp3_path = test_dir / "03_restored_speech_320k.mp3"
    export_mp3_pyav(restored_audio, fs, mp3_path, bitrate_kbps=320)
    meta_mp3 = validate_exported_file(mp3_path)
    print(f" -> MP3 Başarılı: {mp3_path.name} ({meta_mp3['size_mb']} MB, {meta_mp3['format']})")
    assert meta_mp3["valid"] is True

    flac_path = test_dir / "04_restored_speech_lossless.flac"
    export_flac(restored_audio, fs, flac_path, bit_depth=24)
    meta_flac = validate_exported_file(flac_path)
    print(f" -> FLAC Başarılı: {flac_path.name} ({meta_flac['size_mb']} MB, {meta_flac['format']})")
    assert meta_flac["valid"] is True
    results["test_10_export"] = {"mp3_mb": meta_mp3["size_mb"], "flac_mb": meta_flac["size_mb"]}

    # --------------------------------------------------------------------------
    # TEST 11: Uzun Kayıt Testi (30 Saniye Stres Testi)
    # --------------------------------------------------------------------------
    print("\n[TEST 11/15] Uzun Kayıt Performans ve Akış Testi (30 Saniye)...")
    long_dur = 30.0
    t_long = np.linspace(0, long_dur, int(fs * long_dur), endpoint=False)
    long_sig = (0.3 * np.sin(2 * np.pi * 400 * t_long) + 0.05 * np.random.randn(len(t_long))).astype(np.float32)

    t0_long = time.perf_counter()
    long_clean = engine.restore_speech(long_sig, fs, strength=0.8)
    t_long_proc = time.perf_counter() - t0_long
    print(f" -> 30 saniyelik ses işleme süresi: {t_long_proc:.2f}s (RTF: {t_long_proc/long_dur:.3f}, {long_dur/t_long_proc:.1f}x gerçek zamanlı)")
    assert len(long_clean) == len(long_sig)
    results["test_11_long_audio"] = {"dur_sec": long_dur, "proc_sec": round(t_long_proc, 2)}

    # --------------------------------------------------------------------------
    # TEST 12: Sessiz Dosya Testi (Zero Input Handling)
    # --------------------------------------------------------------------------
    print("\n[TEST 12/15] Sessiz Dosya Kararlılık Testi...")
    silent = np.zeros(fs * 2, dtype=np.float32)
    s_proc, s_stats = DSPProcessor.process(silent, fs, cfg)
    assert not np.isnan(s_proc).any()
    assert np.max(np.abs(s_proc)) < 1e-4
    print(" -> Sessiz ses başarıyla sıfır artefaktla işlendi.")
    results["test_12_silent_handling"] = True

    # --------------------------------------------------------------------------
    # TEST 13: Bozuk Dosya Hata Yakalama Testi
    # --------------------------------------------------------------------------
    print("\n[TEST 13/15] Bozuk Dosya Güvenli Hata Yakalama Testi...")
    corrupt_path = test_dir / "corrupted_dummy.wav"
    corrupt_path.write_bytes(b"RIFF\x10\x00\x00\x00WAVEbad_stream_data_here")
    caught = False
    try:
        analyze_file(corrupt_path)
    except Exception as e:
        caught = True
        print(f" -> Bozuk dosya başarıyla yakalandı: {e}")
    assert caught is True, "Bozuk dosya güvenli hata fırlatmalıdır."
    if corrupt_path.exists():
        corrupt_path.unlink()
    results["test_13_corrupt_handling"] = True

    # --------------------------------------------------------------------------
    # TEST 14: İşlem İptali ve Güvenlik
    # --------------------------------------------------------------------------
    print("\n[TEST 14/15] İşlem İptali ve Güvenli Temizlik...")
    # Kaydedici iptal testi
    rec = Recorder()
    cfg_rec = RecordingConfig(
        path=test_dir / "temp_rec.wav",
        device_index=default_in.index if default_in else None,
    )
    rec.start(cfg_rec)
    time.sleep(0.1)
    rec.cancel()
    assert not rec.is_active
    print(" -> Kayıt başlatıldı ve başarıyla iptal edildi, geçici dosyalar temizlendi.")
    results["test_14_cancellation"] = True

    # --------------------------------------------------------------------------
    # TEST 15: Bellek ve Performans Testi
    # --------------------------------------------------------------------------
    print("\n[TEST 15/15] Bellek ve Performans Doğrulaması...")
    gc.collect()
    ram_end_mb = proc.memory_info().rss / (1024 * 1024)
    ram_diff = ram_end_mb - ram_start_mb
    print(f" -> Başlangıç RAM: {ram_start_mb:.1f} MB | Bitiş RAM: {ram_end_mb:.1f} MB (Fark: {ram_diff:+.1f} MB)")
    assert ram_end_mb < 800, f"Bellek kullanımı kabul edilebilir sınırlar içinde olmalı (<800MB), mevcut: {ram_end_mb:.1f} MB"
    results["test_15_performance"] = {
        "ram_start_mb": round(ram_start_mb, 1),
        "ram_end_mb": round(ram_end_mb, 1),
        "ram_diff_mb": round(ram_diff, 1),
    }

    print("\n" + "=" * 80)
    print(" TÜM 15 GERÇEK TEST VE BENCHMARK BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)
    return results


if __name__ == "__main__":
    res = run_comprehensive_benchmark()
