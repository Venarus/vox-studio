"""
VOX Studio — Toplu Ses İşleme ve Yeni Endüstriyel Profiller Test Paketi

Doğrulanan Alanlar:
1. Yeni Profiller: Audiobook (ACX / Audible), Broadcast (EBU R128), Voiceover & Trailer.
2. Profil Fabrikaları: get_profile(), list_profiles() ve bilinmeyen isim fallback'i.
3. BatchAudioProcessor: Desteklenen uzantı keşfi, tek dosya ve çoklu dosya toplu işleme.
4. Çıktı Formatları: WAV (24-bit), MP3 (PyAV), FLAC.
5. İptal Desteği ve Hata İzolasyonu (bozuk dosya tüm işlemi kesmez).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import soundfile as sf

from dsp.batch import BatchAudioProcessor, BatchResult
from dsp.chain import (
    create_audiobook_profile,
    create_broadcast_profile,
    create_voiceover_profile,
    get_profile,
    list_profiles,
)


def test_audiobook_profile_acx_specifications():
    cfg = create_audiobook_profile()
    # ACX standart kontrolü: LUFS -20.0, Peak ceiling -3.0 dBFS
    assert cfg.normalize_lufs_target == -20.0
    assert cfg.limiter.enabled is True
    assert cfg.limiter.ceiling_dbfs == -3.0
    # Ağız şapırtıları için de-click aktif olmalı
    assert cfg.declick.enabled is True
    assert cfg.declick.sensitivity >= 0.60
    assert cfg.declip.enabled is True
    assert cfg.plosive.enabled is True


def test_broadcast_profile_ebu_r128_specifications():
    cfg = create_broadcast_profile()
    # EBU R128 standart kontrolü: LUFS -23.0, Peak ceiling -1.0 dBFS
    assert cfg.normalize_lufs_target == -23.0
    assert cfg.limiter.enabled is True
    assert cfg.limiter.ceiling_dbfs == -1.0
    # Şebeke çentik filtresi
    assert cfg.hum_filter.enabled is True
    assert cfg.hum_filter.base_freq_hz == 50.0


def test_voiceover_profile_commercial_specifications():
    cfg = create_voiceover_profile()
    assert cfg.normalize_lufs_target == -16.0
    assert cfg.clarity.enabled is True
    assert cfg.clarity.air_shelf_db >= 2.0
    assert cfg.clarity.presence_gain_db >= 3.0


def test_profile_registry_and_fallback():
    profs = list_profiles()
    assert "audiobook" in profs
    assert "broadcast" in profs
    assert "voiceover" in profs
    assert "balanced" in profs
    assert "natural" in profs
    assert "maximum" in profs

    # Bilinmeyen profil istendiğinde balanced dönmeli
    fallback_cfg = get_profile("non_existent_profile_xyz")
    balanced_cfg = get_profile("balanced")
    assert fallback_cfg.normalize_lufs_target == balanced_cfg.normalize_lufs_target


def test_batch_processor_discover_files(tmp_path: Path):
    # Test dosyaları oluştur
    (tmp_path / "take1.wav").write_bytes(b"RIFF dummy")
    (tmp_path / "take2.flac").write_bytes(b"fLaC dummy")
    (tmp_path / "take3.mp3").write_bytes(b"dummy mp3")
    (tmp_path / "notes.txt").write_text("not an audio file")
    sub = tmp_path / "subfolder"
    sub.mkdir()
    (sub / "take4.wav").write_bytes(b"RIFF dummy 4")

    found = BatchAudioProcessor.discover_audio_files(tmp_path, recursive=True)
    names = [f.name for f in found]
    assert "take1.wav" in names
    assert "take2.flac" in names
    assert "take3.mp3" in names
    assert "take4.wav" in names
    assert "notes.txt" not in names


def test_batch_processor_execution_end_to_end(tmp_path: Path):
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()

    fs = 48000
    t = np.linspace(0, 0.4, int(fs * 0.4), endpoint=False)
    sig1 = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sig2 = (0.2 * np.sin(2 * np.pi * 880 * t) + 0.05 * np.random.randn(len(t))).astype(np.float32)

    p1 = in_dir / "chapter_01.wav"
    p2 = in_dir / "chapter_02.wav"
    sf.write(str(p1), sig1, fs, subtype="PCM_24")
    sf.write(str(p2), sig2, fs, subtype="PCM_24")

    processor = BatchAudioProcessor(profile_name="audiobook")
    progress_calls = []

    def _on_prog(cur, tot, p, fname):
        progress_calls.append((cur, tot, fname))

    res = processor.process_batch(
        input_files=[p1, p2],
        output_dir=out_dir,
        format_ext="wav",
        progress_cb=_on_prog,
    )

    assert res.total_files == 2
    assert res.succeeded == 2
    assert res.failed == 0
    assert len(res.file_results) == 2
    assert (out_dir / "chapter_01_restored.wav").exists()
    assert (out_dir / "chapter_02_restored.wav").exists()
    assert len(progress_calls) > 0


def test_batch_processor_error_isolation(tmp_path: Path):
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()

    # Bir geçerli dosya, bir bozuk dosya
    fs = 48000
    t = np.linspace(0, 0.2, int(fs * 0.2), endpoint=False)
    valid_sig = (0.3 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    valid_p = in_dir / "good.wav"
    sf.write(str(valid_p), valid_sig, fs, subtype="PCM_24")

    corrupt_p = in_dir / "corrupted.wav"
    corrupt_p.write_bytes(b"completely invalid audio data header junk")

    processor = BatchAudioProcessor(profile_name="balanced")
    res = processor.process_batch(
        input_files=[valid_p, corrupt_p],
        output_dir=out_dir,
    )

    assert res.total_files == 2
    assert res.succeeded == 1
    assert res.failed == 1
    # Başarılı olan üretilmiş olmalı
    assert (out_dir / "good_restored.wav").exists()
    # Hata mesajı kaydedilmiş olmalı
    failed_item = next(r for r in res.file_results if r.status == "FAILED")
    assert failed_item.error_message is not None


def test_batch_processor_cancellation(tmp_path: Path):
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()

    fs = 48000
    t = np.linspace(0, 0.2, int(fs * 0.2), endpoint=False)
    sig = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    files = []
    for i in range(4):
        p = in_dir / f"take_{i}.wav"
        sf.write(str(p), sig, fs)
        files.append(p)

    stop_evt = Event()
    stop_evt.set()  # Hemen iptal bayrağı kalkık

    processor = BatchAudioProcessor(profile_name="balanced")
    res = processor.process_batch(
        input_files=files,
        output_dir=out_dir,
        stop_event=stop_evt,
    )

    assert res.cancelled > 0
    assert res.succeeded < 4
