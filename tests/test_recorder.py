"""Kayıt motoru testleri — fiziksel mikrofon kullanılmaz.

Gerçek PortAudio akışı yerine ``stream_factory`` ile sahte bir akış verilir;
motorun tamponlama, dosya yazımı, duraklatma ve kurtarma davranışı sınanır.
Bu testler donanımın çalıştığını iddia etmez (bkz. tools/hw_smoke.py).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio.recorder import (
    RecordingConfig,
    RecordingError,
    RecordingState,
    Recorder,
    build_recording_filename,
    recover_pending_recordings,
    sanitize_filename,
)


class FakeStream:
    def __init__(self, cfg: RecordingConfig, callback):
        self.cfg = cfg
        self.callback = callback
        self.active = False
        self.stopped = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False
        self.stopped = True

    def close(self):
        self.active = False

    def push(self, frames: int, amplitude: float = 0.3, freq: float = 440.0) -> None:
        sr = self.cfg.samplerate
        t = (np.arange(frames) + self._offset()) / sr
        data = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        block = np.repeat(data.reshape(-1, 1), self.cfg.channels, axis=1)
        self.callback(block, frames, None, None)

    _pushed = 0

    def _offset(self) -> int:
        return 0


def make_stream_factory(holder: dict):
    def factory(cfg: RecordingConfig, callback):
        stream = FakeStream(cfg, callback)
        holder["stream"] = stream
        return stream

    return factory


def make_recorder(tmp_path, **kwargs):
    holder: dict = {}
    errors: list = []
    states: list = []
    rec = Recorder(
        on_state=lambda s: states.append(s),
        on_error=errors.append,
        stream_factory=make_stream_factory(holder),
        writer_poll=0.01,
        **kwargs,
    )
    return rec, holder, errors, states


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_record_and_finalize(tmp_path):
    rec, holder, errors, states = make_recorder(tmp_path)
    cfg = RecordingConfig(path=tmp_path / "take1.wav", samplerate=48000, channels=1, blocksize=1024)
    rec.start(cfg)
    assert rec.state is RecordingState.RECORDING

    stream = holder["stream"]
    for _ in range(47):  # 47 * 1024 ≈ 1.003 sn
        stream.push(1024)
    assert wait_until(lambda: rec.frames_written == 47 * 1024), rec.frames_written

    result = rec.stop()
    assert result.path == str(tmp_path / "take1.wav")
    assert Path(result.path).exists()
    assert result.frames == 47 * 1024
    assert result.samplerate == 48000
    assert result.channels == 1
    assert result.subtype == "PCM_24"
    assert result.duration == pytest.approx(1.0027, abs=1e-3)
    assert result.peak == pytest.approx(0.3, abs=0.01)
    assert result.overflow_frames == 0
    assert errors == []
    assert RecordingState.FINISHED in states

    info = sf.info(result.path)
    assert info.frames == 47 * 1024
    assert info.samplerate == 48000
    assert info.subtype == "PCM_24"

    meta = json.loads((tmp_path / "take1.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished"
    assert meta["frames"] == 47 * 1024
    assert not (tmp_path / "take1.part.wav").exists()


def test_pause_drops_input_and_resume_continues(tmp_path):
    rec, holder, errors, states = make_recorder(tmp_path)
    cfg = RecordingConfig(path=tmp_path / "take2.wav", samplerate=48000, channels=1, blocksize=1024)
    rec.start(cfg)
    stream = holder["stream"]

    for _ in range(10):
        stream.push(1024)
    assert wait_until(lambda: rec.frames_written == 10 * 1024)

    rec.pause()
    assert rec.state is RecordingState.PAUSED
    for _ in range(5):
        stream.push(1024)  # duraklatma sırasında gelen ses yazılmamalı
    time.sleep(0.1)
    assert rec.frames_written == 10 * 1024

    rec.resume()
    for _ in range(10):
        stream.push(1024)
    assert wait_until(lambda: rec.frames_written == 20 * 1024)

    result = rec.stop()
    assert result.frames == 20 * 1024
    assert sf.info(result.path).frames == 20 * 1024
    assert errors == []


def test_cancel_removes_partial_files(tmp_path):
    rec, holder, errors, _ = make_recorder(tmp_path)
    cfg = RecordingConfig(path=tmp_path / "iptal.wav", samplerate=48000, channels=1)
    rec.start(cfg)
    holder["stream"].push(4096)
    assert wait_until(lambda: rec.frames_written > 0)

    rec.cancel()
    assert rec.state is RecordingState.CANCELLED
    assert not (tmp_path / "iptal.wav").exists()
    assert not (tmp_path / "iptal.part.wav").exists()
    assert not (tmp_path / "iptal.json").exists()


def test_existing_file_is_never_overwritten_silently(tmp_path):
    target = tmp_path / "ayni_ad.wav"
    sf.write(str(target), np.zeros((1000, 1), np.float32), 48000)
    rec, holder, _, _ = make_recorder(tmp_path)
    rec.start(RecordingConfig(path=target, samplerate=48000, channels=1))
    holder["stream"].push(2048)
    assert wait_until(lambda: rec.frames_written == 2048)
    result = rec.stop()

    assert Path(result.path) != target
    assert sf.info(str(target)).frames == 1000  # eski dosya korunur
    assert Path(result.path).exists()


def test_double_start_rejected(tmp_path):
    rec, holder, _, _ = make_recorder(tmp_path)
    rec.start(RecordingConfig(path=tmp_path / "a.wav", samplerate=48000, channels=1))
    with pytest.raises(RecordingError):
        rec.start(RecordingConfig(path=tmp_path / "b.wav", samplerate=48000, channels=1))
    rec.cancel()


def test_stop_without_recording_raises(tmp_path):
    rec, _, _, _ = make_recorder(tmp_path)
    with pytest.raises(RecordingError):
        rec.stop()


def test_stream_open_failure_reports_clear_error(tmp_path):
    def failing_factory(cfg, callback):
        raise RuntimeError("cihaz meşgul")

    rec = Recorder(stream_factory=failing_factory)
    with pytest.raises(RecordingError) as exc:
        rec.start(RecordingConfig(path=tmp_path / "hata.wav", samplerate=48000, channels=1))
    assert "Ses cihazı açılamadı" in str(exc.value)
    assert rec.state is RecordingState.ERROR
    assert not (tmp_path / "hata.wav").exists()
    assert not (tmp_path / "hata.part.wav").exists()


def test_device_loss_is_reported_and_data_kept(tmp_path):
    errors: list = []
    holder: dict = {}
    rec = Recorder(
        on_error=errors.append,
        stream_factory=make_stream_factory(holder),
        writer_poll=0.01,
    )
    rec.start(RecordingConfig(path=tmp_path / "kopma.wav", samplerate=48000, channels=1))
    holder["stream"].push(8192)
    assert wait_until(lambda: rec.frames_written == 8192)

    rec._on_stream_finished()  # cihaz koptu simülasyonu
    assert errors and ("akışı" in errors[0] or "durdu" in errors[0])
    result = rec.stop()
    assert result.frames == 8192
    assert Path(result.path).exists()


def test_waveform_and_level_available_during_recording(tmp_path):
    rec, holder, _, _ = make_recorder(tmp_path)
    rec.start(RecordingConfig(path=tmp_path / "wf.wav", samplerate=48000, channels=1))
    holder["stream"].push(24000, amplitude=0.5)
    assert wait_until(lambda: rec.frames_written == 24000)

    mins, maxs = rec.waveform_view(500)
    assert maxs.size > 0
    assert maxs.max() == pytest.approx(0.5, abs=0.02)

    snap = rec.level_snapshot()
    assert snap.clip_count == 0
    assert snap.frames == 24000
    rec.stop()


def test_stereo_recording_writes_two_channels(tmp_path):
    rec, holder, _, _ = make_recorder(tmp_path)
    rec.start(RecordingConfig(path=tmp_path / "stereo.wav", samplerate=44100, channels=2))
    holder["stream"].push(4410)
    assert wait_until(lambda: rec.frames_written == 4410)
    result = rec.stop()
    info = sf.info(result.path)
    assert info.channels == 2
    assert info.samplerate == 44100


def test_recovery_finalises_part_file(tmp_path):
    part = tmp_path / "kurtar.part.wav"
    sf.write(str(part), np.zeros((4800, 1), np.float32), 48000, subtype="PCM_24")
    meta = tmp_path / "kurtar.json"
    meta.write_text(json.dumps({"status": "recording", "target_path": str(part)}), encoding="utf-8")

    recovered = recover_pending_recordings(tmp_path)
    assert recovered == [tmp_path / "kurtar.wav"]
    assert (tmp_path / "kurtar.wav").exists()
    assert not part.exists()
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["recovered"] is True
    assert data["frames"] == 4800


def test_recovery_skips_corrupt_part_files(tmp_path):
    bad = tmp_path / "bozuk.part.wav"
    bad.write_bytes(b"not audio")
    assert recover_pending_recordings(tmp_path) == []
    assert bad.exists()  # kullanıcı verisi sessizce silinmez


def test_recovery_keeps_existing_file(tmp_path):
    sf.write(str(tmp_path / "kurtar.wav"), np.zeros((100, 1), np.float32), 48000)
    sf.write(str(tmp_path / "kurtar.part.wav"), np.zeros((4800, 1), np.float32), 48000)
    recovered = recover_pending_recordings(tmp_path)
    assert recovered == [tmp_path / "kurtar-kurtarilan.wav"]
    assert sf.info(str(tmp_path / "kurtar.wav")).frames == 100


# ---------------------------------------------------------------- dosya adları
def test_sanitize_filename_keeps_turkish_and_strips_illegal():
    assert sanitize_filename("Bölüm 1: Giriş/Çıkış") == "Bölüm_1_Giriş_Çıkış"
    assert sanitize_filename("   ") == "kayit"
    assert sanitize_filename("CON") == "_CON"
    assert "?" not in sanitize_filename("ne?")
    assert len(sanitize_filename("a" * 200)) == 80


def test_build_recording_filename_uses_timestamp_and_label():
    from datetime import datetime

    name = build_recording_filename("Sahne 4 – replik", when=datetime(2026, 10, 29, 14, 5, 9))
    assert name == "20261029_140509_Sahne_4_–_replik.wav"
    assert build_recording_filename("", when=datetime(2026, 10, 29, 14, 5, 9)) == "20261029_140509.wav"
