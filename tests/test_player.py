"""Oynatıcı testleri — sahte çıkış akışı ile (donanım gerekmez)."""

from __future__ import annotations

import time

import numpy as np
import pytest
import soundfile as sf

from audio.player import PlaybackError, PlaybackState, Player, probe_audio_file


class FakeOutput:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.channels = int(kwargs.get("channels", 1))
        self.active = False
        self.stopped = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False
        self.stopped = True

    def close(self):
        self.active = False

    def pull(self, frames: int = 1024) -> np.ndarray:
        out = np.zeros((frames, self.channels), dtype=np.float32)
        self.callback(out, frames, None, None)
        return out


def make_player(holder: dict) -> Player:
    def factory(**kwargs):
        stream = FakeOutput(**kwargs)
        holder["stream"] = stream
        return stream

    return Player(output_factory=factory)


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@pytest.fixture
def tone_file(tmp_path):
    path = tmp_path / "ton.wav"
    sr = 48000
    t = np.arange(sr) / sr
    data = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), data, sr, subtype="PCM_24")
    return path


def test_probe_reports_format(tone_file):
    info = probe_audio_file(tone_file)
    assert info.samplerate == 48000
    assert info.channels == 1
    assert info.frames == 48000
    assert info.duration == pytest.approx(1.0)
    assert info.subtype == "PCM_24"


def test_probe_missing_and_corrupt(tmp_path):
    with pytest.raises(PlaybackError) as exc:
        probe_audio_file(tmp_path / "yok.wav")
    assert "bulunamadı" in str(exc.value)

    bad = tmp_path / "bozuk.wav"
    bad.write_bytes(b"RIFFxxxxWAVEjunk")
    with pytest.raises(PlaybackError) as exc2:
        probe_audio_file(bad)
    assert "okunamadı" in str(exc2.value)


def pull_until_signal(stream, frames: int = 1024, timeout: float = 3.0) -> np.ndarray:
    """Sahte çıkıştan veri gelene kadar çeker (gerçek callback zamanlamasını taklit eder)."""
    end = time.monotonic() + timeout
    out = np.zeros((frames, stream.channels), dtype=np.float32)
    while time.monotonic() < end:
        out = stream.pull(frames)
        if np.any(out != 0.0):
            return out
        time.sleep(0.01)
    return out


def test_playback_advances_and_outputs_signal(tone_file):
    holder: dict = {}
    player = make_player(holder)
    player.load(tone_file)
    player.play()

    assert wait_until(lambda: "stream" in holder)
    stream = holder["stream"]
    assert player.state is PlaybackState.PLAYING

    out = pull_until_signal(stream)
    assert np.any(out != 0.0)
    assert player.position > 0.0
    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(0.4 / np.sqrt(2), abs=0.05)

    player.stop()
    assert player.state is PlaybackState.IDLE
    assert stream.stopped


def test_seek_repositions(tone_file):
    holder: dict = {}
    player = make_player(holder)
    player.load(tone_file)
    player.play()
    wait_until(lambda: "stream" in holder)
    holder["stream"].pull(1024)

    player.seek(0.75)
    assert player.position == pytest.approx(0.75, abs=1e-3)
    wait_until(lambda: "stream" in holder)
    holder["stream"].pull(256) if "stream" in holder else None
    player.stop()


def test_pause_and_resume(tone_file):
    holder: dict = {}
    player = make_player(holder)
    player.load(tone_file)
    player.play()
    wait_until(lambda: "stream" in holder)
    holder["stream"].pull(2048)

    pos_before = player.position
    player.pause()
    assert player.state is PlaybackState.PAUSED
    player.resume()
    assert player.state is PlaybackState.PLAYING
    assert player.position == pytest.approx(pos_before, abs=0.02)
    player.stop()


def test_natural_end_returns_to_idle(tmp_path):
    path = tmp_path / "kisa.wav"
    t = np.arange(2400) / 48000
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 48000)

    holder: dict = {}
    player = make_player(holder)
    player.load(path)
    player.play()
    stream = holder["stream"]
    for _ in range(12):
        stream.pull(256)
        time.sleep(0.01)

    assert wait_until(lambda: player.state is PlaybackState.IDLE, timeout=3.0)
    assert player.position == pytest.approx(0.05, abs=0.02)


def test_play_without_load_raises():
    player = Player()
    with pytest.raises(PlaybackError):
        player.play()


def test_stereo_file_keeps_two_channels(tmp_path):
    path = tmp_path / "stereo.wav"
    t = np.arange(4800) / 48000
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.stack([mono, -mono], axis=1)
    sf.write(str(path), stereo, 48000)

    holder: dict = {}
    player = make_player(holder)
    info = player.load(path)
    assert info.channels == 2
    player.play()
    stream = holder["stream"]
    assert stream.channels == 2
    out = pull_until_signal(stream, 512)
    assert np.any(out[:, 0] != 0)
    player.stop()


def test_missing_file_load_raises(tmp_path):
    player = Player()
    with pytest.raises(PlaybackError):
        player.load(tmp_path / "yok.wav")
