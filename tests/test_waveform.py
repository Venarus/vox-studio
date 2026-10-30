"""Waveform tepe özetleyici testleri."""

from __future__ import annotations

import numpy as np

from audio.waveform import WaveformPeaks


def test_extent_matches_signal_envelope():
    peaks = WaveformPeaks(bucket_frames=100, channels=1)
    data = np.zeros((1000, 1), dtype=np.float32)
    data[0:100, 0] = 0.5
    data[100:200, 0] = -0.25
    peaks.add(data)
    mins, maxs = peaks.extent()
    assert peaks.bucket_count == 10
    assert peaks.frames == 1000
    assert maxs[0] == 0.5
    assert mins[1] == -0.25


def test_partial_bucket_not_published_until_flush():
    peaks = WaveformPeaks(bucket_frames=100, channels=1)
    peaks.add(np.zeros((150, 1), dtype=np.float32))
    assert peaks.bucket_count == 1
    peaks.flush_partial()
    assert peaks.bucket_count == 2


def test_stereo_is_downmixed_for_display():
    peaks = WaveformPeaks(bucket_frames=4, channels=2)
    block = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5], [0.5, -0.5]], dtype=np.float32)
    peaks.add(block)
    mins, maxs = peaks.extent()
    assert maxs[0] == 0.0 and mins[0] == 0.0  # (L+R)/2 = 0


def test_view_decimates_to_max_points():
    peaks = WaveformPeaks(bucket_frames=1, channels=1)
    signal = np.sin(np.linspace(0, 40 * np.pi, 5000)).astype(np.float32).reshape(-1, 1)
    peaks.add(signal)
    mins, maxs = peaks.view(200)
    assert mins.shape[0] <= 200
    assert maxs.shape[0] == mins.shape[0]
    assert maxs.max() <= signal.max() + 1e-6
    assert mins.min() >= signal.min() - 1e-6


def test_view_returns_all_when_small():
    peaks = WaveformPeaks(bucket_frames=100, channels=1)
    peaks.add(np.ones((300, 1), dtype=np.float32))
    mins, maxs = peaks.view(1000)
    assert mins.shape[0] == 3
