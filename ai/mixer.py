"""
VOX Studio — Çok Kanallı Stem Mikseri
Ayrıştırılan ses bileşenlerini bağımsız seviye, pan, mute ve solo kontrolleriyle miksler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.logger import get_logger

logger = get_logger("ai.mixer")


@dataclass
class StemTrack:
    name: str
    audio: np.ndarray  # (samples, channels) or (samples,)
    volume: float = 1.0  # 0.0 - 2.0 (1.0 = 0 dB)
    pan: float = 0.0  # -1.0 (sol) .. +1.0 (sağ)
    is_muted: bool = False
    is_solo: bool = False
    color: str = "#9E2A2B"


class StemMixer:
    """Çok kanallı stem mikseri."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.tracks: Dict[str, StemTrack] = {}

    def add_stem(
        self,
        name: str,
        audio: np.ndarray,
        color: Optional[str] = None,
    ) -> StemTrack:
        default_colors = {
            "Vocals": "#C0392B",
            "Dialogue": "#C0392B",
            "Instrumental": "#2980B9",
            "Music": "#2980B9",
            "Drums": "#E67E22",
            "Bass": "#8E44AD",
            "Other": "#16A085",
            "Ambient": "#27AE60",
        }
        track_color = color or default_colors.get(name, "#7F8C8D")
        track = StemTrack(name=name, audio=audio, color=track_color)
        self.tracks[name] = track
        return track

    def set_mute(self, name: str, muted: bool) -> None:
        if name in self.tracks:
            self.tracks[name].is_muted = muted

    def set_solo(self, name: str, solo: bool) -> None:
        if name in self.tracks:
            self.tracks[name].is_solo = solo

    def set_volume(self, name: str, volume: float) -> None:
        if name in self.tracks:
            self.tracks[name].volume = max(0.0, min(2.0, volume))

    def set_pan(self, name: str, pan: float) -> None:
        if name in self.tracks:
            self.tracks[name].pan = max(-1.0, min(1.0, pan))

    def get_mix(self) -> np.ndarray:
        """Tüm kanalları mevcut mute/solo/fader ayarlarına göre stereo miksler."""
        if not self.tracks:
            return np.zeros((0, 2), dtype=np.float32)

        # En uzun kanalın uzunluğunu bul
        max_len = max(len(t.audio) for t in self.tracks.values())
        mixed = np.zeros((max_len, 2), dtype=np.float32)

        has_any_solo = any(t.is_solo for t in self.tracks.values())

        for track in self.tracks.values():
            # Mute/Solo mantığı
            if track.is_muted:
                continue
            if has_any_solo and not track.is_solo:
                continue

            arr = track.audio
            # Stereo formata getir
            if arr.ndim == 1:
                stereo = np.column_stack([arr, arr])
            else:
                stereo = arr.copy()

            # Uzunluk dengeleme
            if len(stereo) < max_len:
                stereo = np.pad(stereo, ((0, max_len - len(stereo)), (0, 0)))

            # Ses seviyesi (volume fader)
            stereo = stereo * track.volume

            # Pan kontrolü (Constant power pan law)
            # p: -1 .. +1 -> angle: 0 .. pi/2
            angle = (track.pan + 1.0) * (math.pi / 4.0)
            gain_l = math.cos(angle)
            gain_r = math.sin(angle)

            stereo[:, 0] *= gain_l
            stereo[:, 1] *= gain_r

            mixed += stereo

        # Tavan koruması (soft limiting)
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = np.tanh(mixed)

        return mixed.astype(np.float32)
