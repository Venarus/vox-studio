"""
VOX Studio — Ses Dışa Aktarma (Export) Motoru

Formatlar:
1. WAV: 24-bit PCM (varsayılan stüdyo master), 16-bit PCM (CD kalitesi), 32-bit Float.
2. MP3: PyAV (libmp3lame) tabanlı 128 / 192 / 256 / 320 kbps yüksek kaliteli enkoder.
3. FLAC: Kayıpsız sıkıştırılmış 24-bit / 16-bit FLAC enkoder.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import soundfile as sf

from core.logger import get_logger

logger = get_logger("audio.exporter")

try:
    import av
    HAS_PYAV = True
except ImportError:
    av = None
    HAS_PYAV = False


def export_wav(
    audio: np.ndarray,
    fs: int,
    output_path: str | Path,
    bit_depth: int = 24,
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Ses verisini standart uncompressed RIFF WAV olarak kaydeder.
    bit_depth: 16 (PCM_16), 24 (PCM_24), 32 (FLOAT).
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subtype_map = {
        16: "PCM_16",
        24: "PCM_24",
        32: "FLOAT",
    }
    subtype = subtype_map.get(bit_depth, "PCM_24")

    sf.write(str(out_path), audio, int(fs), subtype=subtype, format="WAV")
    logger.info(f"WAV dışa aktarıldı ({subtype}, {fs} Hz): {out_path}")
    return out_path


def export_flac(
    audio: np.ndarray,
    fs: int,
    output_path: str | Path,
    bit_depth: int = 24,
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Ses verisini kayıpsız sıkıştırılmış FLAC olarak kaydeder.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subtype = "PCM_24" if bit_depth == 24 else "PCM_16"
    sf.write(str(out_path), audio, int(fs), subtype=subtype, format="FLAC")
    logger.info(f"FLAC dışa aktarıldı ({subtype}, {fs} Hz): {out_path}")
    return out_path


def export_mp3_pyav(
    audio: np.ndarray,
    fs: int,
    output_path: str | Path,
    bitrate_kbps: int = 320,
    metadata: Optional[Dict[str, str]] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    PyAV libmp3lame ile yüksek kaliteli MP3 dosyası üretir.
    bitrate_kbps: 128, 192, 256, 320.
    """
    if not HAS_PYAV:
        raise RuntimeError("PyAV (av) kütüphanesi kurulu değil; MP3 kodlanamıyor.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    channels = 1 if audio.ndim == 1 else audio.shape[1]
    layout = "mono" if channels == 1 else "stereo"
    bit_rate = int(bitrate_kbps * 1000)

    # Ses verisini float32 ve doğru şekle getir
    data = audio.astype(np.float32)
    n_samples = data.shape[0]

    container = av.open(str(out_path), mode="w", format="mp3")
    container.metadata["encoder"] = "VOX Studio Audio Engine V1"
    if metadata:
        for k, v in metadata.items():
            if v:
                container.metadata[str(k).lower()] = str(v)

    stream = container.add_stream("mp3", rate=int(fs), layout=layout)
    stream.bit_rate = bit_rate

    # MP3 çerçeve boyutu 1152 örnektir
    chunk_size = 1152
    total_chunks = max(1, math.ceil(n_samples / chunk_size))

    for chunk_idx, i in enumerate(range(0, n_samples, chunk_size)):
        if channels == 1:
            chunk = data[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            arr = np.ascontiguousarray(chunk.reshape(1, -1))
        else:
            chunk = data[i : i + chunk_size, :]
            if len(chunk) < chunk_size:
                pad = np.zeros((chunk_size - len(chunk), channels), dtype=np.float32)
                chunk = np.vstack([chunk, pad])
            arr = np.ascontiguousarray(chunk.T)

        frame = av.AudioFrame.from_ndarray(arr, format="fltp", layout=layout)
        frame.sample_rate = int(fs)

        for packet in stream.encode(frame):
            container.mux(packet)

        if progress_cb and chunk_idx % 20 == 0:
            progress_cb(min(0.98, float(chunk_idx) / total_chunks))

    # Kapanış paketleri (flush)
    for packet in stream.encode(None):
        container.mux(packet)

    container.close()
    if progress_cb:
        progress_cb(1.0)

    logger.info(f"MP3 dışa aktarıldı ({bitrate_kbps} kbps, {layout}, {fs} Hz): {out_path}")
    return out_path


def validate_exported_file(path: str | Path) -> Dict[str, Any]:
    """
    Dışa aktarılan dosyanın sağlamlığını, boyutunu ve metadata bilgilerini doğrular.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Dışa aktarılan dosya bulunamadı: {p}")

    size_bytes = p.stat().st_size
    if size_bytes < 100:
        raise ValueError(f"Dışa aktarılan dosya bozuk veya boş ({size_bytes} bayt): {p.name}")

    ext = p.suffix.lower()
    if ext in {".wav", ".flac", ".ogg"}:
        info = sf.info(str(p))
        return {
            "path": str(p),
            "format": info.format,
            "subtype": info.subtype,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "duration": round(info.duration, 3),
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "valid": True,
        }
    elif ext == ".mp3" and HAS_PYAV:
        with av.open(str(p)) as cont:
            audio_stream = cont.streams.audio[0]
            dur = float(cont.duration / av.time_base) if cont.duration else 0.0
            return {
                "path": str(p),
                "format": "MP3",
                "subtype": f"{audio_stream.bit_rate // 1000}k" if audio_stream.bit_rate else "CBR",
                "samplerate": audio_stream.rate,
                "channels": audio_stream.channels,
                "duration": round(dur, 3),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "valid": True,
            }
    else:
        return {
            "path": str(p),
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "valid": True,
        }
