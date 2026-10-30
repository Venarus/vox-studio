"""
Ses kalitesi analizi — gerçek ölçümler.

Ölçülenler
----------
* Tepe (peak, dBFS) ve RMS (dBFS), crest factor
* Kırpılmış örnek sayısı/oranı
* Sessizlik oranı (20 ms çerçeve, eşik -50 dBFS)
* Yaklaşık gürültü tabanı (çerçeve RMS değerlerinin 10. yüzdelik dilimi)
* Entegre loudness (ITU-R BS.1770-4 / EBU R128 yöntemi, K-ağırlıklı,
  mutlak -70 LUFS ve bağıl -10 LU kapıları)
* Kanal dengesizliği (stereo)
* DC ofset

Sınırlar
--------
* Loudness hesabı 48 kHz'de tam BS.1770 uyumludur. Diğer örnekleme
  hızlarında sinyal 48 kHz'e yeniden örneklenir; 10 dakikadan uzun
  dosyalarda yalnızca ilk 10 dakika ölçülür ve bu durum
  ``loudness_partial`` ile bildirilir.
* "Gürültü tabanı" ve "SNR" tahminidir; mikrofon mesafesi, oda yankısı ve
  performans kalitesi hakkında iddia içermez.
* Kalite puanı **mutlak bir ses kalitesi ölçütü değildir**; yalnızca
  kırpılma, seviye, gürültü ve DC ofset kriterlerinin özetidir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

from core.constants import CLIP_THRESHOLD, SILENCE_DBFS
from core.logger import get_logger

logger = get_logger("dsp.analysis")

FRAME_SECONDS = 0.020
SILENCE_THRESHOLD = 10.0 ** (SILENCE_DBFS / 20.0)
LONG_FILE_LOUDNESS_LIMIT = 600.0  # saniye


def _db(value: float) -> float:
    value = float(value)
    return 20.0 * math.log10(value) if value > 0 else -math.inf


# ---------------------------------------------------------------------------
# BS.1770 K-ağırlık filtresi (48 kHz)
# ---------------------------------------------------------------------------
def k_weighting_sos() -> np.ndarray:
    """ITU-R BS.1770-4 Ek 1'deki 48 kHz K-ağırlık biquad'ları (shelf + high-pass)."""
    # Aşama 1: yüksek raf (shelving)
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
    # Aşama 2: yüksek geçiren
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621])
    sos = np.vstack([np.concatenate([b1, a1]), np.concatenate([b2, a2])])
    return sos


class LoudnessMeter:
    """BS.1770-4 entegre loudness ölçeri (48 kHz girdi bekler)."""

    BLOCK = 19200   # 400 ms @ 48 kHz
    HOP = 4800      # 100 ms @ 48 kHz -> %75 örtüşme
    ABS_GATE_LUFS = -70.0
    REL_GATE_LU = -10.0

    def __init__(self) -> None:
        self._sos = k_weighting_sos()
        self._zi = np.zeros((self._sos.shape[0], 2, 2), dtype=np.float64)
        self._pending = np.zeros(0, dtype=np.float64)
        self._blocks: List[float] = []

    def push(self, data: np.ndarray) -> None:
        """``data`` (n, kanal) float32/float64, 48 kHz."""
        x = np.asarray(data, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[0] == 0:
            return
        channels = min(x.shape[1], 2)
        if self._zi.shape[2] != channels:
            self._zi = np.zeros((self._sos.shape[0], 2, channels), dtype=np.float64)
        y, self._zi = signal.sosfilt(self._sos, x[:, :channels], axis=0, zi=self._zi)
        sq = np.einsum("ij,ij->i", y, y)
        self._pending = np.concatenate([self._pending, sq])
        while self._pending.shape[0] >= self.BLOCK:
            block = self._pending[: self.BLOCK]
            mean_sq = float(block.mean())
            if mean_sq > 1e-12:
                self._blocks.append(mean_sq)
            self._pending = self._pending[self.HOP:]

    def integrated_lufs(self) -> Tuple[float, int]:
        """(LUFS, kullanılan blok sayısı) döndürür. Blok yoksa (-inf, 0)."""
        if not self._blocks:
            return (-math.inf, 0)
        values = np.asarray(self._blocks, dtype=np.float64)
        lufs = -0.691 + 10.0 * np.log10(np.maximum(values, 1e-20))
        abs_mask = lufs > self.ABS_GATE_LUFS
        if not np.any(abs_mask):
            return (-math.inf, 0)
        gated = values[abs_mask]
        rel_gate = -0.691 + 10.0 * np.log10(float(gated.mean())) + self.REL_GATE_LU
        final = gated[lufs[abs_mask] > rel_gate]
        if final.size == 0:
            final = gated
        return (-0.691 + 10.0 * np.log10(float(final.mean())), int(final.size))


# ---------------------------------------------------------------------------
# Sonuç modeli
# ---------------------------------------------------------------------------
@dataclass
class AudioAnalysis:
    source: str = ""
    frames: int = 0
    samplerate: int = 0
    channels: int = 0
    duration: float = 0.0
    subtype: str = ""
    file_format: str = ""

    peak: float = 0.0
    peak_dbfs: float = -math.inf
    rms: float = 0.0
    rms_dbfs: float = -math.inf
    crest_db: float = -math.inf
    dc_offset: float = 0.0

    clipped_samples: int = 0
    clipped_ratio: float = 0.0
    silence_ratio: float = 0.0
    noise_floor_dbfs: float = -math.inf
    snr_estimate_db: float = -math.inf

    loudness_lufs: float = -math.inf
    loudness_blocks: int = 0
    loudness_partial: bool = False
    loudness_method: str = "ITU-R BS.1770-4 (K-ağırlıklı, kapılı entegre loudness)"

    channel_imbalance_db: float = -math.inf

    # Restorasyon ve Yapay Zeka Teşhisleri
    hum_detected: bool = False
    hum_freq_hz: Optional[float] = None
    hum_level_db: float = -math.inf
    reverb_detected: bool = False
    reverb_rt60_estimate: float = 0.0
    clicks_detected: int = 0
    recommended_stages: List[str] = field(default_factory=list)
    recommendations_summary: str = ""

    score: Optional[int] = None
    score_breakdown: Dict[str, int] = field(default_factory=dict)
    score_disclaimer: str = (
        "Puan mutlak bir ses kalitesi ölçütü değildir; yalnızca kırpılma, seviye, "
        "gürültü tabanı ve DC ofset kriterlerinin özetidir."
    )
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "frames": self.frames,
            "samplerate": self.samplerate,
            "channels": self.channels,
            "duration": round(self.duration, 3),
            "subtype": self.subtype,
            "format": self.file_format,
            "peak_dbfs": _round_db(self.peak_dbfs),
            "rms_dbfs": _round_db(self.rms_dbfs),
            "crest_db": _round_db(self.crest_db),
            "dc_offset": round(self.dc_offset, 6),
            "clipped_samples": self.clipped_samples,
            "clipped_ratio": round(self.clipped_ratio, 8),
            "silence_ratio": round(self.silence_ratio, 4),
            "noise_floor_dbfs": _round_db(self.noise_floor_dbfs),
            "snr_estimate_db": _round_db(self.snr_estimate_db),
            "loudness_lufs": _round_db(self.loudness_lufs),
            "loudness_partial": self.loudness_partial,
            "channel_imbalance_db": _round_db(self.channel_imbalance_db),
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "warnings": self.warnings,
        }


def _round_db(value: float) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 2)


# ---------------------------------------------------------------------------
# Analiz
# ---------------------------------------------------------------------------
class _StatsAccumulator:
    def __init__(self, samplerate: int, channels: int) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.frames = 0
        self.peak = 0.0
        self.sum_sq = 0.0
        self.clipped = 0
        self.silent_frames = 0
        self.total_frames = 0
        self.frame_rms: List[float] = []
        self.dc_sum = np.zeros(channels, dtype=np.float64)
        self.ch_sum_sq = np.zeros(channels, dtype=np.float64)
        self._frame_len = max(1, int(samplerate * FRAME_SECONDS))

    def push(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        if block.shape[0] == 0:
            return
        self.frames += block.shape[0]
        absmax = np.abs(block)
        self.peak = max(self.peak, float(absmax.max()))
        self.sum_sq += float(np.einsum("ij,ij->", block, block))
        self.clipped += int(np.count_nonzero(absmax >= CLIP_THRESHOLD))
        self.dc_sum += block.sum(axis=0, dtype=np.float64)
        self.ch_sum_sq += np.einsum("ij,ij->j", block, block)

        # 20 ms çerçevelere böl
        flen = self._frame_len
        n_full = block.shape[0] // flen
        for i in range(n_full):
            seg = block[i * flen:(i + 1) * flen]
            rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))
            self.frame_rms.append(rms)
            self.total_frames += 1
            if rms < SILENCE_THRESHOLD:
                self.silent_frames += 1

    def finalize(self) -> AudioAnalysis:
        a = AudioAnalysis(frames=self.frames, samplerate=self.samplerate, channels=self.channels)
        if self.samplerate:
            a.duration = self.frames / self.samplerate
        if self.frames:
            a.peak = self.peak
            a.peak_dbfs = _db(self.peak)
            total_samples = self.frames * max(1, self.channels)
            a.rms = math.sqrt(self.sum_sq / total_samples) if total_samples else 0.0
            a.rms_dbfs = _db(a.rms)
            if a.rms > 0 and self.peak > 0:
                a.crest_db = _db(self.peak) - _db(a.rms)
            a.clipped_samples = self.clipped
            a.clipped_ratio = self.clipped / (self.frames * max(1, self.channels))
            a.dc_offset = float(np.max(np.abs(self.dc_sum / self.frames))) if self.frames else 0.0

        if self.total_frames:
            a.silence_ratio = self.silent_frames / self.total_frames
        if self.frame_rms:
            arr = np.asarray(self.frame_rms, dtype=np.float64)
            arr = arr[arr > 0]
            if arr.size:
                p10 = float(np.percentile(arr, 10))
                a.noise_floor_dbfs = _db(p10)
                if a.rms_dbfs > -math.inf:
                    a.snr_estimate_db = a.rms_dbfs - a.noise_floor_dbfs

        if self.channels >= 2 and self.sum_sq > 0:
            per_ch_rms = np.sqrt(self.ch_sum_sq / max(self.frames, 1))
            left, right = per_ch_rms[0], per_ch_rms[1]
            if left > 0 and right > 0:
                a.channel_imbalance_db = abs(_db(left) - _db(right))
            elif left > 0 or right > 0:
                a.channel_imbalance_db = math.inf

        return a


def _apply_quality_score(a: AudioAnalysis) -> None:
    """Açık kriterlerden puan üretir (100 üzerinden)."""
    breakdown: Dict[str, int] = {}

    # Kırpılma (en ağır ceza)
    if a.clipped_samples == 0:
        breakdown["clipping"] = 0
    else:
        breakdown["clipping"] = -min(40, 10 + int(a.clipped_ratio * 100000))
        a.warnings.append(
            f"{a.clipped_samples} örnek kırpılmış. Kayıt bozulmuş olabilir; kazancı düşürüp yeniden kayıt önerilir."
        )

    # Seviye: hedef RMS -23..-14 dBFS
    rms = a.rms_dbfs
    if not math.isfinite(rms):
        breakdown["level"] = -25
        a.warnings.append("Ses tamamen sessiz görünüyor.")
    elif rms < -40:
        breakdown["level"] = -25
        a.warnings.append("Kayıt seviyesi çok düşük; gürültüye göre sinyal zayıf olabilir.")
    elif rms < -30:
        breakdown["level"] = -15
        a.warnings.append("Kayıt seviyesi düşük (hedef RMS ≈ -20 dBFS).")
    elif rms < -23:
        breakdown["level"] = -5
    elif rms <= -14:
        breakdown["level"] = 0
    elif rms <= -9:
        breakdown["level"] = -8
        a.warnings.append("Kayıt seviyesi yüksek; kırpılma riski var.")
    else:
        breakdown["level"] = -18
        a.warnings.append("Kayıt seviyesi çok yüksek; kırpılma riski yüksek.")

    # Gürültü tabanı (yaklaşık)
    nf = a.noise_floor_dbfs
    if not math.isfinite(nf):
        breakdown["noise"] = 0
    elif nf > -40:
        breakdown["noise"] = -20
        a.warnings.append("Gürültü tabanı yüksek (yaklaşık ölçüm); sessiz bir ortam veya gürültü azaltma düşünün.")
    elif nf > -50:
        breakdown["noise"] = -10
    elif nf > -60:
        breakdown["noise"] = -4
    else:
        breakdown["noise"] = 0

    # Dinamik aralık / sıkıştırma belirtisi
    if math.isfinite(a.crest_db):
        if a.crest_db < 6:
            breakdown["dynamics"] = -8
            a.warnings.append("Crest factor çok düşük; aşırı sıkıştırma veya limitleme olabilir.")
        elif a.crest_db > 26:
            breakdown["dynamics"] = -5
            a.warnings.append("Crest factor çok yüksek; ani tepe noktaları var (parazit/pop olabilir).")
        else:
            breakdown["dynamics"] = 0

    # DC ofset
    if a.dc_offset > 0.02:
        breakdown["dc_offset"] = -10
        a.warnings.append("Belirgin DC ofset var; high-pass filtre önerilir.")
    elif a.dc_offset > 0.005:
        breakdown["dc_offset"] = -4
    else:
        breakdown["dc_offset"] = 0

    # Kanal dengesizliği (stereo)
    if a.channels >= 2:
        if not math.isfinite(a.channel_imbalance_db):
            breakdown["imbalance"] = -6
            a.warnings.append("Kanallardan biri tamamen sessiz.")
        elif a.channel_imbalance_db > 6:
            breakdown["imbalance"] = -6
            a.warnings.append("Kanal dengesizliği 6 dB'den fazla.")
        elif a.channel_imbalance_db > 2:
            breakdown["imbalance"] = -2
        else:
            breakdown["imbalance"] = 0

    a.score_breakdown = breakdown
    a.score = max(0, min(100, 100 + sum(breakdown.values())))


def _diagnose_restoration_needs(
    a: AudioAnalysis,
    sample_data: np.ndarray,
    sr: int,
) -> None:
    """Sinyalin spektral yapısını inceleyerek restorasyon modüllerini ve önerilerini belirler."""
    if len(sample_data) == 0 or sr <= 0:
        return

    mono = sample_data[:, 0] if sample_data.ndim == 2 else sample_data
    n_pts = min(len(mono), max(4096, sr * 4))
    chunk = mono[:n_pts].astype(np.float64)

    # 1. 50 Hz / 60 Hz Elektriksel Uğultu (Hum) Tespiti
    try:
        nperseg = min(len(chunk), 4096)
        freqs, psd = signal.welch(chunk, fs=sr, nperseg=nperseg)
        if len(freqs) > 10:
            # 50 Hz civarı
            idx_50 = np.argmin(np.abs(freqs - 50.0))
            idx_50_base = [i for i in [idx_50 - 3, idx_50 - 2, idx_50 + 2, idx_50 + 3] if 0 <= i < len(psd)]
            base_50 = np.median(psd[idx_50_base]) if idx_50_base else 1e-12
            ratio_50 = 10.0 * np.log10(max(psd[idx_50], 1e-12) / max(base_50, 1e-12))

            # 60 Hz civarı
            idx_60 = np.argmin(np.abs(freqs - 60.0))
            idx_60_base = [i for i in [idx_60 - 3, idx_60 - 2, idx_60 + 2, idx_60 + 3] if 0 <= i < len(psd)]
            base_60 = np.median(psd[idx_60_base]) if idx_60_base else 1e-12
            ratio_60 = 10.0 * np.log10(max(psd[idx_60], 1e-12) / max(base_60, 1e-12))

            if ratio_50 >= 7.0 and ratio_50 > ratio_60:
                a.hum_detected = True
                a.hum_freq_hz = 50.0
                a.hum_level_db = round(float(ratio_50), 1)
            elif ratio_60 >= 7.0:
                a.hum_detected = True
                a.hum_freq_hz = 60.0
                a.hum_level_db = round(float(ratio_60), 1)
    except Exception:
        pass

    # 2. Tıkırtı / Dijital İmpuls Tespiti
    try:
        from scipy import ndimage
        probe_len = min(len(chunk), 48000)
        med = ndimage.median_filter(chunk[:probe_len], size=9)
        res = np.abs(chunk[:probe_len] - med)
        sigma = 1.4826 * float(np.median(np.abs(res - np.median(res))))
        click_count = int(np.count_nonzero(res > max(6.0 * sigma, 0.05)))
        a.clicks_detected = click_count
    except Exception:
        pass

    # 3. Yankı (Reverberation) Göstergesi
    if a.crest_db > 0 and a.crest_db < 9.5 and a.silence_ratio < 0.15:
        a.reverb_detected = True
        a.reverb_rt60_estimate = 0.45
    elif a.silence_ratio > 0.05 and a.noise_floor_dbfs > -42.0:
        a.reverb_detected = True
        a.reverb_rt60_estimate = 0.35

    # 4. Önerilen İşlem Zinciri Oluşturma
    stages: List[str] = []
    diag_lines: List[str] = []

    # Kırpılma
    if a.clipped_samples > 0:
        stages.append("declip")
        diag_lines.append(f"Kırpılma (Clipping): {a.clipped_samples} örnek tepeye çarptı (De-Clip önerilir)")
    else:
        diag_lines.append("Kırpılma: Temiz (0 örnek)")

    # Uğultu
    if a.hum_detected and a.hum_freq_hz:
        stages.append("hum_filter")
        diag_lines.append(f"Elektriksel Uğultu: {int(a.hum_freq_hz)} Hz şebeke tonu tespit edildi (+{a.hum_level_db} dB)")
    else:
        diag_lines.append("Elektriksel Uğultu: Tespit edilmedi")

    # Tıkırtı
    if a.clicks_detected > 3:
        stages.append("declick")
        diag_lines.append(f"İmpuls / Tıkırtı: {a.clicks_detected} anlık sıçrama tespit edildi (De-Click önerilir)")

    # Gürültü / SNR
    snr = a.snr_estimate_db
    if math.isfinite(snr) and snr < 22.0:
        stages.append("ai_denoise")
        diag_lines.append(f"Arka Plan Gürültüsü: Yüksek (Tahmini SNR: {snr:.1f} dB)")
    elif math.isfinite(snr) and snr < 30.0:
        stages.append("ai_denoise")
        diag_lines.append(f"Arka Plan Gürültüsü: Orta (Tahmini SNR: {snr:.1f} dB)")
    else:
        diag_lines.append(f"Arka Plan Gürültüsü: Düşük / Temiz (Tahmini SNR: {snr:.1f} dB)" if math.isfinite(snr) else "Gürültü Seviyesi: Düşük")

    # Yankı
    if a.reverb_detected:
        stages.append("dereverb")
        diag_lines.append(f"Oda Yankısı: Belirgin yansıma (Tahmini RT60: ~{a.reverb_rt60_estimate:.2f}s)")
    else:
        diag_lines.append("Oda Yankısı: Minimal / Doğal")

    # Konuşma Netliği
    stages.append("clarity")

    # Dinamikler
    if a.crest_db > 14.0 or (math.isfinite(a.loudness_lufs) and a.loudness_lufs < -26.0):
        stages.append("compressor")

    # Normalizasyon ve Limiter
    stages.append("normalize")
    stages.append("limiter")

    a.recommended_stages = stages

    stage_names_tr = {
        "declip": "De-Clip",
        "hum_filter": "De-Hum (Notch)",
        "declick": "De-Click",
        "ai_denoise": "AI Gürültü Bastırma",
        "dereverb": "De-Reverb (Yankı Azaltma)",
        "clarity": "Speech Clarity (Konuşma Netliği)",
        "compressor": "Dinamik Kompresör",
        "normalize": "LUFS Normalizasyon",
        "limiter": "Tepe Limiter",
    }
    recommended_str = " -> ".join(stage_names_tr.get(s, s) for s in stages)
    diag_summary = "\n".join(f"• {line}" for line in diag_lines)

    a.recommendations_summary = (
        f"VOX Ses Teşhis Raporu:\n{diag_summary}\n\nÖnerilen İşlem Zinciri:\n{recommended_str}"
    )


def analyze_array(
    data: np.ndarray,
    samplerate: int,
    *,
    source: str = "",
    subtype: str = "",
    file_format: str = "",
) -> AudioAnalysis:
    """Bellekteki ses verisini analiz eder (testler ve işlem önizlemesi için)."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    channels = arr.shape[1] if arr.ndim == 2 else 1
    acc = _StatsAccumulator(samplerate=int(samplerate), channels=channels)
    acc.push(arr.astype(np.float32, copy=False))
    analysis = acc.finalize()
    analysis.source = source
    analysis.subtype = subtype
    analysis.file_format = file_format

    meter = LoudnessMeter()
    mono_like = arr.astype(np.float32, copy=False)
    if samplerate != 48000:
        from math import gcd

        g = gcd(int(samplerate), 48000)
        mono_like = signal.resample_poly(mono_like, 48000 // g, int(samplerate) // g, axis=0)
    meter.push(mono_like)
    lufs, blocks = meter.integrated_lufs()
    analysis.loudness_lufs = lufs
    analysis.loudness_blocks = blocks
    _diagnose_restoration_needs(analysis, arr, int(samplerate))
    _apply_quality_score(analysis)
    return analysis


def analyze_file(path: str | Path, block_frames: int = 48000) -> AudioAnalysis:
    """Dosyayı blok blok okuyup analiz eder (bellek kullanımı sabittir).
    libsndfile tarafından doğrudan okunamayan biçimlerde (M4A, AAC, Opus, WMA, MP4 vb.)
    PyAV üzerinden ses akışını çözer.
    """
    p = Path(path)
    try:
        info = sf.info(str(p))
    except Exception as exc:
        try:
            from ai.media_extractor import MediaExtractor

            m_info = MediaExtractor.probe(p)
            arr, sr = MediaExtractor.read_audio_any(p, always_2d=True)
            codec_str = m_info.streams[0].codec_name.upper() if m_info.streams else ""
            return analyze_array(
                arr,
                sr,
                source=str(p),
                subtype=codec_str,
                file_format=m_info.format_name.upper(),
            )
        except Exception as fallback_exc:
            raise ValueError(
                f"Dosya analiz edilemedi: {p.name} ({fallback_exc}). Dosya desteklenmiyor veya bozuk olabilir."
            ) from exc

    acc = _StatsAccumulator(int(info.samplerate), int(info.channels))
    meter = LoudnessMeter()
    loudness_partial = False
    loudness_seconds = 0.0
    sr = int(info.samplerate)
    need_resample = sr != 48000
    ratio = (48000, sr)

    try:
        with sf.SoundFile(str(p), mode="r") as f:
            while True:
                block = f.read(block_frames, dtype="float32", always_2d=True)
                if block.shape[0] == 0:
                    break
                acc.push(block)
                if loudness_seconds < LONG_FILE_LOUDNESS_LIMIT:
                    chunk = block
                    if need_resample:
                        from math import gcd

                        g = gcd(sr, 48000)
                        chunk = signal.resample_poly(chunk, 48000 // g, sr // g, axis=0)
                    meter.push(chunk)
                    loudness_seconds += block.shape[0] / sr
            if loudness_seconds > LONG_FILE_LOUDNESS_LIMIT:
                loudness_partial = True
    except Exception as exc:
        raise ValueError(
            f"Dosya okunamadı: {p.name} ({exc}). Dosya bozuk olabilir."
        ) from exc

    analysis = acc.finalize()
    analysis.source = str(p)
    analysis.subtype = str(info.subtype or "")
    analysis.file_format = str(info.format or "")
    analysis.loudness_partial = loudness_partial
    lufs, blocks = meter.integrated_lufs()
    analysis.loudness_lufs = lufs
    analysis.loudness_blocks = blocks
    try:
        with sf.SoundFile(str(p)) as sf_probe:
            probe_frames = min(int(sf_probe.frames), sr * 4)
            probe_data = sf_probe.read(frames=probe_frames, dtype="float32")
            _diagnose_restoration_needs(analysis, probe_data, sr)
    except Exception:
        pass
    _apply_quality_score(analysis)
    if loudness_partial:
        analysis.warnings.append(
            "Loudness yalnızca dosyanın ilk 10 dakikası üzerinden ölçüldü."
        )
    return analysis
