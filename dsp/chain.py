"""
VOX Studio — Çok Katmanlı Modüler Ses Restorasyon Zinciri (Audio Restoration Chain)

Aşamalar (Sıralanabilir ve Açılıp Kapatılabilir):
1. declip: Kırpılmış tepe noktalarını onarma (Cubic Spline Restorasyonu)
2. hum_filter: 50/60 Hz elektriksel şebeke ve harmonik çentik filtresi
3. declick: Anlık mikrofon tıkırtısı ve dijital sıçrama temizliği
4. ai_denoise: Yapay zeka sinir ağı konuşma temizleme (GTCRN / RNNoise ONNX)
5. denoise: Spektral çıkarma (STFT Oversubtraction DSP Fallback)
6. dereverb: Akustik oda yankısı ve çınlama azaltma (Late reflection suppression)
7. clarity: Speech Clarity Engine (Konuşma varlığı ve anlaşılırlık takviyesi)
8. eq: Parametrik Ton EQ
9. deesser: Sibilans ve sert "s/ş" sesleri kontrolü
10. compressor: Dinamik aralık kompresörü ve gürültü kapısı
11. gain_pan_fade: Seviye, pan ve giriş/çıkış geçişleri
12. normalize: ITU-R BS.1770-4 Entegre LUFS ve True Peak normalizasyonu
13. limiter: Tepe distorsiyon engelleyici brickwall limiter
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from core.logger import get_logger
from dsp.denoise import DenoiseSettings, reduce_noise_spectral
from dsp.dereverb import DereverbSettings, dereverb_spectral
from dsp.dynamics import (
    CompressorSettings,
    DeEsserSettings,
    LimiterSettings,
    apply_compressor,
    apply_deesser,
    apply_limiter,
)
from dsp.filters import EQBand, HumFilterSettings, apply_eq, apply_hum_filter
from dsp.normalize import apply_gain_pan_fade, normalize_lufs, normalize_peak, trim_silence
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

logger = get_logger("dsp.chain")

DEFAULT_STAGE_ORDER: Tuple[str, ...] = (
    "declip",
    "hum_filter",
    "declick",
    "ai_denoise",
    "denoise",
    "dereverb",
    "clarity",
    "eq",
    "deesser",
    "compressor",
    "gain_pan_fade",
    "normalize",
    "limiter",
)

STAGE_DISPLAY_NAMES: Dict[str, str] = {
    "declip": "De-Clip (Kırpılma Onarımı)",
    "hum_filter": "De-Hum / Notch Filtre (50/60 Hz)",
    "declick": "De-Click / De-Pop (Tıkırtı & Patlama Temizliği)",
    "ai_denoise": "Yapay Zeka Konuşma Temizleme (GTCRN ONNX)",
    "denoise": "Spektral Gürültü Azaltma (DSP)",
    "dereverb": "De-Reverb (Oda Yankısı Azaltma)",
    "clarity": "Speech Clarity (Konuşma Netliği)",
    "eq": "Parametrik Ton EQ",
    "deesser": "De-Esser (Sibilans Kontrolü)",
    "compressor": "Dinamik Kompresör",
    "gain_pan_fade": "Kazanç (Gain), Pan & Fade",
    "normalize": "Seviye Normalizasyonu (LUFS / Peak)",
    "limiter": "Brickwall Tepe Limiter",
}


@dataclass
class AIDenoiseSettings:
    enabled: bool = True
    model: str = "gtcrn"               # 'gtcrn' veya 'rnnoise'
    strength: float = 0.80             # 0.0 .. 1.0 temizleme gücü
    preserve_high_frequencies: bool = True  # 8 kHz üstü konuşma havasını koru
    blend: float = 1.0                 # 1.0 = %100 temiz, <1.0 kuru/ıslak miks


@dataclass
class ProcessingChainConfig:
    declip: DeclipSettings = field(default_factory=DeclipSettings)
    hum_filter: HumFilterSettings = field(default_factory=HumFilterSettings)
    declick: DeclickSettings = field(default_factory=DeclickSettings)
    plosive: PlosiveSettings = field(default_factory=PlosiveSettings)
    ai_denoise: AIDenoiseSettings = field(default_factory=AIDenoiseSettings)
    denoise: DenoiseSettings = field(default_factory=lambda: DenoiseSettings(enabled=False))
    dereverb: DereverbSettings = field(default_factory=lambda: DereverbSettings(enabled=False))
    clarity: SpeechClaritySettings = field(default_factory=SpeechClaritySettings)

    eq_bands: List[EQBand] = field(default_factory=lambda: [
        EQBand(band_type="highpass", frequency=80.0, q=0.707, enabled=True),
        EQBand(band_type="peaking", frequency=250.0, gain_db=0.0, q=1.0, enabled=False),
        EQBand(band_type="peaking", frequency=3000.0, gain_db=0.0, q=1.0, enabled=False),
        EQBand(band_type="highshelf", frequency=10000.0, gain_db=0.0, q=0.707, enabled=False),
    ])
    eq_enabled: bool = False
    lowpass_freq: float = 18000.0
    lowpass_enabled: bool = False
    vocal_presence_db: float = 0.0

    deesser: DeEsserSettings = field(default_factory=lambda: DeEsserSettings(enabled=True))
    compressor: CompressorSettings = field(default_factory=lambda: CompressorSettings(enabled=True))
    limiter: LimiterSettings = field(default_factory=lambda: LimiterSettings(enabled=True))

    gain_db: float = 0.0
    pan: float = 0.0
    fade_in_sec: float = 0.0
    fade_out_sec: float = 0.0

    normalize_target_dbfs: Optional[float] = None
    normalize_lufs_target: Optional[float] = -16.0  # -16 LUFS yayın standardı
    trim_silence_enabled: bool = False
    trim_threshold_dbfs: float = -50.0

    stage_order: List[str] = field(default_factory=lambda: list(DEFAULT_STAGE_ORDER))

    def clone(self) -> "ProcessingChainConfig":
        return copy.deepcopy(self)

    def active_stage_names(self) -> List[str]:
        active: List[str] = []
        for st in self.stage_order:
            if st == "declip" and self.declip.enabled:
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "hum_filter" and self.hum_filter.enabled:
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "declick" and (self.declick.enabled or self.plosive.enabled):
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "ai_denoise" and self.ai_denoise.enabled:
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "denoise" and self.denoise.enabled:
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "dereverb" and self.dereverb.enabled:
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "clarity" and self.clarity.enabled:
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "eq" and self.eq_enabled:
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "deesser" and self.deesser.enabled:
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "compressor" and self.compressor.enabled:
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "gain_pan_fade" and (abs(self.gain_db) > 0.05 or abs(self.pan) > 0.02):
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "normalize" and (self.normalize_lufs_target is not None or self.normalize_target_dbfs is not None):
                active.append(STAGE_DISPLAY_NAMES[st])
            elif st == "limiter" and self.limiter.enabled:
                active.append(STAGE_DISPLAY_NAMES[st])
        if self.trim_silence_enabled:
            active.append("Sessizlik Kırpma")
        return active


# ==============================================================================
# 3 ANA KALİTE PROFİLİ (NATURAL, BALANCED, MAXIMUM RESTORATION) + ADAPTIVE
# ==============================================================================
def create_natural_profile() -> ProcessingChainConfig:
    """
    Natural Profili:
    Doğal konuşma karakterini, nefesleri ve konuşmacı dinamiğini en üst düzeyde korur.
    Hafif gürültü azaltma, minimum bozulma, -19 LUFS (podcast standartı).
    """
    cfg = ProcessingChainConfig()
    cfg.declip.enabled = False
    cfg.hum_filter.enabled = False
    cfg.declick.enabled = False
    cfg.ai_denoise.enabled = True
    cfg.ai_denoise.strength = 0.55
    cfg.ai_denoise.blend = 0.90
    cfg.denoise.enabled = False
    cfg.dereverb.enabled = False
    cfg.clarity.enabled = True
    cfg.clarity.presence_gain_db = 1.5
    cfg.clarity.boxiness_cut_db = -1.0
    cfg.clarity.air_shelf_db = 1.0
    cfg.deesser.enabled = True
    cfg.deesser.threshold_dbfs = -18.0
    cfg.deesser.max_reduction_db = 4.0
    cfg.compressor.enabled = True
    cfg.compressor.threshold_dbfs = -18.0
    cfg.compressor.ratio = 2.0
    cfg.compressor.attack_ms = 20.0
    cfg.compressor.release_ms = 120.0
    cfg.compressor.makeup_gain_db = 1.0
    cfg.normalize_lufs_target = -19.0
    cfg.limiter.enabled = True
    cfg.limiter.ceiling_dbfs = -1.0
    return cfg


def create_balanced_profile() -> ProcessingChainConfig:
    """
    Balanced Profili (Varsayılan):
    Gürültü azaltma ile doğallık arasındaki ideal altın denge.
    Bilgisayar fanları, AC uğultusu ve dip seslerini temizler, net ve dengeli ses üretir.
    """
    cfg = ProcessingChainConfig()
    cfg.declip.enabled = True
    cfg.hum_filter.enabled = True
    cfg.hum_filter.base_freq_hz = 50.0
    cfg.hum_filter.harmonics = 3
    cfg.declick.enabled = True
    cfg.declick.sensitivity = 0.55
    cfg.ai_denoise.enabled = True
    cfg.ai_denoise.strength = 0.80
    cfg.ai_denoise.blend = 1.0
    cfg.denoise.enabled = False
    cfg.dereverb.enabled = True
    cfg.dereverb.amount = 0.45
    cfg.dereverb.rt60 = 0.35
    cfg.clarity.enabled = True
    cfg.clarity.presence_gain_db = 2.5
    cfg.clarity.boxiness_cut_db = -2.0
    cfg.clarity.air_shelf_db = 1.5
    cfg.deesser.enabled = True
    cfg.deesser.threshold_dbfs = -16.0
    cfg.deesser.max_reduction_db = 6.0
    cfg.compressor.enabled = True
    cfg.compressor.threshold_dbfs = -16.0
    cfg.compressor.ratio = 3.0
    cfg.compressor.attack_ms = 12.0
    cfg.compressor.release_ms = 100.0
    cfg.compressor.makeup_gain_db = 2.0
    cfg.normalize_lufs_target = -16.0
    cfg.limiter.enabled = True
    cfg.limiter.ceiling_dbfs = -1.0
    return cfg


def create_maximum_restoration_profile() -> ProcessingChainConfig:
    """
    Maximum Restoration Profili:
    Ağır gürültülü ortamlar, sokak sesleri, klima ve fan uğultusunun yoğun olduğu
    zorlu kayıtlar için derin temizlik ve kapsamlı sinyal restorasyonu.
    """
    cfg = ProcessingChainConfig()
    cfg.declip.enabled = True
    cfg.hum_filter.enabled = True
    cfg.hum_filter.base_freq_hz = 50.0
    cfg.hum_filter.harmonics = 4
    cfg.declick.enabled = True
    cfg.declick.sensitivity = 0.75
    cfg.plosive.enabled = True
    cfg.ai_denoise.enabled = True
    cfg.ai_denoise.strength = 1.0
    cfg.ai_denoise.blend = 1.0
    cfg.denoise.enabled = False
    cfg.dereverb.enabled = True
    cfg.dereverb.amount = 0.70
    cfg.dereverb.rt60 = 0.50
    cfg.clarity.enabled = True
    cfg.clarity.presence_gain_db = 3.5
    cfg.clarity.boxiness_cut_db = -3.0
    cfg.clarity.air_shelf_db = 2.0
    cfg.deesser.enabled = True
    cfg.deesser.threshold_dbfs = -14.0
    cfg.deesser.max_reduction_db = 8.0
    cfg.compressor.enabled = True
    cfg.compressor.threshold_dbfs = -14.0
    cfg.compressor.ratio = 4.0
    cfg.compressor.attack_ms = 8.0
    cfg.compressor.release_ms = 80.0
    cfg.compressor.makeup_gain_db = 3.0
    cfg.normalize_lufs_target = -16.0
    cfg.limiter.enabled = True
    cfg.limiter.ceiling_dbfs = -1.0
    return cfg


def create_audiobook_profile() -> ProcessingChainConfig:
    """
    Audiobook Profili (ACX / Audible Standartı):
    Sesli kitap anlatımları için endüstri standardı ACX teslim kriterlerini sağlar:
    - Entegre Loudness: -20.0 LUFS (-23 LUFS ile -18 LUFS arası hedefe tam oturur)
    - True Peak Tavanı: -3.0 dBFS (ACX kuralı: peak kesinlikle -3 dBFS'yi aşamaz)
    - Tükürük ve ağız şapırtılarını temizleyen hassas de-clicker
    - Şeffaf opto-stil kompresör (2.2:1) ve sibilans yumuşatma
    """
    cfg = ProcessingChainConfig()
    cfg.declip.enabled = True
    cfg.declip.threshold = 0.98
    cfg.hum_filter.enabled = False
    cfg.declick.enabled = True
    cfg.declick.sensitivity = 0.65
    cfg.plosive.enabled = True
    cfg.plosive.cutoff_hz = 100.0
    cfg.ai_denoise.enabled = True
    cfg.ai_denoise.strength = 0.65
    cfg.ai_denoise.blend = 0.95
    cfg.denoise.enabled = False
    cfg.dereverb.enabled = True
    cfg.dereverb.amount = 0.35
    cfg.dereverb.rt60 = 0.25
    cfg.clarity.enabled = True
    cfg.clarity.presence_gain_db = 1.2
    cfg.clarity.boxiness_cut_db = -1.5
    cfg.clarity.air_shelf_db = 0.8
    cfg.deesser.enabled = True
    cfg.deesser.threshold_dbfs = -18.0
    cfg.deesser.max_reduction_db = 5.0
    cfg.compressor.enabled = True
    cfg.compressor.threshold_dbfs = -20.0
    cfg.compressor.ratio = 2.2
    cfg.compressor.attack_ms = 25.0
    cfg.compressor.release_ms = 140.0
    cfg.compressor.makeup_gain_db = 1.5
    cfg.normalize_lufs_target = -20.0
    cfg.limiter.enabled = True
    cfg.limiter.ceiling_dbfs = -3.0
    return cfg


def create_broadcast_profile() -> ProcessingChainConfig:
    """
    Broadcast Profili (EBU R128 / ATSC A/85):
    Televizyon, radyo ve profesyonel yayın kuruluşları için standart ses zinciri:
    - Entegre Loudness: -23.0 LUFS
    - True Peak Tavanı: -1.0 dBFS
    - 50/60 Hz şebeke koruması ve güçlü konuşma varlığı
    """
    cfg = ProcessingChainConfig()
    cfg.declip.enabled = True
    cfg.hum_filter.enabled = True
    cfg.hum_filter.base_freq_hz = 50.0
    cfg.hum_filter.harmonics = 3
    cfg.declick.enabled = True
    cfg.declick.sensitivity = 0.50
    cfg.ai_denoise.enabled = True
    cfg.ai_denoise.strength = 0.85
    cfg.ai_denoise.blend = 1.0
    cfg.denoise.enabled = False
    cfg.dereverb.enabled = True
    cfg.dereverb.amount = 0.40
    cfg.dereverb.rt60 = 0.30
    cfg.clarity.enabled = True
    cfg.clarity.presence_gain_db = 2.8
    cfg.clarity.boxiness_cut_db = -2.5
    cfg.clarity.air_shelf_db = 1.5
    cfg.deesser.enabled = True
    cfg.deesser.threshold_dbfs = -15.0
    cfg.deesser.max_reduction_db = 6.0
    cfg.compressor.enabled = True
    cfg.compressor.threshold_dbfs = -16.0
    cfg.compressor.ratio = 3.5
    cfg.compressor.attack_ms = 10.0
    cfg.compressor.release_ms = 90.0
    cfg.compressor.makeup_gain_db = 2.5
    cfg.normalize_lufs_target = -23.0
    cfg.limiter.enabled = True
    cfg.limiter.ceiling_dbfs = -1.0
    return cfg


def create_voiceover_profile() -> ProcessingChainConfig:
    """
    Voiceover & Trailer Profili (Ticari Seslendirme):
    Reklam, YouTube anlatımı ve sinematik fragmanlar için parlak, dolgun ve vurucu ses karakteri:
    - Entegre Loudness: -16.0 LUFS
    - True Peak Tavanı: -1.0 dBFS
    - 12 kHz hava rafı (+2.2 dB) ve 3.8 kHz varlık takviyesi
    - Hızlı ataklı dinamik kompresör
    """
    cfg = ProcessingChainConfig()
    cfg.declip.enabled = True
    cfg.hum_filter.enabled = True
    cfg.hum_filter.base_freq_hz = 50.0
    cfg.hum_filter.harmonics = 2
    cfg.declick.enabled = True
    cfg.declick.sensitivity = 0.60
    cfg.plosive.enabled = True
    cfg.ai_denoise.enabled = True
    cfg.ai_denoise.strength = 0.75
    cfg.ai_denoise.blend = 1.0
    cfg.denoise.enabled = False
    cfg.dereverb.enabled = True
    cfg.dereverb.amount = 0.35
    cfg.dereverb.rt60 = 0.25
    cfg.clarity.enabled = True
    cfg.clarity.presence_gain_db = 3.0
    cfg.clarity.presence_freq_hz = 3800.0
    cfg.clarity.boxiness_cut_db = -2.0
    cfg.clarity.air_shelf_db = 2.2
    cfg.deesser.enabled = True
    cfg.deesser.threshold_dbfs = -16.0
    cfg.deesser.max_reduction_db = 7.0
    cfg.compressor.enabled = True
    cfg.compressor.threshold_dbfs = -15.0
    cfg.compressor.ratio = 3.2
    cfg.compressor.attack_ms = 12.0
    cfg.compressor.release_ms = 110.0
    cfg.compressor.makeup_gain_db = 2.5
    cfg.normalize_lufs_target = -16.0
    cfg.limiter.enabled = True
    cfg.limiter.ceiling_dbfs = -1.0
    return cfg


PROFILES: Dict[str, Callable[[], ProcessingChainConfig]] = {
    "natural": create_natural_profile,
    "balanced": create_balanced_profile,
    "maximum": create_maximum_restoration_profile,
    "audiobook": create_audiobook_profile,
    "broadcast": create_broadcast_profile,
    "voiceover": create_voiceover_profile,
}


def get_profile(name: str) -> ProcessingChainConfig:
    """İsme göre profil nesnesi döndürür. Bilinmeyen isimlerde 'balanced' döner."""
    factory = PROFILES.get(name.lower().strip())
    if factory is not None:
        return factory()
    logger.warning("Bilinmeyen profil adı '%s'; 'balanced' profiline geçiliyor.", name)
    return create_balanced_profile()


def list_profiles() -> List[str]:
    """Kayıtlı profil adlarının listesini döndürür."""
    return list(PROFILES.keys())


def create_adaptive_profile(analysis: Any) -> ProcessingChainConfig:
    """
    Adaptive Restoration ("Maksimum Kaliteyle Temizle"):
    Akıllı analiz motorunun ölçüm sonuçlarına göre sadece gerekli modülleri
    ve en uygun parametreleri dinamik olarak yapılandırır.
    """
    cfg = create_balanced_profile()

    # Kırpılma
    if getattr(analysis, "clipped_samples", 0) > 0:
        cfg.declip.enabled = True
    else:
        cfg.declip.enabled = False

    # Şebeke Uğultusu
    if getattr(analysis, "hum_detected", False) and getattr(analysis, "hum_freq_hz", None):
        cfg.hum_filter.enabled = True
        cfg.hum_filter.base_freq_hz = float(analysis.hum_freq_hz)
    else:
        cfg.hum_filter.enabled = False

    # Tıkırtı
    if getattr(analysis, "clicks_detected", 0) > 3:
        cfg.declick.enabled = True
    else:
        cfg.declick.enabled = False

    # AI Denoise gücünü SNR'a göre uyarla
    snr = getattr(analysis, "snr_estimate_db", 30.0)
    if not math.isfinite(snr):
        snr = 25.0

    if snr < 15.0:
        cfg.ai_denoise.strength = 1.0
    elif snr < 22.0:
        cfg.ai_denoise.strength = 0.85
    elif snr < 28.0:
        cfg.ai_denoise.strength = 0.70
    else:
        cfg.ai_denoise.strength = 0.50

    # Oda Yankısı
    if getattr(analysis, "reverb_detected", False):
        cfg.dereverb.enabled = True
        cfg.dereverb.rt60 = max(0.2, getattr(analysis, "reverb_rt60_estimate", 0.4))
    else:
        cfg.dereverb.enabled = False

    return cfg


def compute_detailed_metrics(audio: np.ndarray, fs: float) -> Dict[str, Any]:
    """
    Öncesi / Sonrası karşılaştırma tablosu için gerçek sinyal metrikleri:
    Peak dBFS, RMS dBFS, Entegre LUFS, Crest Factor, Gürültü Tabanı, Kırpılma ve Süre.
    """
    from dsp.analysis import analyze_array

    rep = analyze_array(audio.astype(np.float32, copy=False), int(fs))
    mag = np.abs(audio) if audio.ndim == 1 else np.max(np.abs(audio), axis=1)
    clip_count = int(np.sum(mag >= 0.999))

    noise_floor_db = (
        round(float(rep.noise_floor_dbfs), 1)
        if math.isfinite(rep.noise_floor_dbfs)
        else round(float(rep.rms_dbfs) - 18.0, 1) if math.isfinite(rep.rms_dbfs) else -70.0
    )

    return {
        "duration_sec": round(float(rep.duration), 3),
        "sample_rate": int(rep.samplerate),
        "channels": int(rep.channels),
        "peak_dbfs": round(float(rep.peak_dbfs), 2) if math.isfinite(rep.peak_dbfs) else -96.0,
        "rms_dbfs": round(float(rep.rms_dbfs), 2) if math.isfinite(rep.rms_dbfs) else -96.0,
        "lufs": round(float(rep.loudness_lufs), 2) if math.isfinite(rep.loudness_lufs) else -70.0,
        "crest_db": round(float(rep.crest_db), 2) if math.isfinite(rep.crest_db) else 0.0,
        "noise_floor_dbfs": noise_floor_db,
        "clipping_samples": max(int(rep.clipped_samples), clip_count),
        "clipped": bool(rep.clipped_samples > 0 or clip_count > 0),
        "snr_db": round(float(rep.snr_estimate_db), 1) if math.isfinite(rep.snr_estimate_db) else None,
        "hum_detected": getattr(rep, "hum_detected", False),
        "reverb_detected": getattr(rep, "reverb_detected", False),
    }


# ==============================================================================
# DSP VE AI İŞLEMCİ ORKESTRASYONU
# ==============================================================================
class DSPProcessor:
    """
    Modüler ve sıralanabilir ses restorasyon zincirini çalıştırır.
    İş parçacığı güvenlidir ve çok kanallı (stereo/mono) yapıyı korur.
    """

    @staticmethod
    def process(
        audio: np.ndarray,
        fs: float,
        config: ProcessingChainConfig,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> Tuple[np.ndarray, dict]:
        stats: Dict[str, Any] = {"stages_executed": []}
        out = audio.copy()
        orig_dtype = audio.dtype

        order = list(config.stage_order) if config.stage_order else list(DEFAULT_STAGE_ORDER)
        total_stages = len(order)

        for s_idx, stage in enumerate(order):
            stage_prog_start = s_idx / total_stages
            stage_prog_end = (s_idx + 1) / total_stages

            if stage == "declip":
                if config.declip.enabled:
                    out, rep_count = declip(out, config.declip)
                    stats["declip_repaired"] = rep_count
                    stats["stages_executed"].append("declip")

            elif stage == "hum_filter":
                if config.hum_filter.enabled:
                    out = apply_hum_filter(out, fs, config.hum_filter)
                    stats["hum_filtered"] = True
                    stats["stages_executed"].append("hum_filter")

            elif stage == "declick":
                if config.declick.enabled:
                    out, click_cnt = declick(out, fs, config.declick)
                    stats["clicks_repaired"] = click_cnt
                    stats["stages_executed"].append("declick")
                if config.plosive.enabled:
                    out = reduce_plosives(out, fs, config.plosive)
                    stats["plosives_reduced"] = True

            elif stage == "ai_denoise":
                if config.ai_denoise.enabled:
                    try:
                        from ai.restoration_engine import get_ai_engine
                        engine = get_ai_engine()
                        if engine.is_available():
                            def sub_cb(p: float) -> None:
                                if progress_cb:
                                    progress_cb(stage_prog_start + p * (stage_prog_end - stage_prog_start))

                            out = engine.restore_speech(
                                out,
                                fs,
                                model=config.ai_denoise.model,
                                strength=config.ai_denoise.strength,
                                preserve_high_frequencies=config.ai_denoise.preserve_high_frequencies,
                                blend=config.ai_denoise.blend,
                                progress_cb=sub_cb,
                            )
                            stats["ai_denoised"] = True
                            stats["stages_executed"].append("ai_denoise")
                        else:
                            logger.warning("AI modeli bulunamadı, fallback spektral denoise uygulanıyor.")
                            out = reduce_noise_spectral(out, fs, DenoiseSettings(enabled=True, amount_db=12.0))
                            stats["spectral_fallback"] = True
                            stats["stages_executed"].append("denoise")
                    except Exception as e:
                        logger.error(f"AI Denoise hatası: {e}, DSP fallback devreye girdi.")
                        out = reduce_noise_spectral(out, fs, DenoiseSettings(enabled=True, amount_db=12.0))
                        stats["spectral_fallback"] = True
                        stats["stages_executed"].append("denoise")

            elif stage == "denoise":
                if config.denoise.enabled:
                    out = reduce_noise_spectral(out, fs, config.denoise)
                    stats["denoised"] = True
                    stats["stages_executed"].append("denoise")

            elif stage == "dereverb":
                if config.dereverb.enabled:
                    out = dereverb_spectral(out, fs, config.dereverb)
                    stats["dereverbed"] = True
                    stats["stages_executed"].append("dereverb")

            elif stage == "clarity":
                if config.clarity.enabled:
                    out = apply_speech_clarity(out, fs, config.clarity)
                    stats["clarity_applied"] = True
                    stats["stages_executed"].append("clarity")

            elif stage == "eq":
                if config.eq_enabled:
                    bands = list(config.eq_bands)
                    if config.lowpass_enabled:
                        bands.append(
                            EQBand(band_type="lowpass", frequency=config.lowpass_freq, q=0.707, enabled=True)
                        )
                    if abs(config.vocal_presence_db) > 0.1:
                        bands.append(
                            EQBand(
                                band_type="peaking",
                                frequency=3400.0,
                                gain_db=config.vocal_presence_db,
                                q=1.1,
                                enabled=True,
                            )
                        )
                    out = apply_eq(out, fs, bands)
                    stats["eq_applied"] = True
                    stats["stages_executed"].append("eq")

            elif stage == "deesser":
                if config.deesser.enabled:
                    out, deess_red = apply_deesser(out, fs, config.deesser)
                    stats["deesser_reduction_db"] = deess_red
                    stats["stages_executed"].append("deesser")

            elif stage == "compressor":
                if config.compressor.enabled:
                    out, max_gr = apply_compressor(out, fs, config.compressor)
                    stats["max_gain_reduction_db"] = max_gr
                    stats["stages_executed"].append("compressor")

            elif stage == "gain_pan_fade":
                if (
                    abs(config.gain_db) > 1e-3
                    or abs(config.pan) > 1e-3
                    or config.fade_in_sec > 0.005
                    or config.fade_out_sec > 0.005
                ):
                    out = apply_gain_pan_fade(
                        out,
                        fs,
                        gain_db=config.gain_db,
                        pan=config.pan,
                        fade_in_sec=config.fade_in_sec,
                        fade_out_sec=config.fade_out_sec,
                    )
                    stats["gain_pan_fade_applied"] = True
                    stats["stages_executed"].append("gain_pan_fade")

            elif stage == "normalize":
                if config.normalize_lufs_target is not None:
                    out, lufs_gain = normalize_lufs(out, fs, config.normalize_lufs_target)
                    stats["lufs_normalization_gain_db"] = lufs_gain
                    stats["stages_executed"].append("normalize_lufs")
                elif config.normalize_target_dbfs is not None:
                    out, gain_db = normalize_peak(out, config.normalize_target_dbfs)
                    stats["normalization_gain_db"] = gain_db
                    stats["stages_executed"].append("normalize_peak")

            elif stage == "limiter":
                if config.limiter.enabled:
                    out = apply_limiter(out, fs, config.limiter)
                    stats["limited"] = True
                    stats["stages_executed"].append("limiter")

            if progress_cb:
                progress_cb(stage_prog_end)

        if config.trim_silence_enabled:
            out, s_trim, e_trim = trim_silence(out, fs, threshold_dbfs=config.trim_threshold_dbfs)
            stats["trimmed_samples"] = s_trim + e_trim
            stats["stages_executed"].append("trim_silence")

        return out.astype(orig_dtype), stats

    @classmethod
    def process_file(
        cls,
        input_path: str | Path,
        output_path: str | Path,
        config: ProcessingChainConfig,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> Tuple[Path, dict]:
        """Bir ses dosyasını okuyup işleyerek yeni bir WAV/FLAC/MP3 dosyasına yazar."""
        inp = Path(input_path)
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)

        audio, fs = sf.read(str(inp), dtype="float32")
        processed, stats = cls.process(audio, fs, config, progress_cb=progress_cb)

        # Uzantıya göre kaydet
        ext = outp.suffix.lower()
        if ext in {".wav", ".flac", ".ogg"}:
            subtype = "PCM_24" if ext == ".wav" else None
            sf.write(str(outp), processed, int(fs), subtype=subtype)
        elif ext == ".mp3":
            from audio.exporter import export_mp3_pyav
            export_mp3_pyav(processed, int(fs), outp, bitrate_kbps=320)
        else:
            sf.write(str(outp), processed, int(fs))

        return outp, stats
