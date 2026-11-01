from .analysis import AudioAnalysis, analyze_array, analyze_file
from .filters import EQBand, apply_eq, make_biquad
from .dynamics import CompressorSettings, LimiterSettings, apply_compressor, apply_limiter
from .denoise import DenoiseSettings, reduce_noise_spectral
from .normalize import normalize_peak, trim_silence
from .chain import (
    DSPProcessor,
    PROFILES,
    ProcessingChainConfig,
    create_adaptive_profile,
    create_audiobook_profile,
    create_balanced_profile,
    create_broadcast_profile,
    create_maximum_restoration_profile,
    create_natural_profile,
    create_voiceover_profile,
    get_profile,
    list_profiles,
)

__all__ = [
    "AudioAnalysis",
    "analyze_array",
    "analyze_file",
    "EQBand",
    "apply_eq",
    "make_biquad",
    "CompressorSettings",
    "LimiterSettings",
    "apply_compressor",
    "apply_limiter",
    "DenoiseSettings",
    "reduce_noise_spectral",
    "normalize_peak",
    "trim_silence",
    "ProcessingChainConfig",
    "DSPProcessor",
    "create_natural_profile",
    "create_balanced_profile",
    "create_maximum_restoration_profile",
    "create_audiobook_profile",
    "create_broadcast_profile",
    "create_voiceover_profile",
    "create_adaptive_profile",
    "get_profile",
    "list_profiles",
    "PROFILES",
]
