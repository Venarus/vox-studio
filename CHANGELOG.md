# Changelog

All notable changes to **VOX Studio** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-30

### Initial Release — Audio Engine V1 & Ultimate AI Speech Restoration Engine

#### 🎙️ Professional Audio Recording Studio
- **WASAPI/MME Real Device Detection:** Native Windows audio device enumeration with device persistence and automatic fallback.
- **Reference Master Quality:** 48 kHz / 24-bit PCM WAV capture via high-performance double-buffered ring buffers.
- **Visual Audio Metering:** Real-time peak VU meter and clipping detection indicators.
- **Crash Recovery:** Automatic crash protection via `.part` incremental writes and automatic recovery on startup.

#### 🧠 11-Stage AI & DSP Speech Restoration Engine
- **Intelligent Audio Diagnosis:** Automated Welch PSD hum detection (50/60 Hz), SNR evaluation, clipping count, and human-readable diagnostic recommendations.
- **GTCRN Neural Speech Enhancement:** Offline ONNX inference utilizing Glance-and-Gaze Targeted CRN with a 11.7x faster than real-time execution factor (RTF 0.086).
- **Hybrid Linkwitz-Riley Crossover (7.6 kHz):** Blends 16 kHz neural denoising with unprocessed 48 kHz high-frequency spectrum, preserving vocal breath, brilliance, and air without robotic artifacts.
- **De-Hum Filter:** Precise 50/60 Hz notch filtering with configurable harmonics and high-Q factor.
- **De-Clip Algorithm:** Reconstructs flat-topped clipping distortion using cubic spline interpolation.
- **De-Click & De-Pop:** Median Absolute Deviation (MAD) residual spike detection with Hermite reconstruction for mouth clicks and digital dropouts.
- **De-Plosive:** Dynamic 120 Hz high-pass attenuation triggered exclusively on wind bursts.
- **Parametric Spectral De-Reverb:** Removes room reverberation and early reflections via RT60 decay tail subtraction.
- **Speech Clarity Engine:** 4-band voicing contour (80 Hz rumble roll-off, 350 Hz boxiness notch, 3.5 kHz consonant presence boost, 12 kHz air shelf).
- **Vocal Compressor & De-Esser:** Dynamic range control with 5.5–9.5 kHz sibilance suppression.
- **ITU-R BS.1770-4 Normalizer & Limiter:** Gated integrated loudness normalization target (-16 LUFS) with lookahead ceiling protection.

#### 🎚️ Synchronized A/B Player
- **Zero-Latency Channel Switching:** Instantaneous seamless toggle between Original (A) and Restored (B) at matching playhead positions.
- **Perceptual Loudness Matching:** Automatic LUFS/RMS gain compensation eliminating loudness bias during comparison.

#### 📦 Master Audio Exporter
- **24-bit / 16-bit PCM WAV:** Studio broadcast master standard.
- **PyAV libmp3lame MP3:** High-fidelity CBR up to 320 kbps.
- **Lossless FLAC:** Broadcast compression with metadata verification.

#### 🧪 Verification & Benchmarks
- 88 unit tests passing across all engines (100% test coverage for core audio paths).
- 15-test end-to-end benchmark suite passing on Windows hardware.
