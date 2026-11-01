# Security Policy

## Supported Versions

VOX Studio security patches and vulnerability fixes are actively maintained for the following versions:

| Version | Supported          | Security Status |
| :------ | :----------------- | :-------------- |
| `1.x.x` | :white_check_mark: | Active support & continuous validation |
| `< 1.0` | :x:                | Deprecated |

---

## 🔒 Privacy & Architecture Principles

VOX Studio was designed from the ground up for strict confidentiality and air-gapped security:

- **100% Local Inference & DSP Execution:** All neural networks (GTCRN, RNNoise) and digital signal processing algorithms execute entirely on the local machine (CPU / DirectML). No audio streams, voice samples, or transcripts are ever transmitted to external cloud APIs or remote servers.
- **Zero Telemetry:** The application contains zero analytics trackers, user profiling hooks, or background network calls.
- **Safe Audio Parsing:** Input audio files (WAV, FLAC, MP3, OGG) are processed through memory-safe bindings via `soundfile` (libsndfile) and `PyAV` (FFmpeg) with strict buffer size bounds and error traps.
- **Journal & Crash Resilience:** In-progress recordings are written atomically using `.part` journal recovery, protecting against partial data corruption or system power cuts without leaking data across sessions.

---

## Reporting a Vulnerability

If you discover a security vulnerability, buffer overflow risk, or privacy flaw within VOX Studio, please **do not open a public issue**. Instead, report it responsibly:

1. **Email:** Contact the maintainers directly at [143168102+Venarus@users.noreply.github.com](mailto:143168102+Venarus@users.noreply.github.com).
2. **Details:** Include a clear description of the vulnerability, steps or a sample script to reproduce it, and your environment details (OS version, Python version, audio driver).
3. **Response Timeline:**
   - **Initial acknowledgement:** within 48 hours.
   - **Triage and fix proposal:** within 7 days.
   - **Coordinated security release:** published with appropriate CVE attribution.

We greatly appreciate the security research community's efforts to keep VOX Studio safe and reliable for voice actors and audio professionals worldwide.
