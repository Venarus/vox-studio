# 🏛️ VOX Studio — Architecture & Technical Specifications

This document outlines the internal architecture, mathematical formulations, and engineering principles powering the **VOX Studio Audio Engine**.

---

## 1. System Overview

VOX Studio is engineered with a strict decoupling between real-time hardware I/O, digital signal processing (DSP), neural network inference (ONNX Runtime), and the user interface (PySide6 / Qt).

```
┌─────────────────────────────────────────────────────────────┐
│                    PySide6 / Qt UI Layer                    │
│   (Recording Studio, Restoration Lab, Timeline, A/B Player) │
└──────────────────────────────┬──────────────────────────────┘
                               │ Signals & Events
┌──────────────────────────────▼──────────────────────────────┐
│                    Application Context                      │
│             (Config, Audio Devices, Project State)          │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
┌──────────────▼──────────────┐ ┌──────────────▼──────────────┐
│   Hardware I/O & Drivers   │ │      Master DSP Chain        │
│  (WASAPI / MME Ring Buffer, │ │ (11-Stage Modular Pipeline, │
│   Atomic Journal Recovery)  │ │  Batch Processor Engine)    │
└─────────────────────────────┘ └──────────────┬──────────────┘
                                               │
                                ┌──────────────▼──────────────┐
                                │ Hybrid Neural / Air Core    │
                                │ (Linkwitz-Riley Crossover,  │
                                │  GTCRN ONNX Recurrent Net)  │
                                └─────────────────────────────┘
```

---

## 2. The 11-Stage Restoration Pipeline

### Stage 1: Audio Diagnosis Engine
Before applying any transformation, the raw signal $x[n]$ is evaluated using Welch's Power Spectral Density (PSD) method:
- **Mains Hum Detection:** Peak spectral analysis between 48–62 Hz with a sharpness factor $Q > 15$.
- **SNR Estimation:** Statistical ratio of active voice energy (detected via 30 ms energy windows) to the estimated minimum noise floor.
- **Clipping Crests:** Detection of consecutive sample blocks where $|x[n]| \ge 0.999$.

### Stage 2: De-Hum & Harmonic Notch Filtering
When electrical interference is identified at frequency $f_0$ (50 Hz or 60 Hz), multi-harmonic IIR notch biquad filters are applied:
$$H(z) = \frac{1 - 2\cos(\omega_k)z^{-1} + z^{-2}}{1 - 2r\cos(\omega_k)z^{-1} + r^2 z^{-2}}$$
where $\omega_k = 2\pi k f_0 / f_s$ for harmonics $k \in \{1, 2, 3, 4\}$ and $r = 1 - \frac{\pi B}{f_s}$.

### Stage 3: De-Clip via Cubic Spline Crest Repair
Flat-topped sample sequences exceeding the clipping threshold are identified. The destroyed crest is excised and reconstructed via piecewise cubic spline interpolation bounded by second-derivative continuity, restoring natural harmonic crest dynamics.

### Stage 4: De-Click & De-Pop (MAD Residual Spike Detection)
Transient mouth clicks and digital dropouts are identified using the Median Absolute Deviation (MAD) of the high-frequency residual signal:
$$\text{MAD} = \text{median}(|e[n] - \text{median}(e[n])|)$$
Points where $|e[n]| > \tau \cdot \text{MAD}$ are flagged as artifacts and reconstructed using Hermite polynomial curve fitting.

### Stage 5: Dynamic Plosive Attenuator
Transient bursts of air against the microphone diaphragm generate massive sub-120 Hz energy. A lookahead envelope detector attenuates low-frequency resonance strictly during plosive transients while leaving the speaker's vocal fundamental untouched during normal phonation.

---

## 3. The Hybrid Neural & Vocal Air Band Core

Most neural speech enhancement models (e.g. GTCRN, DeepFilterNet, RNNoise) operate natively at 16 kHz or 24 kHz. Directly downsampling a 48 kHz master recording to 16 kHz and upsampling the model output cuts off all frequencies above 8 kHz, producing an artificial, "underwater" telephone quality.

VOX Studio solves this with a **Linkwitz-Riley 4th-Order (LR4) Crossover Split**:

```
                         Input Audio (48 kHz Master)
                                      │
               ┌──────────────────────┴──────────────────────┐
               │                                             │
      Low-Pass Filter (7.6 kHz)                     High-Pass Filter (7.6 kHz)
               │                                             │
      Polyphase Decimate (16 kHz)                   High-Frequency Vocal Air
               │                                    (Uncompressed & Untouched)
        GTCRN ONNX Model                                     │
    (Recurrent STFT Inference)                               │
               │                                             │
      Polyphase Upsample (48 kHz)                            │
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      │
                         Phase-Aligned Recombination
                                      │
                              Restored Full Band
```

### Linkwitz-Riley 4th-Order (LR4) Characteristics:
1. **Flat Amplitude Summation:** $|H_{LP}(\omega) + H_{HP}(\omega)| = 1$ across the entire spectrum.
2. **Zero Phase Distortion:** Cascading two 2nd-order Butterworth filters yields identical phase shifts at the crossover point, eliminating phase cancellation artifacts.
3. **Preservation of "Air":** Breath sounds, harmonic brilliance, and natural presence above 7.6 kHz are kept 100% pristine.

### GTCRN Recurrent State Management:
The Glance-and-Gaze Targeted Convolutional Recurrent Network executes in streaming mode:
- **Frame Size:** 512 samples (32 ms), Hop Size: 256 samples (16 ms).
- **Recurrent Caches:** `conv_cache`, `tra_cache`, and `inter_cache` maintain acoustic memory across chunk boundaries, preventing buffer discontinuity clicks and preventing out-of-memory errors on long audio files.

---

## 4. Post-Processing & Mastering Stages

### Stage 7: Parametric Spectral De-Reverb
Acoustic room reflections are modeled as an exponentially decaying tail:
$$R(t) = R_0 \cdot 10^{-6 t / \text{RT}_{60}}$$
Late reverberant energy is estimated across short-time Fourier frames and subtracted spectrally with an over-subtraction factor and spectral floor ceiling.

### Stage 8: Speech Clarity Voicing Engine
A specialized 4-band voicing contour enhances intelligibility:
1. **80 Hz Low-Cut:** Eliminates room rumble and stand thumps.
2. **350 Hz Boxiness Notch:** Attenuates desk reflections and nasal buildup.
3. **3.5 kHz Consonant Presence Boost:** Enhances clarity and speech comprehension.
4. **12 kHz Air Shelf:** Adds broadcast polish and brilliance.

### Stage 9: De-Esser & Dynamic Compressor
- **De-Esser:** Sidechain band-pass detector (5.5–9.5 kHz) applies smooth gain reduction exclusively during harsh sibilant spikes ("s", "sh", "ch").
- **Compressor:** Soft-knee RMS dynamic compression controls dynamic range with program-dependent attack and release envelopes.

### Stage 10: ITU-R BS.1770-4 LUFS Normalization
Integrated loudness is measured using K-weighting filters with absolute and relative gating (-70 LKFS and -10 LU below ungated loudness). The gain delta is applied precisely to hit target delivery standards (-16 LUFS podcast, -20 LUFS ACX audiobook, -23 LUFS EBU R128).

### Stage 11: True Peak Lookahead Limiter
A 5 ms lookahead brickwall limiter prevents digital inter-sample clipping, ensuring true peak ceilings (e.g. -1.0 dBFS or -3.0 dBFS) are strictly respected.

---

## 5. Thread Safety & Crash Journaling

- **Double-Buffered WASAPI Capture:** Audio samples captured by the sound card are written into a lock-free circular ring buffer in the high-priority audio callback thread.
- **Worker Thread Decoupling:** Heavy disk writes and UI waveform peak extractions run in low-priority background threads to prevent buffer overruns or dropped samples.
- **Atomic Journal Recovery:** During live recording, audio is streamed directly to a temporary `.part` file. Upon normal completion, atomic OS renaming transforms it into the final master file. If power cuts or process interruptions occur, the startup recovery wizard detects pending `.part` segments and repairs the audio headers.
