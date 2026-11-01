# 📖 VOX Studio — User & Workflow Guide

Welcome to **VOX Studio**, a broadcast-grade voice recording workstation and AI speech restoration engine designed for voice actors, narrators, podcasters, and sound designers.

---

## 🎙️ 1. Studio Recording Setup

### Hardware & Driver Selection
1. Launch VOX Studio.
2. In the top toolbar, verify that your microphone is detected in the **Giriş Aygıtı (Input Device)** selector.
3. Windows WASAPI is the recommended host API for lowest latency and bit-perfect hardware access.
4. Verify your levels using the **Peak / RMS VU Meter**:
   - Optimal voice peaks: **-18 dBFS to -12 dBFS**.
   - If the red **CLIP** indicator illuminates, lower your microphone pre-amp gain to prevent analogue-to-digital clipping.

### Recording Takes
- Press **Ctrl+R** or click the large circular **Kayıt (Record)** button to begin recording.
- Press **Esc** or click **Durdur (Stop)** to finish.
- Your take is automatically indexed in the take list and preserved with zero-loss crash protection.

---

## 🧠 2. Speech Restoration & Quality Profiles

When opening a recording or importing an external audio file, navigate to the **Restorasyon (Processing)** view.

### Choosing the Ideal Profile

| Profile | Target LUFS | Peak Ceiling | Best Suited For |
| :--- | :--- | :--- | :--- |
| **Audiobook (ACX)** | `-20.0 LUFS` | `-3.0 dBFS` | Audible / ACX submissions, audiobooks, spoken narration. High-sensitivity saliva and mouth de-clicker. |
| **Broadcast (EBU)** | `-23.0 LUFS` | `-1.0 dBFS` | TV broadcast (EBU R128), European radio, documentary narration. 50 Hz notch filter included. |
| **Voiceover & Trailer**| `-16.0 LUFS`| `-1.0 dBFS` | Commercial advertisements, YouTube videos, podcast intros. High-end air (+2.2 dB @ 12 kHz). |
| **Balanced (Recommended)**| `-16.0 LUFS`| `-1.0 dBFS` | General dialogue, online streaming, video tutorials. Cleans fan and AC noise transparently. |
| **Natural** | `-19.0 LUFS` | `-1.0 dBFS` | Acoustic environments where natural breaths and room timbre must be completely preserved. |
| **Maximum** | `-16.0 LUFS` | `-1.0 dBFS` | Heavy background traffic, loud restaurant rumble, or severely degraded microphone signals. |

### Simple Mode vs. Pro Mode
- **Simple Mode:** Select a profile and adjust the single **AI Gürültü Azaltma Gücü (Denoise Strength)** slider (10% to 100%).
- **Pro Mode:** Click **Pro Modu** to access individual tabbed racks for **De-Clip**, **De-Hum (50/60 Hz)**, **De-Click**, **De-Reverb**, **Speech Clarity (4-band voicing)**, **De-Esser**, and **Brickwall Limiter**.

---

## 🎚️ 3. Synchronized A/B Comparison

When verifying audio restoration, human ears naturally perceive louder audio as "better" (volume bias). VOX Studio eliminates this flaw:

1. Click **A/B Karşılaştırma** in the transport bar.
2. Channel **A** plays the original, unprocessed microphone take.
3. Channel **B** plays the restored audio with zero latency at the exact same sample position.
4. Enable **Loudness Eşitleme (Loudness Matching)**: The engine calculates the real-time LUFS delta and applies gain compensation to eliminate volume differences, allowing you to evaluate signal purity strictly on noise and distortion reduction.

---

## 📦 4. Batch Processing Sessions

For multi-chapter audiobooks or episodic podcast takes, use the batch processing engine:

### Via Command-Line Interface (CLI):
```powershell
# Batch process a whole folder of WAV chapters for Audible/ACX:
python -m dsp.batch --input "C:\Recordings\ChapterTakes" --output "C:\Recordings\ACX_Master" --profile audiobook --format wav

# Batch process into high-definition 320k MP3s with commercial voiceover profile:
python -m dsp.batch --input "C:\Podcasts" --output "C:\Podcasts\MP3_Release" --profile voiceover --format mp3 --bitrate 320
```

### Via Application Entrypoint:
```powershell
python main.py --batch "C:\Recordings\MyTake.wav" --profile broadcast --batch-format flac
```

---

## 💾 5. Master Audio Export

Click **Dışa Aktar (Export)** to produce studio masters:
- **WAV (24-bit PCM Master):** 48 kHz / 24-bit uncompressed master standard.
- **MP3 (320 kbps CBR):** Encoded via PyAV `libmp3lame` with embedded studio metadata tags.
- **FLAC (24-bit Lossless):** Compressed broadcast archival format with bit-perfect fidelity.
