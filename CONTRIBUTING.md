# Contributing to VOX Studio

We welcome contributions to **VOX Studio — Audio Engine**! Whether reporting bugs, proposing new digital signal processing (DSP) filters, or optimizing neural network inference pipelines, your support is valued.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Venarus/vox-studio.git
   cd vox-studio
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows PowerShell
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the test suite:**
   ```bash
   pytest -v
   ```

5. **Run the end-to-end benchmark:**
   ```bash
   python tools/e2e_audio_engine_benchmark.py
   ```

---

## 📐 Code Style & Architecture Guidelines

- **Architecture Boundary:** Core DSP and neural inference modules (`dsp/`, `ai/`) must remain strictly decoupled from UI frameworks (PySide6 / Qt).
- **Type Annotations:** All new functions and public methods must include Python 3.10+ type annotations (`from __future__ import annotations`).
- **Memory Safety:** When dealing with audio buffers in PyAV or NumPy, ensure arrays are contiguous (`np.ascontiguousarray`) and explicitly manage audio thread locks (`threading.RLock`).
- **Master Audio Standard:** All recording pipelines default to 48 kHz / 24-bit PCM WAV. High-frequency preservation (>8 kHz air) must be verified when updating downsampled neural models.

---

## 🧪 Testing Requirements

Before submitting a Pull Request:
- Ensure all 96+ unit tests pass: `pytest`
- Run the batch processor verification and E2E benchmark to confirm real-time factor (RTF < 0.1) and zero clipping:
  ```bash
  python tools/e2e_audio_engine_benchmark.py
  ```
- Keep commit messages descriptive using Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `perf:`, `test:`, `chore:`).

---

## 📄 License

By contributing to VOX Studio, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
