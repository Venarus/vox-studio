"""
VOX Studio — Otomatik Konuşma Tanıma (ASR), Çok Dilli Transkripsiyon,
Sessizlik Analizi ve Altyazı Doğrulama Motoru

Özellikler:
- faster-whisper kuruluysa gerçek yerel AI modeliyle (30 dil + Otomatik Algılama)
  ve kelime/cümle zaman damgalarıyla transkripsiyon yapar.
- faster-whisper kurulu değilse dürüst biçimde "Akustik VAD Segmentasyonu" modunu
  raporlar; asla sahte kelime zaman damgası veya sahte konuşmacı ayrımı uydurmaz.
- Uzun video/ses dosyalarını (örn. 30+ dk) bellek şişmesini (OOM) önlemek için
  parçalı (chunked) olarak işler, ilerleme bildirir ve her an iptal edilebilir.
- Orijinal transkript metnini (`original_text`) ve kullanıcının düzenlediği metni
  (`text`) ayrı saklar; bölme (`split`), birleştirme (`merge`), arama/değiştirme
  ve SRT / VTT / TXT / JSON dışa aktarma + geri okuma doğrulaması sağlar.
"""

from __future__ import annotations

import copy
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from ai.media_extractor import MediaExtractor
from core.i18n import get_language_spec, normalize_unicode_text
from core.logger import get_logger

logger = get_logger("ai.transcription")


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "confidence": round(float(self.confidence), 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WordTimestamp":
        return cls(
            word=str(data.get("word", "")),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class TranscriptSegment:
    id: int
    start: float  # saniye
    end: float    # saniye
    text: str
    speaker: str = "Konuşmacı 1"
    confidence: float = 1.0
    original_text: str = ""
    words: List[WordTimestamp] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text = normalize_unicode_text(self.text)
        if not self.original_text:
            self.original_text = self.text
        else:
            self.original_text = normalize_unicode_text(self.original_text)

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    @property
    def is_modified(self) -> bool:
        return self.text.strip() != self.original_text.strip()

    def reset_to_original(self) -> None:
        self.text = self.original_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "text": self.text,
            "original_text": self.original_text,
            "speaker": self.speaker,
            "confidence": round(float(self.confidence), 3),
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptSegment":
        words_raw = data.get("words", []) or []
        return cls(
            id=int(data.get("id", 1)),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")),
            original_text=str(data.get("original_text", data.get("text", ""))),
            speaker=str(data.get("speaker", "Konuşmacı 1")),
            confidence=float(data.get("confidence", 1.0)),
            words=[WordTimestamp.from_dict(w) for w in words_raw if isinstance(w, dict)],
        )


@dataclass
class SilenceRegion:
    start: float
    end: float
    duration: float
    mean_db: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "duration": round(float(self.duration), 3),
            "mean_db": round(float(self.mean_db), 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SilenceRegion":
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            duration=float(data.get("duration", 0.0)),
            mean_db=float(data.get("mean_db", -60.0)),
        )


@dataclass
class TranscriptionResult:
    segments: List[TranscriptSegment] = field(default_factory=list)
    silences: List[SilenceRegion] = field(default_factory=list)
    speech_intervals: List[Tuple[float, float]] = field(default_factory=list)
    language: str = "tr"
    duration_sec: float = 0.0
    used_whisper: bool = False
    has_word_timestamps: bool = False
    speaker_mode: str = "manual"  # 'acoustic_estimate' | 'manual'
    source_path: str = ""
    transcript_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    full_text: str = ""
    duration: float = 0.0

    def __post_init__(self):
        if self.duration and not self.duration_sec:
            self.duration_sec = self.duration
        elif self.duration_sec and not self.duration:
            self.duration = self.duration_sec
        if not self.full_text and self.segments:
            self.full_text = " ".join(s.text for s in self.segments)

    def reindex(self) -> None:
        self.segments.sort(key=lambda s: (s.start, s.end))
        for idx, seg in enumerate(self.segments, start=1):
            seg.id = idx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transcript_id": self.transcript_id,
            "source_path": self.source_path,
            "language": self.language,
            "duration_sec": round(float(self.duration_sec), 3),
            "used_whisper": bool(self.used_whisper),
            "has_word_timestamps": bool(self.has_word_timestamps),
            "speaker_mode": self.speaker_mode,
            "segments": [s.to_dict() for s in self.segments],
            "silences": [s.to_dict() for s in self.silences],
            "speech_intervals": [[round(float(a), 3), round(float(b), 3)] for a, b in self.speech_intervals],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionResult":
        res = cls(
            transcript_id=str(data.get("transcript_id", str(uuid.uuid4()))),
            source_path=str(data.get("source_path", "")),
            language=str(data.get("language", "tr")),
            duration_sec=float(data.get("duration_sec", 0.0)),
            used_whisper=bool(data.get("used_whisper", False)),
            has_word_timestamps=bool(data.get("has_word_timestamps", False)),
            speaker_mode=str(data.get("speaker_mode", "manual")),
            segments=[TranscriptSegment.from_dict(s) for s in data.get("segments", [])],
            silences=[SilenceRegion.from_dict(s) for s in data.get("silences", [])],
            speech_intervals=[(float(p[0]), float(p[1])) for p in data.get("speech_intervals", []) if len(p) >= 2],
        )
        return res


def split_transcript_segment(
    segments_or_seg: List[TranscriptSegment] | TranscriptSegment,
    segment_index: int = 0,
    split_time: Optional[float] = None,
) -> Tuple[TranscriptSegment, TranscriptSegment]:
    """
    Belirtilen indeksteki segmenti (veya doğrudan verilen TranscriptSegment nesnesini)
    iki ardışık segmente böler.
    Kelime zaman damgaları varsa split_time'a göre kelimeleri dağıtır;
    yoksa metni kelime sınırından ortadan ikiye ayırır.
    """
    if isinstance(segments_or_seg, TranscriptSegment):
        segments = [segments_or_seg]
        segment_index = 0
    else:
        segments = segments_or_seg
    if not (0 <= segment_index < len(segments)):
        raise IndexError("Bölünecek segment indeksi geçersiz.")

    target = segments[segment_index]
    dur = target.duration
    if dur < 0.16:
        raise ValueError("Segment bölünemeyecek kadar kısa (< 0.16 sn).")

    mid_t = split_time if (split_time is not None and target.start + 0.05 < split_time < target.end - 0.05) else (target.start + dur * 0.5)
    mid_t = round(float(mid_t), 3)

    if target.words:
        left_words = [copy.deepcopy(w) for w in target.words if (w.start + w.end) * 0.5 <= mid_t]
        right_words = [copy.deepcopy(w) for w in target.words if (w.start + w.end) * 0.5 > mid_t]
        if not left_words and right_words:
            left_words = [right_words.pop(0)]
        elif not right_words and left_words:
            right_words = [left_words.pop()]
        left_text = " ".join(w.word.strip() for w in left_words).strip()
        right_text = " ".join(w.word.strip() for w in right_words).strip()
    else:
        left_words, right_words = [], []
        tokens = target.text.strip().split()
        if len(tokens) >= 2:
            half = len(tokens) // 2
            left_text = " ".join(tokens[:half])
            right_text = " ".join(tokens[half:])
        else:
            chars = target.text.strip()
            half_c = max(1, len(chars) // 2)
            left_text = chars[:half_c].strip() or chars
            right_text = chars[half_c:].strip() or chars

    seg_left = TranscriptSegment(
        id=target.id,
        start=target.start,
        end=mid_t,
        text=left_text,
        original_text=left_text,
        speaker=target.speaker,
        confidence=target.confidence,
        words=left_words,
    )
    seg_right = TranscriptSegment(
        id=target.id + 1,
        start=mid_t,
        end=target.end,
        text=right_text,
        original_text=right_text,
        speaker=target.speaker,
        confidence=target.confidence,
        words=right_words,
    )

    segments[segment_index : segment_index + 1] = [seg_left, seg_right]
    for i, s in enumerate(segments, start=1):
        s.id = i
    return seg_left, seg_right


def merge_transcript_segments(
    segments_or_first: List[TranscriptSegment] | TranscriptSegment,
    first_index_or_second: int | TranscriptSegment = 0,
    second_index: Optional[int] = None,
) -> TranscriptSegment:
    """Ardışık iki segmenti tek bir segmentte birleştirir."""
    if isinstance(segments_or_first, TranscriptSegment) and isinstance(
        first_index_or_second, TranscriptSegment
    ):
        segments = [segments_or_first, first_index_or_second]
        idx_a, idx_b = 0, 1
    else:
        segments = segments_or_first  # type: ignore[assignment]
        first_index = int(first_index_or_second)  # type: ignore[arg-type]
        if second_index is None:
            second_index = first_index + 1
        idx_a, idx_b = min(first_index, second_index), max(first_index, second_index)
    if idx_a < 0 or idx_b >= len(segments) or idx_a == idx_b:
        raise IndexError("Birleştirilecek iki geçerli segment seçilmelidir.")

    seg_a = segments[idx_a]
    seg_b = segments[idx_b]

    merged_text = f"{seg_a.text.strip()} {seg_b.text.strip()}".strip()
    merged_orig = f"{seg_a.original_text.strip()} {seg_b.original_text.strip()}".strip()
    merged_words = copy.deepcopy(seg_a.words) + copy.deepcopy(seg_b.words)

    merged = TranscriptSegment(
        id=seg_a.id,
        start=min(seg_a.start, seg_b.start),
        end=max(seg_a.end, seg_b.end),
        text=merged_text,
        original_text=merged_orig,
        speaker=seg_a.speaker,
        confidence=round((seg_a.confidence + seg_b.confidence) * 0.5, 3),
        words=merged_words,
    )

    segments[idx_a] = merged
    del segments[idx_b]
    for i, s in enumerate(segments, start=1):
        s.id = i
    return merged


def _load_mono_audio_for_analysis(
    audio_or_path: np.ndarray | str | Path,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    if isinstance(audio_or_path, (str, Path)):
        p = Path(audio_or_path)
        try:
            data, sr = sf.read(str(p), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data, int(sr)
        except Exception:
            data, sr = MediaExtractor.extract_audio(p, target_sample_rate=16000, target_channels=1)
            return data.reshape(-1), int(sr)
    else:
        data = audio_or_path
        sr = sample_rate or 48000
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32, copy=False), int(sr)


def estimate_speaker_labels_acoustic(
    mono_audio_or_path: np.ndarray | str | Path,
    sr_or_segments: int | List[TranscriptSegment],
    segments: Optional[List[TranscriptSegment]] = None,
) -> None:
    """
    Harici konuşmacı ayrımı (diarization) modeli olmadığında, her segmentin
    spektral ağırlık merkezi (spectral centroid) ve RMS enerjisini ölçerek
    akustik konuşmacı değişim ipuçları ('Konuşmacı 1', 'Konuşmacı 2') üretir.
    """
    if segments is None and isinstance(sr_or_segments, list):
        segments = sr_or_segments
        mono_audio, sr = _load_mono_audio_for_analysis(mono_audio_or_path)
    else:
        mono_audio, sr = _load_mono_audio_for_analysis(mono_audio_or_path, int(sr_or_segments))  # type: ignore[arg-type]
    if not segments or len(segments) <= 1 or len(mono_audio) == 0:
        return

    centroids: List[float] = []
    for seg in segments:
        s_idx = max(0, int(seg.start * sr))
        e_idx = min(len(mono_audio), int(seg.end * sr))
        chunk = mono_audio[s_idx:e_idx]
        if len(chunk) < 256:
            centroids.append(0.0)
            continue
        spec = np.abs(np.fft.rfft(chunk[: min(len(chunk), sr * 4)]))
        freqs = np.fft.rfftfreq(min(len(chunk), sr * 4), d=1.0 / sr)
        denom = float(np.sum(spec))
        c_hz = float(np.sum(freqs * spec) / denom) if denom > 1e-9 else 0.0
        centroids.append(c_hz)

    valid_c = [c for c in centroids if c > 50.0]
    if len(valid_c) < 2:
        return

    median_c = float(np.median(valid_c))
    spread = float(np.std(valid_c))
    # Yalnızca belirgin tını/frekans farkı varsa 2 konuşmacıya ayır
    if spread < 120.0:
        for seg in segments:
            seg.speaker = "Konuşmacı 1"
        return

    for seg, c_val in zip(segments, centroids):
        if c_val <= 50.0:
            seg.speaker = "Konuşmacı 1"
        elif c_val < median_c:
            seg.speaker = "Konuşmacı 1"
        else:
            seg.speaker = "Konuşmacı 2"


def detect_speech_and_silence(
    audio_or_path: np.ndarray | str | Path,
    sample_rate: Optional[int] = None,
    silence_threshold_db: float = -40.0,
    min_silence_sec: float = 0.35,
    min_speech_sec: float = 0.25,
) -> Tuple[List[Tuple[float, float]], List[SilenceRegion], float]:
    """
    Enerji tabanlı konuşma/sessizlik algılayıcı (VAD).
    Dönüş: (konuşma_aralıkları, sessizlik_bölgeleri, toplam_süre_sn)
    """
    data, sr = _load_mono_audio_for_analysis(audio_or_path, sample_rate)
    total_dur = len(data) / float(sr) if sr > 0 else 0.0
    if len(data) == 0:
        return [], [], 0.0

    hop_sec = 0.02  # 20 ms
    hop = max(1, int(sr * hop_sec))
    n_frames = len(data) // hop
    if n_frames == 0:
        return [(0.0, total_dur)], [], total_dur

    trimmed = data[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms)

    is_speech = db > silence_threshold_db

    speech_intervals: List[Tuple[float, float]] = []
    silences: List[SilenceRegion] = []

    in_speech = bool(is_speech[0])
    seg_start = 0
    for idx in range(1, n_frames):
        flag = bool(is_speech[idx])
        if flag != in_speech:
            t_start = seg_start * hop_sec
            t_end = idx * hop_sec
            dur = t_end - t_start
            if in_speech:
                if dur >= min_speech_sec:
                    speech_intervals.append((t_start, t_end))
            else:
                if dur >= min_silence_sec:
                    mean_db = float(np.mean(db[seg_start:idx]))
                    silences.append(
                        SilenceRegion(
                            start=round(t_start, 3),
                            end=round(t_end, 3),
                            duration=round(dur, 3),
                            mean_db=round(mean_db, 1),
                        )
                    )
            in_speech = flag
            seg_start = idx

    t_start = seg_start * hop_sec
    t_end = total_dur
    dur = t_end - t_start
    if in_speech and dur >= min_speech_sec:
        speech_intervals.append((round(t_start, 3), round(t_end, 3)))
    elif not in_speech and dur >= min_silence_sec:
        mean_db = float(np.mean(db[seg_start:])) if seg_start < n_frames else -60.0
        silences.append(
            SilenceRegion(
                start=round(t_start, 3),
                end=round(t_end, 3),
                duration=round(dur, 3),
                mean_db=round(mean_db, 1),
            )
        )

    return speech_intervals, silences, total_dur


class WhisperTranscriber:
    """
    faster-whisper mevcutsa yerel AI transkripsiyonu çalıştırır (30 dil + Otomatik algılama +
    kelime zaman damgaları + uzun dosyalarda parçalı/chunked işleme).
    Mevcut değilse VAD konuşma aralıklarından zaman kodlu replik taslakları üretir.
    """

    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self._model = None

    @staticmethod
    def is_whisper_available() -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = "tr",
        silence_threshold_db: float = -40.0,
        chunk_duration_sec: float = 300.0,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> TranscriptionResult:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Transkripsiyon için medya dosyası bulunamadı: {path}")

        if progress_cb:
            progress_cb(5, f"Ses akışı ve konuşma bölgeleri analiz ediliyor: {path.name}")

        mono_audio, sr = _load_mono_audio_for_analysis(path)
        speech_intervals, silences, total_dur = detect_speech_and_silence(
            mono_audio,
            sample_rate=sr,
            silence_threshold_db=silence_threshold_db,
        )

        if cancel_check and cancel_check():
            raise RuntimeError("Transkripsiyon işlemi kullanıcı tarafından iptal edildi.")

        # Dil kodunu çözümle ('auto' veya None -> otomatik algılama)
        whisper_lang: Optional[str] = None
        resolved_lang_code = "tr"
        if language and str(language).strip().lower() not in ("auto", "none", ""):
            spec = get_language_spec(str(language))
            whisper_lang = spec.whisper_code
            resolved_lang_code = spec.code

        segments: List[TranscriptSegment] = []
        used_whisper = False
        has_word_ts = False

        if self.is_whisper_available():
            try:
                from faster_whisper import WhisperModel

                if self._model is None:
                    if progress_cb:
                        progress_cb(15, f"Yerel Whisper ({self.model_size}) modeli yükleniyor...")
                    try:
                        self._model = WhisperModel(
                            self.model_size,
                            device=self.device,
                            compute_type="float16" if self.device == "cuda" else "int8",
                        )
                    except Exception as dev_err:
                        logger.warning(f"Whisper {self.device} başlatılamadı ({dev_err}), CPU moduna geçiliyor...")
                        self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

                # Uzun dosyaları chunk_duration_sec dilimlerine bölerek işle (OOM koruması)
                chunk_sec = max(60.0, float(chunk_duration_sec))
                n_chunks = max(1, int(math.ceil(max(0.1, total_dur) / chunk_sec)))
                seg_counter = 1

                for c_idx in range(n_chunks):
                    if cancel_check and cancel_check():
                        raise RuntimeError("Transkripsiyon işlemi kullanıcı tarafından iptal edildi.")

                    c_start_sec = c_idx * chunk_sec
                    c_end_sec = min(total_dur, (c_idx + 1) * chunk_sec)
                    if c_end_sec <= c_start_sec:
                        continue

                    if progress_cb:
                        pct = 20 + int((c_idx / float(n_chunks)) * 65)
                        progress_cb(
                            pct,
                            f"Konuşma metne dönüştürülüyor ({c_idx + 1}/{n_chunks}. bölüm: "
                            f"{c_start_sec:.0f}s - {c_end_sec:.0f}s)...",
                        )

                    s_sample = int(c_start_sec * sr)
                    e_sample = int(c_end_sec * sr)
                    chunk_arr = mono_audio[s_sample:e_sample]

                    # faster-whisper 16kHz float32 veya geçici dosya kabul eder; örnekleme hızı farklıysa yeniden örnekle
                    if sr != 16000 and len(chunk_arr) > 0:
                        target_len = max(1, int(len(chunk_arr) * 16000.0 / sr))
                        chunk_16k = np.interp(
                            np.linspace(0, len(chunk_arr) - 1, target_len),
                            np.arange(len(chunk_arr)),
                            chunk_arr,
                        ).astype(np.float32)
                    else:
                        chunk_16k = chunk_arr.astype(np.float32)

                    try:
                        raw_segments, info = self._model.transcribe(
                            chunk_16k,
                            language=whisper_lang,
                            beam_size=5,
                            vad_filter=True,
                            word_timestamps=True,
                        )
                    except Exception as tr_err:
                        if cancel_check and cancel_check():
                            raise RuntimeError("Transkripsiyon işlemi kullanıcı tarafından iptal edildi.")
                        logger.warning(f"Whisper çıkarımında GPU/bellek hatası ({tr_err}), CPU moduna geçiliyor...")
                        self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                        raw_segments, info = self._model.transcribe(
                            chunk_16k,
                            language=whisper_lang,
                            beam_size=3,
                            vad_filter=True,
                            word_timestamps=True,
                        )
                    if c_idx == 0 and getattr(info, "language", None) and whisper_lang is None:
                        resolved_lang_code = get_language_spec(info.language).code

                    for rseg in raw_segments:
                        if cancel_check and cancel_check():
                            raise RuntimeError("Transkripsiyon işlemi kullanıcı tarafından iptal edildi.")
                        txt = normalize_unicode_text(rseg.text.strip())
                        if not txt:
                            continue
                        words_list: List[WordTimestamp] = []
                        if getattr(rseg, "words", None):
                            has_word_ts = True
                            for w in rseg.words:
                                words_list.append(
                                    WordTimestamp(
                                        word=normalize_unicode_text(w.word),
                                        start=round(c_start_sec + float(w.start), 3),
                                        end=round(c_start_sec + float(w.end), 3),
                                        confidence=round(float(getattr(w, "probability", 1.0)), 3),
                                    )
                                )
                        segments.append(
                            TranscriptSegment(
                                id=seg_counter,
                                start=round(c_start_sec + float(rseg.start), 3),
                                end=round(c_start_sec + float(rseg.end), 3),
                                text=txt,
                                original_text=txt,
                                confidence=round(float(getattr(rseg, "avg_logprob", 0.0)), 2),
                                words=words_list,
                            )
                        )
                        seg_counter += 1
                used_whisper = True
                logger.info(f"faster-whisper ile {len(segments)} replik çıkarıldı ({resolved_lang_code}).")
            except Exception as e:
                if cancel_check and cancel_check():
                    raise RuntimeError("Transkripsiyon işlemi kullanıcı tarafından iptal edildi.")
                if "iptal" in str(e).lower():
                    raise
                logger.warning(f"faster-whisper çalıştırılamadı, akustik VAD moduna geçiliyor: {e}")

        if not used_whisper:
            if progress_cb:
                progress_cb(60, "Akustik VAD konuşma aralıklarından zaman kodlu segmentler oluşturuluyor...")
            for idx, (s_start, s_end) in enumerate(speech_intervals, start=1):
                if cancel_check and cancel_check():
                    raise RuntimeError("Transkripsiyon işlemi kullanıcı tarafından iptal edildi.")
                placeholder = f"[Konuşma Bölgesi #{idx} — {s_start:.2f}s .. {s_end:.2f}s]"
                segments.append(
                    TranscriptSegment(
                        id=idx,
                        start=round(s_start, 3),
                        end=round(s_end, 3),
                        text=placeholder,
                        original_text=placeholder,
                        confidence=0.85,
                        words=[],  # VAD modunda sahte kelime zaman damgası üretilmez
                    )
                )

        # Çok düşük seviyeli veya sessiz ses kontrolü
        rms_val = float(np.sqrt(np.mean(mono_audio ** 2) + 1e-12)) if len(mono_audio) > 0 else 0.0
        db_overall = float(20.0 * np.log10(rms_val))
        if len(segments) == 0 and db_overall < -50.0:
            logger.info(f"Medya dosyası çok sessiz ({db_overall:.1f} dBFS). Bilgilendirici segment eklendi.")
            segments.append(
                TranscriptSegment(
                    id=1,
                    start=0.0,
                    end=max(1.0, round(total_dur, 2)),
                    text=f"[Sessiz Medya — Konuşma Seviyesi Çok Düşük ({db_overall:.1f} dBFS)]",
                    original_text=f"[Sessiz Medya — Konuşma Seviyesi Çok Düşük ({db_overall:.1f} dBFS)]",
                    confidence=0.5,
                    words=[],
                )
            )

        # Akustik konuşmacı değişim tahmini
        estimate_speaker_labels_acoustic(mono_audio, sr, segments)

        if progress_cb:
            progress_cb(100, "Transkripsiyon tamamlandı.")

        return TranscriptionResult(
            segments=segments,
            silences=silences,
            speech_intervals=speech_intervals,
            language=resolved_lang_code,
            duration_sec=total_dur,
            used_whisper=used_whisper,
            has_word_timestamps=has_word_ts,
            speaker_mode="acoustic_estimate",
            source_path=str(path),
        )

    # ------------------------------------------------------------ SRT & VTT & TXT & JSON Export
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        if millis >= 1000:
            millis = 999
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        return WhisperTranscriber._format_srt_time(seconds).replace(",", ".")

    @classmethod
    def export_srt(
        cls,
        segments_or_result: List[TranscriptSegment] | TranscriptionResult,
        output_path: str | Path,
        include_speaker: bool = False,
    ) -> Path:
        segments = (
            segments_or_result.segments
            if isinstance(segments_or_result, TranscriptionResult)
            else segments_or_result
        )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        for idx, seg in enumerate(segments, start=1):
            t_in = cls._format_srt_time(seg.start)
            t_out = cls._format_srt_time(max(seg.start + 0.05, seg.end))
            text = normalize_unicode_text(seg.text.strip())
            if include_speaker and seg.speaker:
                text = f"[{seg.speaker}] {text}"
            lines.append(f"{idx}\n{t_in} --> {t_out}\n{text}\n")
        out.write_text("\n".join(lines), encoding="utf-8-sig")
        return out

    @classmethod
    def export_vtt(
        cls,
        segments_or_result: List[TranscriptSegment] | TranscriptionResult,
        output_path: str | Path,
        include_speaker: bool = False,
    ) -> Path:
        segments = (
            segments_or_result.segments
            if isinstance(segments_or_result, TranscriptionResult)
            else segments_or_result
        )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = ["WEBVTT\n"]
        for idx, seg in enumerate(segments, start=1):
            t_in = cls._format_vtt_time(seg.start)
            t_out = cls._format_vtt_time(max(seg.start + 0.05, seg.end))
            text = normalize_unicode_text(seg.text.strip())
            if include_speaker and seg.speaker:
                text = f"<v {seg.speaker}>{text}"
            lines.append(f"{idx}\n{t_in} --> {t_out}\n{text}\n")
        out.write_text("\n".join(lines), encoding="utf-8-sig")
        return out

    @classmethod
    def export_txt(
        cls,
        segments_or_result: List[TranscriptSegment] | TranscriptionResult,
        output_path: str | Path,
        include_timestamps: bool = True,
        include_speaker: bool = True,
    ) -> Path:
        segments = (
            segments_or_result.segments
            if isinstance(segments_or_result, TranscriptionResult)
            else segments_or_result
        )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        for seg in segments:
            parts: List[str] = []
            if include_timestamps:
                parts.append(f"[{seg.start:06.2f}s - {seg.end:06.2f}s]")
            if include_speaker and seg.speaker:
                parts.append(f"{seg.speaker}:")
            parts.append(normalize_unicode_text(seg.text.strip()))
            lines.append(" ".join(parts))
        out.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        return out

    @classmethod
    def export_json(
        cls,
        result: TranscriptionResult | List[TranscriptSegment],
        output_path: str | Path,
        language: str = "auto",
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(result, list):
            full_text = " ".join(s.text for s in result)
            duration = max((s.end for s in result), default=0.0)
            data = {
                "language": language,
                "duration": duration,
                "full_text": full_text,
                "segments": [s.to_dict() if hasattr(s, "to_dict") else dict(s) for s in result],
            }
        else:
            data = result.to_dict()
        out.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out


_SRT_TIMECODE_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$"
)
_VTT_TIMECODE_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}$"
)


def validate_exported_subtitle_file(
    path: str | Path,
    fmt: Optional[str] = None,
    expected_min_segments: int = 0,
) -> Tuple[bool, str]:
    """
    Dışa aktarılan SRT, VTT, TXT veya JSON dosyasını UTF-8 olarak geri okur,
    boş olmadığını, zaman kodlarının standartlara uygunluğunu ve Unicode bütünlüğünü doğrular.
    """
    p = Path(path)
    if not p.exists():
        return False, f"Dışa aktarılan dosya bulunamadı: {p}"
    if p.stat().st_size == 0:
        return False, f"Dışa aktarılan dosya boş (0 bayt): {p}"

    try:
        content = p.read_text(encoding="utf-8-sig")
    except Exception as exc:
        return False, f"Dosya UTF-8 olarak okunamadı: {exc}"

    if not content.strip():
        return False, "Dosya içeriği yalnızca boşluktan oluşuyor."

    ext = (fmt or p.suffix.lstrip(".")).lower()
    if ext == "srt":
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        tc_lines = [ln for ln in lines if "-->" in ln]
        if not tc_lines:
            return False, "SRT dosyasında geçerli '-->' zaman kodu satırı bulunamadı."
        if expected_min_segments > 0 and len(tc_lines) < expected_min_segments:
            return False, f"SRT beklenen minimum segment sayısını ({expected_min_segments}) karşılamıyor: {len(tc_lines)}"
        for tc in tc_lines:
            if not _SRT_TIMECODE_RE.match(tc):
                return False, f"Geçersiz SRT zaman kodu biçimi: '{tc}'"
        return True, f"SRT doğrulandı ({len(tc_lines)} altyazı bloğu, UTF-8)."

    if ext == "vtt":
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if not lines or not lines[0].startswith("WEBVTT"):
            return False, "VTT dosyası 'WEBVTT' başlığı ile başlamıyor."
        tc_lines = [ln for ln in lines if "-->" in ln]
        if not tc_lines:
            return False, "VTT dosyasında geçerli '-->' zaman kodu satırı bulunamadı."
        if expected_min_segments > 0 and len(tc_lines) < expected_min_segments:
            return False, f"VTT beklenen minimum segment sayısını ({expected_min_segments}) karşılamıyor: {len(tc_lines)}"
        for tc in tc_lines:
            if not _VTT_TIMECODE_RE.match(tc):
                return False, f"Geçersiz VTT zaman kodu biçimi: '{tc}'"
        return True, f"WEBVTT doğrulandı ({len(tc_lines)} altyazı bloğu, UTF-8)."

    if ext == "json":
        try:
            obj = json.loads(content)
            if not isinstance(obj, dict) or "segments" not in obj:
                return False, "JSON transkript şemasında 'segments' alanı eksik."
            if expected_min_segments > 0 and len(obj["segments"]) < expected_min_segments:
                return False, f"JSON beklenen segment sayısını ({expected_min_segments}) karşılamıyor: {len(obj['segments'])}"
            return True, f"JSON transkript doğrulandı ({len(obj['segments'])} segment)."
        except Exception as exc:
            return False, f"Geçersiz JSON biçimi: {exc}"

    # TXT
    non_empty = [ln for ln in content.splitlines() if ln.strip()]
    if expected_min_segments > 0 and len(non_empty) < expected_min_segments:
        return False, f"TXT beklenen satır sayısını ({expected_min_segments}) karşılamıyor: {len(non_empty)}"
    return True, f"Metin (TXT) doğrulandı ({len(non_empty)} satır, UTF-8)."
