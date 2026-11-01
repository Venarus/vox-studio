"""
VOX Studio — Çoklu Dosya Toplu Ses İşleme Motoru (Batch Audio Processor)

Sorumluluklar:
* Klasör veya dosya listesi bazında toplu ses restorasyonu ve normalizasyonu,
* Seçili DSP/AI profili (Audiobook, Broadcast, Voiceover, Balanced vb.) uygulama,
* Çok kanallı ve farklı formatlardaki (WAV, FLAC, MP3, OGG, AIFF) dosyaları destekleme,
* Dosya bazında hata izolasyonu (bir dosya bozuksa diğerlerini durdurmaz),
* İlerleme bildirimi (dosya bazında ve genel), iptal desteği ve ayrıntılı özet raporu.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from core.logger import get_logger
from dsp.chain import (
    DSPProcessor,
    ProcessingChainConfig,
    compute_detailed_metrics,
    get_profile,
    list_profiles,
)

logger = get_logger("dsp.batch")

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}


@dataclass
class BatchFileResult:
    """Tek bir dosyanın toplu işleme sonucu."""

    input_path: Path
    output_path: Optional[Path] = None
    duration_sec: float = 0.0
    before_lufs: Optional[float] = None
    after_lufs: Optional[float] = None
    snr_before_db: Optional[float] = None
    status: str = "PENDING"  # SUCCESS, FAILED, SKIPPED, CANCELLED
    error_message: Optional[str] = None
    processing_time_sec: float = 0.0
    rtf: float = 0.0


@dataclass
class BatchResult:
    """Tüm toplu işleme oturumunun özet raporu."""

    total_files: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    total_audio_duration_sec: float = 0.0
    total_processing_time_sec: float = 0.0
    average_rtf: float = 0.0
    file_results: List[BatchFileResult] = field(default_factory=list)

    def summary_text(self) -> str:
        speedup = f"{1.0 / self.average_rtf:.1f}x" if self.average_rtf > 0 else "N/A"
        return (
            f"Toplam Dosya: {self.total_files} | "
            f"Başarılı: {self.succeeded} | "
            f"Hatalı: {self.failed} | "
            f"İşlenen Süre: {self.total_audio_duration_sec:.1f}s | "
            f"Geçen Süre: {self.total_processing_time_sec:.2f}s | "
            f"Ortalama RTF: {self.average_rtf:.3f} ({speedup} gerçek zamanlı)"
        )


class BatchAudioProcessor:
    """VOX Studio endüstriyel toplu ses restorasyon ve dışa aktarma motoru."""

    def __init__(
        self,
        config: Optional[ProcessingChainConfig] = None,
        profile_name: str = "balanced",
    ) -> None:
        self.config = config or get_profile(profile_name)
        self.profile_name = profile_name

    @staticmethod
    def discover_audio_files(
        directory: str | Path,
        recursive: bool = True,
    ) -> List[Path]:
        """Belirtilen dizindeki desteklenen ses dosyalarını listeler."""
        p = Path(directory)
        if not p.exists() or not p.is_dir():
            return []

        pattern = "**/*" if recursive else "*"
        found = [
            f
            for f in p.glob(pattern)
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(found)

    def process_file_single(
        self,
        input_file: str | Path,
        output_file: str | Path,
        format_ext: str = "wav",
        bit_depth: int = 24,
        mp3_bitrate: int = 320,
        metadata: Optional[Dict[str, str]] = None,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> BatchFileResult:
        """Tek bir ses dosyasını zincirden geçirir ve dışa aktarır."""
        inp = Path(input_file)
        outp = Path(output_file)
        t_start = time.perf_counter()

        result = BatchFileResult(input_path=inp, output_path=outp)

        try:
            if not inp.exists():
                raise FileNotFoundError(f"Girdi dosyası bulunamadı: {inp}")

            # Dosyayı oku
            audio, fs = sf.read(str(inp), dtype="float32")
            dur = len(audio) / float(fs) if audio.ndim == 1 else len(audio) / float(fs)
            result.duration_sec = dur

            # Öncesi metrikleri
            try:
                before_m = compute_detailed_metrics(audio, fs)
                result.before_lufs = before_m.get("lufs")
                result.snr_before_db = before_m.get("snr_db")
            except Exception:
                pass

            # DSP / AI Zinciri
            processed, stats = DSPProcessor.process(
                audio,
                float(fs),
                self.config,
                progress_cb=progress_cb,
            )

            # Sonrası metrikleri
            try:
                after_m = compute_detailed_metrics(processed, fs)
                result.after_lufs = after_m.get("lufs")
            except Exception:
                pass

            outp.parent.mkdir(parents=True, exist_ok=True)
            fmt = format_ext.lower().replace(".", "")

            if fmt == "mp3":
                from audio.exporter import export_mp3_pyav
                export_mp3_pyav(
                    processed,
                    int(fs),
                    outp,
                    bitrate_kbps=mp3_bitrate,
                    metadata=metadata,
                )
            elif fmt == "flac":
                from audio.exporter import export_flac
                export_flac(
                    processed,
                    int(fs),
                    outp,
                    bit_depth=bit_depth,
                    metadata=metadata,
                )
            else:
                from audio.exporter import export_wav
                export_wav(
                    processed,
                    int(fs),
                    outp,
                    bit_depth=bit_depth,
                    metadata=metadata,
                )

            t_elapsed = time.perf_counter() - t_start
            result.processing_time_sec = t_elapsed
            result.rtf = t_elapsed / dur if dur > 0 else 0.0
            result.status = "SUCCESS"
            logger.info(
                "Dosya başarıyla işlendi: %s -> %s (RTF: %.3f)",
                inp.name,
                outp.name,
                result.rtf,
            )

        except Exception as exc:
            t_elapsed = time.perf_counter() - t_start
            result.processing_time_sec = t_elapsed
            result.status = "FAILED"
            result.error_message = str(exc)
            logger.exception("Dosya işlenirken hata oluştu: %s", inp.name)

        return result

    def process_batch(
        self,
        input_files: Sequence[str | Path],
        output_dir: str | Path,
        format_ext: str = "wav",
        bit_depth: int = 24,
        mp3_bitrate: int = 320,
        metadata: Optional[Dict[str, str]] = None,
        progress_cb: Optional[Callable[[int, int, float, str], None]] = None,
        stop_event: Optional[Event] = None,
    ) -> BatchResult:
        """
        Tüm dosyaları sıralı olarak işler.
        progress_cb: (current_file_idx, total_files, file_progress, filename)
        stop_event: İptal talep edilirse döngüyü güvenle durdurur.
        """
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        files = [Path(f) for f in input_files]
        total = len(files)
        res = BatchResult(total_files=total)

        t_batch_start = time.perf_counter()

        for idx, file_p in enumerate(files, 1):
            if stop_event and stop_event.is_set():
                logger.warning("Toplu işleme kullanıcı tarafından iptal edildi.")
                res.cancelled = total - (res.succeeded + res.failed)
                break

            target_name = file_p.stem + f"_restored.{format_ext.lstrip('.')}"
            target_path = out_root / target_name

            def _file_prog(p: float):
                if progress_cb:
                    progress_cb(idx, total, p, file_p.name)

            file_res = self.process_file_single(
                input_file=file_p,
                output_file=target_path,
                format_ext=format_ext,
                bit_depth=bit_depth,
                mp3_bitrate=mp3_bitrate,
                metadata=metadata,
                progress_cb=_file_prog,
            )

            res.file_results.append(file_res)
            if file_res.status == "SUCCESS":
                res.succeeded += 1
                res.total_audio_duration_sec += file_res.duration_sec
            else:
                res.failed += 1

            if progress_cb:
                progress_cb(idx, total, 1.0, file_p.name)

        res.total_processing_time_sec = time.perf_counter() - t_batch_start
        if res.total_audio_duration_sec > 0:
            res.average_rtf = res.total_processing_time_sec / res.total_audio_duration_sec

        logger.info("Toplu işleme tamamlandı: %s", res.summary_text())
        return res


def main(argv: Optional[List[str]] = None) -> int:
    """CLI toplu ses işleme giriş noktası."""
    parser = argparse.ArgumentParser(
        prog="python -m dsp.batch",
        description="VOX Studio — Endüstriyel Toplu Ses Restorasyon Aracı",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="İşlenecek ses dosyası veya klasör yolu",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Temizlenen dosyaların kaydedileceği klasör",
    )
    parser.add_argument(
        "--profile",
        "-p",
        default="balanced",
        choices=list_profiles(),
        help=f"Restorasyon profili (seçenekler: {', '.join(list_profiles())})",
    )
    parser.add_argument(
        "--format",
        "-f",
        default="wav",
        choices=["wav", "mp3", "flac"],
        help="Çıktı formatı (wav, mp3, flac)",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=320,
        choices=[128, 192, 256, 320],
        help="MP3 çıkışı için bitrate (kbps)",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=24,
        choices=[16, 24, 32],
        help="WAV/FLAC için bit derinliği",
    )

    args = parser.parse_args(argv)

    inp = Path(args.input)
    if inp.is_file():
        files = [inp]
    elif inp.is_dir():
        files = BatchAudioProcessor.discover_audio_files(inp)
    else:
        print(f"Hata: Girdi yolu mevcut değil: {inp}", file=sys.stderr)
        return 1

    if not files:
        print(f"Uyarı: '{inp}' içinde desteklenen ses dosyası bulunamadı.", file=sys.stderr)
        return 0

    print(f"VOX Studio Toplu İşlem Başlatılıyor: {len(files)} dosya | Profil: {args.profile}")
    processor = BatchAudioProcessor(profile_name=args.profile)

    def _prog(cur, total, prog, fname):
        pct = int(prog * 100)
        sys.stdout.write(f"\r[{cur}/{total}] {fname} (%{pct})    ")
        sys.stdout.flush()

    res = processor.process_batch(
        input_files=files,
        output_dir=args.output,
        format_ext=args.format,
        bit_depth=args.bit_depth,
        mp3_bitrate=args.bitrate,
        progress_cb=_prog,
    )

    print("\n" + "=" * 60)
    print("SONUÇ:", res.summary_text())
    print("=" * 60)
    return 0 if res.failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
