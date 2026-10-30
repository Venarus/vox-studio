"""
VOX Studio — Medya Motoru, Kodek Doğrulayıcı, Altyazı Çözücü ve Ses Çıkarıcı (PyAV / FFmpeg + PIL)
Video, ses, görsel ve altyazı dosyalarını uzantıya körü körüne güvenmeden gerçek akış
ve kodek düzeyinde inceler; ses akışlarını ve küçük resimleri (thumbnail) güvenle çıkarır.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import numpy as np
from PIL import Image
import soundfile as sf

from core import constants as C
from core import paths
from core.logger import get_logger

logger = get_logger("ai.media_extractor")


@dataclass
class AudioStreamInfo:
    index: int
    codec_name: str
    sample_rate: int
    channels: int
    duration_sec: float
    bit_rate: Optional[int] = None
    language: Optional[str] = None


@dataclass
class SubtitleCue:
    index: int
    start_sec: float
    end_sec: float
    text: str


@dataclass
class MediaInfo:
    path: str
    format_name: str
    duration_sec: float
    streams: List[AudioStreamInfo]
    has_video: bool
    video_fps: Optional[float] = None
    video_resolution: Optional[Tuple[int, int]] = None
    video_codec: Optional[str] = None
    pixel_format: Optional[str] = None
    is_vfr: bool = False
    media_category: str = "audio"  # "video", "audio", "image", "subtitle"
    file_size: int = 0
    subtitle_count: int = 0


def _parse_timestamp(ts: str) -> float:
    """SRT (00:01:02,345) veya WebVTT (00:01:02.345 / 01:02.345) zaman damgasını saniyeye çevirir."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600.0 + m * 60.0 + s
    elif len(parts) == 2:
        m, s = int(parts[0]), float(parts[1])
        return m * 60.0 + s
    return float(parts[0])


def parse_subtitles(file_path: str | Path) -> List[SubtitleCue]:
    """SRT ve WebVTT altyazı dosyalarını okur ve zamanlı replik listesi döndürür."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Altyazı dosyası bulunamadı: {p}")
    raw = p.read_text(encoding="utf-8-sig", errors="replace").strip()
    if not raw:
        raise ValueError(f"Altyazı dosyası boş: {p.name}")

    # WebVTT başlığını temizle
    lines = raw.splitlines()
    if lines and lines[0].strip().upper().startswith("WEBVTT"):
        lines = lines[1:]

    content = "\n".join(lines).strip()
    blocks = re.split(r"\n\s*\n", content)
    cues: List[SubtitleCue] = []
    time_pattern = re.compile(
        r"((?:\d{1,2}:)?\d{1,2}:\d{1,2}[,\.]\d{1,3})\s*-->\s*((?:\d{1,2}:)?\d{1,2}:\d{1,2}[,\.]\d{1,3})"
    )

    for block in blocks:
        b_lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not b_lines:
            continue
        match_idx = -1
        match_obj = None
        for idx, ln in enumerate(b_lines):
            m = time_pattern.search(ln)
            if m:
                match_idx = idx
                match_obj = m
                break
        if match_obj is None:
            continue
        start_s = _parse_timestamp(match_obj.group(1))
        end_s = _parse_timestamp(match_obj.group(2))
        text_lines = b_lines[match_idx + 1 :]
        # HTML/VTT etiketlerini temizle
        clean_text = re.sub(r"<[^>]+>", "", "\n".join(text_lines)).strip()
        if clean_text and end_s > start_s:
            cues.append(
                SubtitleCue(
                    index=len(cues) + 1,
                    start_sec=start_s,
                    end_sec=end_s,
                    text=clean_text,
                )
            )
    if not cues:
        raise ValueError(f"Geçerli SRT/VTT zaman damgası bulunamadı: {p.name}")
    return cues


class MediaExtractor:
    """Video, ses, görsel ve altyazı dosyalarını gerçek akış düzeyinde inceler ve işler."""

    @staticmethod
    def probe(media_path: str | Path) -> MediaInfo:
        p = Path(media_path)
        if not p.exists():
            raise FileNotFoundError(f"Medya dosyası bulunamadı: {p}")
        if not p.is_file():
            raise ValueError(f"Belirtilen yol bir dosya değil: {p}")

        file_size = p.stat().st_size
        if file_size == 0:
            raise ValueError(f"Medya dosyası boş (0 bayt): {p.name}")

        suffix = p.suffix.lower()

        # 1. Altyazı dosyası kontrolü (.srt, .vtt)
        if suffix in C.SUBTITLE_EXTENSIONS:
            cues = parse_subtitles(p)
            dur = max((c.end_sec for c in cues), default=0.0)
            return MediaInfo(
                path=str(p),
                format_name=suffix.lstrip(".").upper(),
                duration_sec=dur,
                streams=[],
                has_video=False,
                media_category="subtitle",
                file_size=file_size,
                subtitle_count=len(cues),
            )

        # 2. Görsel dosyası kontrolü (.png, .jpg, .jpeg, .webp, .bmp)
        if suffix in C.IMAGE_EXTENSIONS:
            try:
                with Image.open(p) as img:
                    img.verify()
                with Image.open(p) as img2:
                    w, h = img2.size
                    fmt = (img2.format or suffix.lstrip(".")).upper()
                return MediaInfo(
                    path=str(p),
                    format_name=fmt,
                    duration_sec=5.0,  # Varsayılan görsel klip süresi
                    streams=[],
                    has_video=False,
                    video_resolution=(w, h),
                    video_codec=fmt.lower(),
                    media_category="image",
                    file_size=file_size,
                )
            except Exception as exc:
                raise RuntimeError(f"Görsel dosyası bozuk veya desteklenmiyor ({p.name}): {exc}") from exc

        # 3. Video ve Ses dosyalarını PyAV (FFmpeg) ile gerçek akış düzeyinde incele
        try:
            container = av.open(str(p))
        except Exception as exc:
            raise RuntimeError(
                f"Medya kapsayıcısı açılamadı veya dosya bozuk ({p.name}): {exc}"
            ) from exc

        try:
            audio_streams: List[AudioStreamInfo] = []
            has_video = False
            fps: Optional[float] = None
            resolution: Optional[Tuple[int, int]] = None
            video_codec: Optional[str] = None
            pixel_format: Optional[str] = None
            is_vfr = False

            if container.streams.video:
                v_stream = container.streams.video[0]
                v_codec_name = (v_stream.codec_context.name or "").lower()
                # Albüm kapağı (attached_pic) olmayan gerçek video akışı mı?
                is_cover_art = v_codec_name in ("mjpeg", "png", "bmp", "webp") and suffix in C.AUDIO_EXTENSIONS
                if not is_cover_art:
                    has_video = True
                    video_codec = v_codec_name or "unknown"
                    pixel_format = getattr(v_stream.codec_context, "pix_fmt", None)
                    if v_stream.average_rate:
                        fps = float(v_stream.average_rate)
                    elif v_stream.guessed_rate:
                        fps = float(v_stream.guessed_rate)
                    else:
                        fps = 30.0
                    if v_stream.average_rate and v_stream.base_rate:
                        try:
                            if abs(float(v_stream.average_rate) - float(v_stream.base_rate)) > 0.5:
                                is_vfr = True
                        except Exception:
                            pass
                    resolution = (v_stream.width or 1920, v_stream.height or 1080)

            for idx, a_stream in enumerate(container.streams.audio):
                dur = 0.0
                if a_stream.duration and a_stream.time_base:
                    dur = float(a_stream.duration * a_stream.time_base)
                elif container.duration:
                    dur = float(container.duration / 1_000_000.0)

                lang = a_stream.metadata.get("language") if a_stream.metadata else None
                audio_streams.append(
                    AudioStreamInfo(
                        index=idx,
                        codec_name=a_stream.codec_context.name or "unknown",
                        sample_rate=a_stream.codec_context.sample_rate or 48000,
                        channels=a_stream.codec_context.channels or 2,
                        duration_sec=dur,
                        bit_rate=a_stream.codec_context.bit_rate,
                        language=lang,
                    )
                )

            container_dur = float(container.duration / 1_000_000.0) if container.duration else 0.0
            if container_dur <= 0 and audio_streams:
                container_dur = max((s.duration_sec for s in audio_streams), default=0.0)

            # Eğer süre hâlâ 0 ise (bazı WAV/FLAC veya ham akışlar), soundfile ile kontrol et
            if container_dur <= 0 and not has_video:
                try:
                    sf_info = sf.info(str(p))
                    if sf_info.samplerate > 0:
                        container_dur = float(sf_info.frames) / float(sf_info.samplerate)
                except Exception:
                    pass

            fmt_name = container.format.name if container.format else suffix.lstrip(".")
        finally:
            container.close()

        if not has_video and not audio_streams:
            raise RuntimeError(
                f"Dosyada desteklenen bir video veya ses akışı bulunamadı: {p.name}"
            )

        category = "video" if has_video else "audio"
        return MediaInfo(
            path=str(p),
            format_name=fmt_name,
            duration_sec=max(0.05, container_dur),
            streams=audio_streams,
            has_video=has_video,
            video_fps=fps,
            video_resolution=resolution,
            video_codec=video_codec,
            pixel_format=pixel_format,
            is_vfr=is_vfr,
            media_category=category,
            file_size=file_size,
        )

    @staticmethod
    def extract_audio(
        media_path: str | Path,
        output_wav_path: Optional[str | Path] = None,
        stream_index: int = 0,
        target_sample_rate: int = 48000,
        target_channels: int = 2,
    ) -> Tuple[np.ndarray, int]:
        """
        Medya dosyasından seçili ses akışını okur, hedef örnekleme hızına ve kanal sayısına
        dönüştürerek float32 numpy dizisi olarak döndürür. İsteğe bağlı olarak WAV'a kaydeder.
        """
        path = str(media_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Medya dosyası bulunamadı: {path}")

        container = av.open(path)
        try:
            if not container.streams.audio:
                raise RuntimeError(f"Dosyada ses akışı bulunamadı: {Path(path).name}")

            if stream_index >= len(container.streams.audio):
                stream_index = 0

            audio_stream = container.streams.audio[stream_index]

            resampler = av.AudioResampler(
                format="fltp",
                layout="stereo" if target_channels == 2 else "mono",
                rate=target_sample_rate,
            )

            frames_data = []
            for frame in container.decode(audio_stream):
                resampled_frames = resampler.resample(frame)
                if resampled_frames:
                    for rf in resampled_frames:
                        arr = rf.to_ndarray()
                        frames_data.append(arr)

            # Resampler tamponunu boşalt
            flushed = resampler.resample(None)
            if flushed:
                for rf in flushed:
                    frames_data.append(rf.to_ndarray())
        finally:
            container.close()

        if not frames_data:
            raise RuntimeError(f"Ses akışından veri okunamadı: {Path(path).name}")

        concatenated = np.concatenate(frames_data, axis=1)
        if target_channels == 1:
            audio_array = concatenated[0].astype(np.float32)
        else:
            audio_array = concatenated.T.astype(np.float32)

        if output_wav_path is not None:
            out_p = Path(output_wav_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_p), audio_array, target_sample_rate, subtype="PCM_24")
            logger.info(f"Ses akışı çıkarıldı ve kaydedildi: {output_wav_path}")

        return audio_array, target_sample_rate

    @staticmethod
    def read_audio_any(
        media_path: str | Path,
        target_sample_rate: Optional[int] = None,
        always_2d: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Herhangi bir ses veya video dosyasından (WAV, FLAC, MP3, OGG, M4A, AAC, Opus, WMA, MP4, MKV vb.)
        ses verisini float32 numpy dizisi olarak okur.
        Önce hızlı libsndfile dener, desteklenmiyorsa PyAV/FFmpeg ile çözer.
        """
        p = Path(media_path)
        if not p.exists():
            raise FileNotFoundError(f"Ses dosyası bulunamadı: {p}")

        try:
            data, sr = sf.read(str(p), dtype="float32", always_2d=always_2d)
            if target_sample_rate and sr != target_sample_rate:
                from scipy import signal
                num_out = max(1, int(round(len(data) * float(target_sample_rate) / float(sr))))
                data = signal.resample(data, num_out, axis=0).astype(np.float32)
                sr = target_sample_rate
            return data, int(sr)
        except Exception:
            # PyAV fallback (M4A, AAC, Opus, WMA veya Video kapsayıcıları)
            info = MediaExtractor.probe(p)
            if not info.streams:
                raise RuntimeError(f"Dosyada okunabilir ses akışı yok: {p.name}")
            sr = target_sample_rate or info.streams[0].sample_rate or 48000
            ch = min(2, max(1, info.streams[0].channels))
            arr, out_sr = MediaExtractor.extract_audio(
                p, target_sample_rate=sr, target_channels=ch
            )
            if always_2d and arr.ndim == 1:
                arr = arr[:, np.newaxis]
            return arr, out_sr

    @staticmethod
    def extract_thumbnail(
        media_path: str | Path,
        size: Tuple[int, int] = (160, 90),
        time_sec: Optional[float] = None,
    ) -> Optional[Path]:
        """
        Video veya görsel dosyasından küçük resim (PNG thumbnail) üretir ve önbellek yolunu döndürür.
        """
        p = Path(media_path)
        if not p.exists():
            return None

        thumb_dir = paths.cache_dir() / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        stat = p.stat()
        key = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime}|{size[0]}x{size[1]}|{time_sec}"
        digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:20]
        out_png = thumb_dir / f"thumb_{digest}.png"
        if out_png.exists() and out_png.stat().st_size > 0:
            return out_png

        suffix = p.suffix.lower()
        try:
            if suffix in C.IMAGE_EXTENSIONS:
                with Image.open(p) as img:
                    rgb = img.convert("RGB")
                    rgb.thumbnail(size, Image.Resampling.BILINEAR)
                    canvas = Image.new("RGB", size, (18, 20, 25))
                    ox = (size[0] - rgb.width) // 2
                    oy = (size[1] - rgb.height) // 2
                    canvas.paste(rgb, (ox, oy))
                    canvas.save(out_png, "PNG")
                    return out_png

            container = av.open(str(p))
            try:
                if not container.streams.video:
                    return None
                v_stream = container.streams.video[0]
                if time_sec is not None and time_sec > 0 and v_stream.time_base:
                    pts = int(time_sec / float(v_stream.time_base))
                    container.seek(pts, stream=v_stream)
                for frame in container.decode(v_stream):
                    img = frame.to_image().convert("RGB")
                    img.thumbnail(size, Image.Resampling.BILINEAR)
                    canvas = Image.new("RGB", size, (18, 20, 25))
                    ox = (size[0] - img.width) // 2
                    oy = (size[1] - img.height) // 2
                    canvas.paste(img, (ox, oy))
                    canvas.save(out_png, "PNG")
                    return out_png
            finally:
                container.close()
        except Exception as exc:
            logger.debug(f"Thumbnail üretilemedi ({p.name}): {exc}")
        return None

    @staticmethod
    def get_supported_capabilities() -> Dict[str, object]:
        """
        Sistemde kurulu PyAV / FFmpeg ve kütüphanelerin gerçek zamanlı kodek/format
        desteğini ve donanım hızlandırma durumunu döndürür.
        """
        codecs_avail = av.codecs_available
        formats_avail = av.formats_available

        def _has_codec(name: str) -> bool:
            return name in codecs_avail

        def _can_encode(name: str) -> bool:
            if name not in codecs_avail:
                return False
            try:
                c = av.Codec(name, "w")
                return c is not None
            except Exception:
                return False

        # NVENC gerçek donanım testi (yalnızca adının listede olması yetmez, bağlam açılabilmeli)
        nvenc_h264 = False
        try:
            ctx = av.CodecContext.create("h264_nvenc", "w")
            ctx.width = 256
            ctx.height = 256
            ctx.pix_fmt = "yuv420p"
            ctx.time_base = av.Rational(1, 30)
            ctx.open()
            nvenc_h264 = True
        except Exception:
            nvenc_h264 = False

        lib_versions = getattr(av, "library_versions", {})
        libavcodec_ver = ".".join(str(x) for x in lib_versions.get("libavcodec", ())) or "bilinmiyor"
        libavformat_ver = ".".join(str(x) for x in lib_versions.get("libavformat", ())) or "bilinmiyor"

        return {
            "pyav_version": getattr(av, "__version__", "bilinmiyor"),
            "libavcodec": libavcodec_ver,
            "libavformat": libavformat_ver,
            "nvenc_h264": nvenc_h264,
            "video_containers": [
                ext for ext in C.VIDEO_EXTENSIONS
                if ext.lstrip(".") in formats_avail or ext in (".mkv", ".mov", ".ts", ".wmv", ".mpg", ".mpeg", ".m4v")
            ],
            "video_decoders": {
                "H.264 / AVC": _has_codec("h264"),
                "H.265 / HEVC": _has_codec("hevc"),
                "VP9": _has_codec("vp9"),
                "VP8": _has_codec("vp8"),
                "AV1": _has_codec("av1") or _has_codec("libdav1d"),
                "MPEG-4": _has_codec("mpeg4"),
                "ProRes": _has_codec("prores"),
            },
            "video_encoders": {
                "H.264 (libx264 CPU)": _can_encode("libx264"),
                "H.264 (NVIDIA NVENC GPU)": nvenc_h264,
                "H.265 / HEVC (libx265)": _can_encode("libx265"),
                "VP9 (libvpx-vp9)": _can_encode("libvpx-vp9"),
                "MPEG-4": _can_encode("mpeg4"),
            },
            "audio_codecs": {
                "PCM WAV (16/24/32-bit)": _has_codec("pcm_s16le") and _has_codec("pcm_s24le"),
                "MP3 (LAME)": _has_codec("mp3"),
                "FLAC (Kayıpsız)": _has_codec("flac"),
                "AAC (M4A/MP4)": _has_codec("aac"),
                "Opus": _has_codec("opus") or _has_codec("libopus"),
                "Vorbis (OGG)": _has_codec("vorbis"),
                "ALAC (Apple Lossless)": _has_codec("alac"),
                "AIFF": _has_codec("pcm_s16be"),
                "WMA": _has_codec("wmav2"),
            },
            "image_formats": ["PNG", "JPG / JPEG", "WebP", "BMP"],
            "subtitle_formats": ["SRT (SubRip)", "VTT (WebVTT)"],
        }
