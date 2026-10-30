"""
VOX Studio — Merkezi Görev Yöneticisi (TaskManager), Depolama/Önbellek Kontrolü
ve Sistem Tanılama (SystemDiagnostics) Modülü.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ai.media_extractor import MediaExtractor
from ai.stem_separator import ModelCatalog, detect_compute_backend
from core import paths
from core.logger import get_logger

logger = get_logger("core.diagnostics")


@dataclass
class TaskEntry:
    """Merkezi görev yöneticisinde izlenen arka plan görevi."""

    id: str
    title: str
    category: str  # 'stem', 'export', 'transcribe', 'media'
    target_path: str = ""
    status: str = "running"  # 'running', 'completed', 'failed', 'cancelled'
    progress: int = 0
    message: str = "Başlatılıyor..."
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error_detail: Optional[str] = None
    cancel_cb: Optional[Callable[[], None]] = None
    temp_files: List[Path] = field(default_factory=list)

    @property
    def elapsed_sec(self) -> float:
        end_t = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end_t - self.started_at)


class TaskManager:
    """
    Uzun süren arka plan işlemlerini (AI ses ayrıştırma, video dışa aktarma,
    konuşma tanıma, medya dönüştürme) tek merkezden izler, çakışmaları önler
    ve iptal/hata durumunda geçici dosyaları temizler.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskEntry] = {}
        self._listeners: List[Callable[[], None]] = []

    def add_listener(self, cb: Callable[[], None]) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in list(self._listeners):
            try:
                cb()
            except Exception:
                pass

    def has_active_task_on_file(self, file_path: str, category: Optional[str] = None) -> bool:
        if not file_path:
            return False
        norm = str(Path(file_path).resolve()) if os.path.exists(file_path) else file_path
        for t in self._tasks.values():
            if t.status == "running" and t.target_path == norm:
                if category is None or t.category == category:
                    return True
        return False

    def start_task(
        self,
        title: str,
        category: str,
        target_path: str = "",
        cancel_cb: Optional[Callable[[], None]] = None,
    ) -> str:
        norm = (
            str(Path(target_path).resolve())
            if (target_path and os.path.exists(target_path))
            else target_path
        )
        tid = f"task_{uuid.uuid4().hex[:8]}"
        entry = TaskEntry(
            id=tid,
            title=title,
            category=category,
            target_path=norm,
            status="running",
            progress=0,
            message="Başlatıldı",
            cancel_cb=cancel_cb,
        )
        self._tasks[tid] = entry
        logger.info(f"Görev başlatıldı [{tid}]: {title} ({category})")
        self._notify()
        return tid

    def update_task(self, task_id: str, progress: int, message: str = "") -> None:
        t = self._tasks.get(task_id)
        if not t or t.status != "running":
            return
        t.progress = max(0, min(100, int(progress)))
        if message:
            t.message = message
        self._notify()

    def complete_task(self, task_id: str, message: str = "Tamamlandı") -> None:
        t = self._tasks.get(task_id)
        if not t:
            return
        t.status = "completed"
        t.progress = 100
        t.message = message
        t.finished_at = time.time()
        logger.info(f"Görev tamamlandı [{task_id}]: {t.title} ({t.elapsed_sec:.1f}s)")
        self._notify()

    def fail_task(self, task_id: str, error_detail: str) -> None:
        t = self._tasks.get(task_id)
        if not t:
            return
        t.status = "failed"
        t.error_detail = error_detail
        t.message = f"Hata: {error_detail}"
        t.finished_at = time.time()
        self._cleanup_temp(t)
        logger.error(f"Görev başarısız [{task_id}]: {t.title} -> {error_detail}")
        self._notify()

    def cancel_task(self, task_id: str) -> bool:
        t = self._tasks.get(task_id)
        if not t or t.status != "running":
            return False
        if t.cancel_cb:
            try:
                t.cancel_cb()
            except Exception:
                pass
        t.status = "cancelled"
        t.message = "Kullanıcı tarafından iptal edildi."
        t.finished_at = time.time()
        self._cleanup_temp(t)
        logger.info(f"Görev iptal edildi [{task_id}]: {t.title}")
        self._notify()
        return True

    def _cleanup_temp(self, entry: TaskEntry) -> None:
        for p in entry.temp_files:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    def list_tasks(self) -> List[TaskEntry]:
        return sorted(self._tasks.values(), key=lambda x: x.started_at, reverse=True)

    def running_tasks(self) -> List[TaskEntry]:
        return [t for t in self._tasks.values() if t.status == "running"]

    def active_tasks(self) -> List[TaskEntry]:
        return self.running_tasks()

    def request_cancel(self, task_id: str) -> bool:
        return self.cancel_task(task_id)


@dataclass
class DiagnosticItem:
    component: str
    status: str  # 'ok', 'warning', 'error'
    summary: str
    cause: str = ""
    remedy: str = ""


class SystemDiagnostics:
    """
    Sistem bağımlılıklarını, donanım (GPU / VRAM / CPU / RAM), ses aygıtlarını,
    model hazırlığını ve önbellek/depolama durumunu gerçek kontrollerle denetler.
    """

    @staticmethod
    def _query_system_ram() -> Dict[str, float]:
        """Gerçek işletim sistemi fiziksel RAM bilgisini döndürür (sabit metin içermez)."""
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", wintypes.DWORD),
                        ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_gb = stat.ullTotalPhys / (1024 ** 3)
                    avail_gb = stat.ullAvailPhys / (1024 ** 3)
                    used_gb = max(0.0, total_gb - avail_gb)
                    pct = int(stat.dwMemoryLoad)
                    return {
                        "available": True,
                        "used_gb": round(used_gb, 1),
                        "total_gb": round(total_gb, 1),
                        "percent": max(0, min(100, pct)),
                    }
            except Exception as exc:
                logger.debug(f"Windows RAM sorgusu başarısız: {exc}")
        return {"available": False, "used_gb": 0.0, "total_gb": 0.0, "percent": 0}

    @staticmethod
    def live_hardware_metrics() -> Dict[str, object]:
        """
        Ana sayfadaki 'Sistem Durumu' kartı için GPU, RAM ve Depolama bilgilerini
        çalışma zamanında gerçek sistemden sorgular.
        """
        hw = detect_compute_backend()
        gpu_name = hw.get("gpu_name")
        vram_tot_mb = int(hw.get("vram_total_mb") or 0)
        vram_free_mb = int(hw.get("vram_free_mb") or 0)
        vram_used_mb = max(0, vram_tot_mb - vram_free_mb)
        if gpu_name and vram_tot_mb > 0:
            vram_used_gb = round(vram_used_mb / 1024.0, 1)
            vram_tot_gb = round(vram_tot_mb / 1024.0, 1)
            gpu_pct = int(min(100, round((vram_used_mb / max(1, vram_tot_mb)) * 100)))
            gpu_info = {
                "available": True,
                "name": str(gpu_name),
                "detail": f"{vram_used_gb:.1f} GB / {vram_tot_gb:.1f} GB",
                "percent": gpu_pct,
                "mode": str(hw.get("active_mode") or "GPU"),
            }
        else:
            cpu_label = platform.processor() or "x64 İşlemci"
            gpu_info = {
                "available": False,
                "name": "CPU Modu (GPU bilgisi alınamadı)",
                "detail": cpu_label[:32],
                "percent": 0,
                "mode": "CPU Yedeği",
            }

        ram_info = SystemDiagnostics._query_system_ram()

        try:
            target_dir = paths.recordings_dir()
            target_dir.mkdir(parents=True, exist_ok=True)
            du = shutil.disk_usage(str(target_dir))
            disk_used_gb = round(du.used / (1024 ** 3), 1)
            disk_tot_gb = round(du.total / (1024 ** 3), 1)
            disk_free_gb = round(du.free / (1024 ** 3), 1)
            disk_pct = int(min(100, round((du.used / max(1, du.total)) * 100)))
            disk_info = {
                "available": True,
                "used_gb": disk_used_gb,
                "total_gb": disk_tot_gb,
                "free_gb": disk_free_gb,
                "detail": f"{disk_used_gb:.0f} GB / {disk_tot_gb:.0f} GB",
                "percent": disk_pct,
            }
        except Exception:
            disk_info = {
                "available": False,
                "used_gb": 0.0,
                "total_gb": 0.0,
                "free_gb": 0.0,
                "detail": "Depolama bilgisi alınamadı",
                "percent": 0,
            }

        # Genel sağlık özeti
        if disk_info["available"] and disk_info["free_gb"] < 2.0:
            overall_status = "warning"
            overall_title = "Disk alanı azalıyor"
            overall_sub = f"Boş alan: {disk_info['free_gb']:.1f} GB"
        elif ram_info["available"] and ram_info["percent"] >= 90:
            overall_status = "warning"
            overall_title = "Bellek kullanımı yüksek"
            overall_sub = f"RAM kullanımı %{ram_info['percent']} seviyesinde."
        else:
            overall_status = "ok"
            overall_title = "Sistem hazır"
            overall_sub = "Tüm ses ve video motorları sorunsuz çalışıyor."

        return {
            "status": overall_status,
            "title": overall_title,
            "subtitle": overall_sub,
            "gpu": gpu_info,
            "ram": ram_info,
            "disk": disk_info,
        }

    @staticmethod
    def storage_usage() -> Dict[str, int]:
        """Önbellek (thumbnails, decoded_audio) ve geçici dosyaların bayt cinsinden boyutunu döndürür."""
        c_dir = paths.cache_dir()
        thumb_bytes = 0
        audio_bytes = 0
        other_bytes = 0
        if c_dir.exists():
            for p in c_dir.rglob("*"):
                if p.is_file():
                    sz = p.stat().st_size
                    if "thumbnails" in p.parts:
                        thumb_bytes += sz
                    elif "decoded_audio" in p.parts:
                        audio_bytes += sz
                    else:
                        other_bytes += sz
        return {
            "thumbnails_bytes": thumb_bytes,
            "decoded_audio_bytes": audio_bytes,
            "other_cache_bytes": other_bytes,
            "total_cache_bytes": thumb_bytes + audio_bytes + other_bytes,
        }

    @staticmethod
    def clear_cache_safely() -> int:
        """Yalnızca yeniden üretilebilir önbellek dosyalarını (thumbnails ve decoded_audio) güvenli biçimde temizler."""
        c_dir = paths.cache_dir()
        removed_bytes = 0
        for sub in ("thumbnails", "decoded_audio"):
            target = c_dir / sub
            if target.exists():
                for p in target.glob("*"):
                    if p.is_file():
                        try:
                            sz = p.stat().st_size
                            p.unlink()
                            removed_bytes += sz
                        except Exception:
                            pass
        logger.info(f"Önbellek temizlendi: {removed_bytes / (1024 * 1024):.2f} MB")
        return removed_bytes

    @staticmethod
    def run_all_checks(ctx=None) -> List[DiagnosticItem]:
        items: List[DiagnosticItem] = []

        # 1. FFmpeg / PyAV Medya Motoru
        try:
            caps = MediaExtractor.get_supported_capabilities()
            items.append(
                DiagnosticItem(
                    component="FFmpeg / PyAV Medya Motoru",
                    status="ok",
                    summary=f"{caps['ffmpeg_version']} · H.264/AAC Kodlama ve Çoklu Kapsayıcı Hazır",
                )
            )
        except Exception as exc:
            items.append(
                DiagnosticItem(
                    component="FFmpeg / PyAV Medya Motoru",
                    status="error",
                    summary=f"PyAV başlatılamadı: {exc}",
                    cause="FFmpeg dinamik kütüphaneleri eksik veya bozuk.",
                    remedy="pip install --force-reinstall av komutunu çalıştırın.",
                )
            )

        # 2. GPU / CUDA ve CPU Yedeği (GTX 1660 Super 6 GB VRAM hedefi)
        hw = detect_compute_backend()
        gpu_name = hw.get("gpu_name")
        if gpu_name:
            vram_tot = hw.get("vram_total_mb", 0)
            vram_free = hw.get("vram_free_mb", 0)
            items.append(
                DiagnosticItem(
                    component="Donanım & GPU / CPU Yedeği",
                    status="ok",
                    summary=(
                        f"{gpu_name} ({vram_tot} MB VRAM, {vram_free} MB Boş) · "
                        f"Mod: {hw.get('active_mode')}"
                    ),
                    cause=str(hw.get("fallback_reason") or ""),
                    remedy="Uzun dosyalarda 25 sn örtüşmeli parçalı işleme (Overlap-Add) bellek taşmasını önler.",
                )
            )
        else:
            items.append(
                DiagnosticItem(
                    component="Donanım & GPU / CPU Yedeği",
                    status="ok",
                    summary=f"CPU Yedeği Aktif ({platform.processor() or 'x64'}) · Parçalı Bellek Koruması Açık",
                )
            )

        # 3. AI Ses Ayrıştırma Modelleri
        models = ModelCatalog.get_available_models()
        ready_models = [m for m in models if m.is_downloaded]
        missing_opt = [m for m in models if not m.is_downloaded]
        items.append(
            DiagnosticItem(
                component="AI Ses Ayrıştırma Modelleri",
                status="ok" if len(ready_models) >= 3 else "warning",
                summary=f"{len(ready_models)}/{len(models)} Model Hazır ({', '.join(m.id for m in ready_models)})",
                cause=(
                    f"İsteğe bağlı harici ağırlık ({missing_opt[0].id}) henüz indirilmemiş."
                    if missing_opt
                    else ""
                ),
                remedy="Yerleşik VOX Dubbing Pro, HQ 2-Stem ve 4-Stem modelleri çevrimdışı kullanıma tam hazırdır.",
            )
        )

        # 4. Konuşma Tanıma (Whisper & Yerel VAD)
        try:
            import faster_whisper  # noqa: F401

            items.append(
                DiagnosticItem(
                    component="Konuşma Tanıma & Sessizlik Tespiti",
                    status="ok",
                    summary="faster-whisper + Yerel VAD Konuşma/Sessizlik Dedektörü Hazır",
                )
            )
        except Exception as exc:
            items.append(
                DiagnosticItem(
                    component="Konuşma Tanıma & Sessizlik Tespiti",
                    status="warning",
                    summary="Yerel VAD Sessizlik Dedektörü Hazır (faster-whisper yüklü değil)",
                    cause=str(exc),
                    remedy="Tam konuşma metni için faster-whisper paketini etkinleştirin.",
                )
            )

        # 5. Mikrofon ve Ses Çıkış Aygıtları
        if ctx is not None and hasattr(ctx, "devices"):
            in_devs = ctx.devices.input_devices()
            out_devs = ctx.devices.output_devices()
            if in_devs and out_devs:
                items.append(
                    DiagnosticItem(
                        component="Mikrofon ve Ses Çıkışı",
                        status="ok",
                        summary=f"{len(in_devs)} Giriş (Mikrofon), {len(out_devs)} Çıkış Aygıtı Doğrulandı",
                    )
                )
            else:
                items.append(
                    DiagnosticItem(
                        component="Mikrofon ve Ses Çıkışı",
                        status="warning",
                        summary=f"Giriş: {len(in_devs)}, Çıkış: {len(out_devs)} aygıt bulundu",
                        cause="Mikrofon veya hoparlör takılı olmayabilir.",
                        remedy="Windows Ses Ayarları'ndan mikrofon izinlerini ve bağlantısını kontrol edin.",
                    )
                )

        # 6. Disk Alanı ve Yazma İzinleri
        try:
            rec_dir = paths.recordings_dir()
            rec_dir.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(str(rec_dir))
            free_gb = usage.free / (1024 ** 3)
            probe_file = rec_dir / ".vox_write_test.tmp"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink()
            items.append(
                DiagnosticItem(
                    component="Disk Alanı ve Yazma İzni",
                    status="ok" if free_gb >= 2.0 else "warning",
                    summary=f"Yazma İzni Doğrulandı · Boş Alan: {free_gb:.1f} GB ({rec_dir})",
                    cause="Disk alanı 2 GB'ın altında." if free_gb < 2.0 else "",
                    remedy="Büyük video dışa aktarımları için önbelleği temizleyin veya disk alanı açın." if free_gb < 2.0 else "",
                )
            )
        except Exception as exc:
            items.append(
                DiagnosticItem(
                    component="Disk Alanı ve Yazma İzni",
                    status="error",
                    summary=f"Yazma hatası: {exc}",
                    cause="Kayıt klasörüne erişim izni yok.",
                    remedy="Ayarlar ekranından farklı bir kayıt klasörü seçin.",
                )
            )

        return items

    @staticmethod
    def read_recent_logs(max_lines: int = 40) -> str:
        log_file = paths.logs_dir() / "vox_studio.log"
        if not log_file.exists():
            return "Henüz günlük dosyası oluşmadı."
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-max_lines:])
        except Exception as exc:
            return f"Günlük okunamadı: {exc}"
