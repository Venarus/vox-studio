"""
VOX Studio — Aktif Proje Yöneticisi
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from core.config import ConfigManager
from core.constants import PROJECTS_DIR
from core.logger import get_logger
from project.database import ProjectDatabase
from project.models import Project

logger = get_logger("project.manager")


class ProjectManager:
    """Tekil aktif projeyi, son projeleri ve proje oluşturma/açma döngüsünü yönetir."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.active_db: Optional[ProjectDatabase] = None
        self.active_project: Optional[Project] = None
        self._on_project_changed: List[Callable[[Optional[Project]], None]] = []

    @property
    def current_file(self) -> Optional[Path]:
        if self.active_db is not None:
            return Path(self.active_db.db_path)
        return None

    def on_changed(self, callback: Callable[[Optional[Project]], None]) -> None:
        self._on_project_changed.append(callback)

    def _notify(self) -> None:
        for cb in self._on_project_changed:
            try:
                cb(self.active_project)
            except Exception as e:
                logger.error(f"Proje değişim bildirimi hatası: {e}")

    def create_project(
        self,
        name: str,
        folder_path: Optional[str | Path] = None,
        description: str = "",
        sample_rate: int = 48000,
        channels: int = 1,
        video_path: Optional[str] = None,
        project_type: str = "Video Projesi",
        resolution: Tuple[int, int] = (1920, 1080),
        fps: float = 30.0,
    ) -> Project:
        if folder_path is None:
            safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip() or "Yeni Proje"
            base_dir = PROJECTS_DIR / safe_name
            target_dir = base_dir
            counter = 1
            while target_dir.exists() and (target_dir / "project.voxproj").exists():
                target_dir = PROJECTS_DIR / f"{safe_name}_{counter}"
                counter += 1
        else:
            target_dir = Path(folder_path)
            if (target_dir / "project.voxproj").exists():
                safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip() or "Yeni Proje"
                sub = target_dir / safe_name
                counter = 1
                while sub.exists() and (sub / "project.voxproj").exists():
                    sub = target_dir / f"{safe_name}_{counter}"
                    counter += 1
                target_dir = sub

        target_dir.mkdir(parents=True, exist_ok=True)
        db_file = target_dir / "project.voxproj"

        desc_final = description.strip() if description.strip() else project_type
        db = ProjectDatabase(db_file)
        proj = db.create_project(
            name=name,
            description=desc_final,
            sample_rate=sample_rate,
            channels=channels,
            video_path=video_path,
        )

        self.active_db = db
        self.active_project = proj

        # Başlangıç timeline durumunu kaydet (çözünürlük, fps ve proje türü korunur)
        init_tl = {
            "name": name,
            "project_type": project_type,
            "fps": float(fps),
            "resolution": [int(resolution[0]), int(resolution[1])],
            "sample_rate": int(sample_rate),
            "playhead_pos": 0.0,
            "tracks": [],
            "markers": [],
        }
        init_bin = [video_path] if (video_path and Path(video_path).exists()) else []
        self.save_timeline_state(init_tl, init_bin)

        self.config.add_recent_project(str(db_file))
        self.config.set("last_project_path", str(db_file))
        self._notify()
        logger.info(f"Yeni proje oluşturuldu ve açıldı: {proj.name} ({db_file})")
        return proj

    def open_project(self, db_path: str | Path) -> Optional[Project]:
        db_path = Path(db_path)
        if not db_path.exists():
            logger.error(f"Proje dosyası bulunamadı: {db_path}")
            return None

        try:
            db = ProjectDatabase(db_path)
            proj = db.get_project()
            if not proj:
                logger.error(f"Dosyada geçerli proje kaydı yok: {db_path}")
                return None

            self.active_db = db
            self.active_project = proj
            self.config.add_recent_project(str(db_path))
            self.config.set("last_project_path", str(db_path))
            self._notify()
            logger.info(f"Proje açıldı: {proj.name} ({db_path})")
            return proj
        except Exception as e:
            logger.error(f"Proje açılırken hata: {e}")
            return None

    def close_project(self) -> None:
        self.active_db = None
        self.active_project = None
        self.config.set("last_project_path", "")
        self._notify()
        logger.info("Aktif proje kapatıldı.")

    def auto_save(self) -> None:
        if self.active_project and self.active_db:
            # WAL modunda veri zaten anlık senkronizedir; güncelleme zamanını yenile
            self.active_db.update_project()
            logger.debug(f"Otomatik kaydetme yapıldı: {self.active_project.name}")

    def timeline_state_path(self) -> Optional[Path]:
        if not self.active_db:
            return None
        return Path(self.active_db.db_path).with_suffix(".timeline.json")

    def save_timeline_state(
        self,
        timeline_dict: Dict[str, object],
        media_bin: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """Aktif projenin timeline ve medya kutusu durumunu atomik olarak kaydeder."""
        state_file = self.timeline_state_path()
        if state_file is None:
            return None
        payload = {
            "timeline": timeline_dict,
            "media_bin": list(media_bin or []),
            "saved_at": time.time(),
        }
        tmp_file = state_file.with_suffix(".timeline.json.tmp")
        tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_file, state_file)
        if self.active_db and self.active_project:
            new_name = str(timeline_dict.get("name") or self.active_project.name)
            self.active_project.name = new_name
            self.active_db.update_project(name=new_name)
        logger.info(f"Timeline durumu kaydedildi: {state_file}")
        return state_file

    def load_timeline_state(self) -> Tuple[Optional[Dict[str, object]], List[str]]:
        """Aktif projenin timeline ve medya kutusu durumunu yükler."""
        state_file = self.timeline_state_path()
        if state_file is None or not state_file.exists():
            return None, []
        try:
            payload = json.loads(state_file.read_text(encoding="utf-8"))
            tl = payload.get("timeline")
            bin_items = [str(p) for p in (payload.get("media_bin") or [])]
            return (tl if isinstance(tl, dict) else None), bin_items
        except Exception as exc:
            logger.error(f"Timeline durumu okunamadı ({state_file}): {exc}")
            return None, []

    def list_recent_projects(self) -> List[dict]:
        recent_paths = list(self.config.get("recent_projects", []) or [])
        seen_resolved = set()
        candidate_paths: List[Path] = []

        for p_str in recent_paths:
            p = Path(p_str)
            if p.exists() and p.is_file():
                res_str = str(p.resolve())
                if res_str not in seen_resolved:
                    seen_resolved.add(res_str)
                    candidate_paths.append(p)

        # Ayrıca PROJECTS_DIR altındaki gerçek .voxproj dosyalarını tara
        try:
            if PROJECTS_DIR.exists():
                for p in PROJECTS_DIR.rglob("*.voxproj"):
                    if p.is_file():
                        res_str = str(p.resolve())
                        if res_str not in seen_resolved:
                            seen_resolved.add(res_str)
                            candidate_paths.append(p)
        except Exception:
            pass

        result: List[dict] = []
        valid_paths: List[str] = []
        for p in candidate_paths:
            try:
                db = ProjectDatabase(p)
                proj = db.get_project()
                if not proj:
                    continue
                valid_paths.append(str(p))

                tl_file = p.with_suffix(".timeline.json")
                tl_dict: Optional[dict] = None
                media_bin: List[str] = []
                if tl_file.exists():
                    try:
                        payload = json.loads(tl_file.read_text(encoding="utf-8"))
                        if isinstance(payload.get("timeline"), dict):
                            tl_dict = payload["timeline"]
                        media_bin = [str(m) for m in (payload.get("media_bin") or [])]
                    except Exception:
                        pass

                # Proje türünü belirle
                proj_type = "Video Projesi"
                desc_lower = (proj.description or "").lower()
                if tl_dict and tl_dict.get("project_type"):
                    proj_type = str(tl_dict["project_type"])
                elif "dublaj" in desc_lower:
                    proj_type = "Dublaj Projesi"
                elif "stem" in desc_lower or "ayrıştırma" in desc_lower:
                    proj_type = "Ses Ayrıştırma"
                elif "ses" in desc_lower:
                    proj_type = "Ses Projesi"

                # Süre ve çözünürlük hesapla
                duration_sec = 0.0
                res_str = "1920x1080"
                fps_val = 30.0
                first_visual_media: Optional[str] = proj.video_path if (proj.video_path and Path(proj.video_path).exists()) else None
                if tl_dict:
                    res = tl_dict.get("resolution")
                    if isinstance(res, (list, tuple)) and len(res) == 2:
                        res_str = f"{res[0]}x{res[1]}"
                    fps_val = float(tl_dict.get("fps") or 30.0)
                    max_out = 0.0
                    for tr in tl_dict.get("tracks", []):
                        for cl in tr.get("clips", []):
                            t_out = float(cl.get("timeline_out") or 0.0)
                            if t_out > max_out:
                                max_out = t_out
                            mp = cl.get("media_path")
                            if not first_visual_media and mp and Path(mp).exists():
                                ext = Path(mp).suffix.lower()
                                if ext in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".png", ".jpg", ".jpeg", ".webp"):
                                    first_visual_media = str(mp)
                    duration_sec = max_out

                if not first_visual_media:
                    for mp in media_bin:
                        if mp and Path(mp).exists():
                            ext = Path(mp).suffix.lower()
                            if ext in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".png", ".jpg", ".jpeg", ".webp"):
                                first_visual_media = mp
                                break

                thumb_path: Optional[str] = None
                if first_visual_media:
                    ext = Path(first_visual_media).suffix.lower()
                    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                        thumb_path = first_visual_media
                    else:
                        try:
                            from ai.media_extractor import MediaExtractor

                            tp = MediaExtractor.extract_thumbnail(first_visual_media)
                            if tp and Path(tp).exists():
                                thumb_path = str(tp)
                        except Exception:
                            thumb_path = None

                missing = self.find_missing_media(tl_dict, media_bin)

                result.append({
                    "name": proj.name,
                    "path": str(p),
                    "folder": str(p.parent),
                    "created_at": float(proj.created_at or proj.updated_at or 0.0),
                    "updated_at": float(proj.updated_at or 0.0),
                    "description": proj.description,
                    "project_type": proj_type,
                    "duration_sec": duration_sec,
                    "resolution": res_str,
                    "fps": fps_val,
                    "sample_rate": int(proj.sample_rate or 48000),
                    "thumbnail_path": thumb_path,
                    "missing_media_count": len(missing),
                })
            except Exception:
                continue

        if valid_paths != recent_paths:
            self.config.set("recent_projects", valid_paths)
        result.sort(key=lambda r: r.get("updated_at", 0.0), reverse=True)
        return result

    def rename_project(self, db_path: str | Path, new_name: str) -> bool:
        """Projeyi yeniden adlandırır (hem .voxproj hem .timeline.json içinde günceller)."""
        p = Path(db_path)
        clean_name = new_name.strip()
        if not p.exists() or not clean_name:
            return False
        try:
            db = ProjectDatabase(p)
            db.update_project(name=clean_name)
            tl_file = p.with_suffix(".timeline.json")
            if tl_file.exists():
                try:
                    payload = json.loads(tl_file.read_text(encoding="utf-8"))
                    if isinstance(payload.get("timeline"), dict):
                        payload["timeline"]["name"] = clean_name
                    payload["saved_at"] = time.time()
                    tl_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
            if self.active_db and Path(self.active_db.db_path).resolve() == p.resolve():
                if self.active_project:
                    self.active_project.name = clean_name
                self._notify()
            logger.info(f"Proje yeniden adlandırıldı: {clean_name} ({p})")
            return True
        except Exception as exc:
            logger.error(f"Proje yeniden adlandırılamadı: {exc}")
            return False

    def duplicate_project(self, db_path: str | Path, new_name: Optional[str] = None) -> Optional[Path]:
        """Mevcut bir projenin tam kopyasını (WAL durumu dahil) oluşturur ve yeni .voxproj yolunu döndürür."""
        import sqlite3

        src_db = Path(db_path)
        if not src_db.exists():
            return None
        try:
            orig_db = ProjectDatabase(src_db)
            orig_proj = orig_db.get_project()
            base_title = (
                new_name.strip()
                if (new_name and new_name.strip())
                else f"{(orig_proj.name if orig_proj else src_db.parent.name)} (Kopya)"
            )
            safe_folder = (
                "".join(c for c in base_title if c.isalnum() or c in (" ", "_", "-")).strip()
                or "Proje_Kopya"
            )
            parent_root = src_db.parent.parent if src_db.parent.parent.exists() else PROJECTS_DIR
            dest_dir = parent_root / safe_folder
            counter = 1
            while dest_dir.exists():
                dest_dir = parent_root / f"{safe_folder}_{counter}"
                counter += 1
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_db = dest_dir / src_db.name

            import contextlib

            with contextlib.closing(sqlite3.connect(str(src_db))) as s_conn, contextlib.closing(
                sqlite3.connect(str(dest_db))
            ) as d_conn:
                s_conn.backup(d_conn)

            src_tl = src_db.with_suffix(".timeline.json")
            dest_tl = dest_db.with_suffix(".timeline.json")
            if src_tl.exists():
                shutil.copy2(src_tl, dest_tl)

            self.rename_project(dest_db, base_title)
            self.config.add_recent_project(str(dest_db))
            logger.info(f"Proje kopyası oluşturuldu: {dest_db}")
            return dest_db
        except Exception as exc:
            logger.error(f"Proje kopyalanamadı: {exc}")
            return None

    def delete_project(self, db_path: str | Path) -> bool:
        """Projeyi ve ilişkili proje dosyalarını siler, aktifse kapatır."""
        p = Path(db_path)
        try:
            if self.active_db and Path(self.active_db.db_path).resolve() == p.resolve():
                self.close_project()

            # Recent listesinden çıkar
            rec = [r for r in (self.config.get("recent_projects", []) or []) if Path(r).resolve() != p.resolve()]
            self.config.set("recent_projects", rec)

            parent_dir = p.parent
            for f in (
                p,
                p.with_suffix(".voxproj-wal"),
                p.with_suffix(".voxproj-shm"),
                p.with_suffix(".timeline.json"),
                p.with_suffix(".autosave.json"),
            ):
                if f.exists():
                    try:
                        f.unlink()
                    except Exception:
                        pass

            # Eğer proje klasörü PROJECTS_DIR altındaysa ve boşaldıysa klasörü de temizle
            try:
                if PROJECTS_DIR.exists() and parent_dir.resolve().parent == PROJECTS_DIR.resolve():
                    remaining_files = list(parent_dir.iterdir())
                    if not remaining_files:
                        parent_dir.rmdir()
            except Exception:
                pass
            logger.info(f"Proje silindi: {p}")
            return True
        except Exception as exc:
            logger.error(f"Proje silinemedi ({p}): {exc}")
            return False

    def autosave_path(self) -> Path:
        if self.current_file is not None:
            return self.current_file.with_suffix(".autosave.json")
        from core import paths

        return paths.cache_dir() / "autosave_workspace.json"

    def save_autosave_snapshot(
        self,
        timeline_dict: dict,
        media_bin: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """Beklenmedik kapanmalara karşı otomatik kurtarma yedeği yazar."""
        target = self.autosave_path()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "project_file": str(self.current_file) if self.current_file else None,
                "timeline": timeline_dict,
                "media_bin": list(media_bin or []),
            }
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)
            return target
        except Exception as exc:
            logger.warning(f"Otomatik kayıt yazılamadı: {exc}")
            return None

    @staticmethod
    def find_missing_media(timeline_dict: Optional[dict], media_bin: Optional[List[str]] = None) -> List[str]:
        """Timeline kliplerinde veya medya kutusunda bulunamayan (taşınmış/silinmiş) dosyaları tespit eder."""
        missing: List[str] = []
        seen = set()
        for p in media_bin or []:
            if p and p not in seen:
                seen.add(p)
                if not Path(p).exists():
                    missing.append(p)
        if isinstance(timeline_dict, dict):
            for tr in timeline_dict.get("tracks", []):
                for cl in tr.get("clips", []):
                    mp = cl.get("media_path")
                    if mp and mp not in seen:
                        seen.add(mp)
                        if not Path(mp).exists():
                            missing.append(mp)
        return missing

