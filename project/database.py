"""
VOX Studio — SQLite Proje Veritabanı Yöneticisi
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.logger import get_logger
from project.models import Character, Line, Project, Scene, Take

logger = get_logger("project.db")

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    sample_rate INTEGER DEFAULT 48000,
    channels INTEGER DEFAULT 1,
    media_dir TEXT,
    video_path TEXT
);

CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    color TEXT DEFAULT '#9E2A2B',
    description TEXT DEFAULT '',
    voice_actor TEXT DEFAULT '',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS scenes (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    order_index INTEGER DEFAULT 0,
    description TEXT DEFAULT '',
    timecode_start REAL DEFAULT 0.0,
    timecode_end REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS lines (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    character_id TEXT REFERENCES characters(id) ON DELETE SET NULL,
    scene_id TEXT REFERENCES scenes(id) ON DELETE SET NULL,
    text TEXT NOT NULL,
    cue_in REAL DEFAULT 0.0,
    cue_out REAL DEFAULT 0.0,
    status TEXT DEFAULT 'pending',
    notes TEXT DEFAULT '',
    emotion TEXT DEFAULT '',
    order_index INTEGER DEFAULT 0,
    primary_take_id TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS takes (
    id TEXT PRIMARY KEY,
    line_id TEXT NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
    take_number INTEGER NOT NULL,
    audio_path TEXT NOT NULL,
    duration REAL DEFAULT 0.0,
    is_favorite INTEGER DEFAULT 0,
    is_primary INTEGER DEFAULT 0,
    notes TEXT DEFAULT '',
    peak_dbfs REAL DEFAULT -100.0,
    rms_dbfs REAL DEFAULT -100.0,
    quality_score INTEGER,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lines_project ON lines(project_id);
CREATE INDEX IF NOT EXISTS idx_takes_line ON takes(line_id);
"""


class ProjectDatabase:
    """Tekil bir .voxproj (SQLite) dosyasına erişim sağlayan veritabanı sürücüsü."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextlib.contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.debug(f"Veritabanı hazır: {self.db_path}")

    # ----------------------------------------------------------- Proje İşlemleri
    def create_project(
        self,
        name: str,
        description: str = "",
        sample_rate: int = 48000,
        channels: int = 1,
        video_path: Optional[str] = None,
    ) -> Project:
        proj_id = str(uuid.uuid4())
        now = time.time()
        media_dir = str(self.db_path.parent / "audio")
        Path(media_dir).mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO projects (id, name, description, created_at, updated_at, sample_rate, channels, media_dir, video_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (proj_id, name, description, now, now, sample_rate, channels, media_dir, video_path),
            )
        return Project(
            id=proj_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            sample_rate=sample_rate,
            channels=channels,
            db_path=str(self.db_path),
            media_dir=media_dir,
            video_path=video_path,
        )

    def get_project(self) -> Optional[Project]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM projects LIMIT 1")
            row = cur.fetchone()
            if not row:
                return None
            return Project(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                sample_rate=row["sample_rate"],
                channels=row["channels"],
                db_path=str(self.db_path),
                media_dir=row["media_dir"],
                video_path=row["video_path"],
            )

    def update_project(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> None:
        proj = self.get_project()
        if not proj:
            return
        fields = []
        vals = []
        if name is not None:
            fields.append("name = ?")
            vals.append(name)
        if description is not None:
            fields.append("description = ?")
            vals.append(description)
        if video_path is not None:
            fields.append("video_path = ?")
            vals.append(video_path)

        fields.append("updated_at = ?")
        vals.append(time.time())
        vals.append(proj.id)

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE projects SET {', '.join(fields)} WHERE id = ?",
                vals,
            )

    # --------------------------------------------------------- Karakter İşlemleri
    def add_character(
        self,
        name: str,
        color: str = "#9E2A2B",
        description: str = "",
        voice_actor: str = "",
    ) -> Character:
        proj = self.get_project()
        if not proj:
            raise RuntimeError("Proje bulunamadı")
        char_id = str(uuid.uuid4())
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO characters (id, project_id, name, color, description, voice_actor, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (char_id, proj.id, name, color, description, voice_actor, now),
            )
        return Character(
            id=char_id,
            project_id=proj.id,
            name=name,
            color=color,
            description=description,
            voice_actor=voice_actor,
            created_at=now,
        )

    def list_characters(self) -> List[Character]:
        proj = self.get_project()
        if not proj:
            return []
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM characters WHERE project_id = ? ORDER BY name COLLATE NOCASE",
                (proj.id,),
            )
            return [
                Character(
                    id=r["id"],
                    project_id=r["project_id"],
                    name=r["name"],
                    color=r["color"],
                    description=r["description"],
                    voice_actor=r["voice_actor"],
                    created_at=r["created_at"],
                )
                for r in cur.fetchall()
            ]

    # ------------------------------------------------------------ Sahne İşlemleri
    def add_scene(
        self,
        name: str,
        order_index: int = 0,
        description: str = "",
        timecode_start: float = 0.0,
        timecode_end: float = 0.0,
    ) -> Scene:
        proj = self.get_project()
        if not proj:
            raise RuntimeError("Proje bulunamadı")
        scene_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO scenes (id, project_id, name, order_index, description, timecode_start, timecode_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (scene_id, proj.id, name, order_index, description, timecode_start, timecode_end),
            )
        return Scene(
            id=scene_id,
            project_id=proj.id,
            name=name,
            order_index=order_index,
            description=description,
            timecode_start=timecode_start,
            timecode_end=timecode_end,
        )

    def list_scenes(self) -> List[Scene]:
        proj = self.get_project()
        if not proj:
            return []
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM scenes WHERE project_id = ? ORDER BY order_index, name",
                (proj.id,),
            )
            return [
                Scene(
                    id=r["id"],
                    project_id=r["project_id"],
                    name=r["name"],
                    order_index=r["order_index"],
                    description=r["description"],
                    timecode_start=r["timecode_start"],
                    timecode_end=r["timecode_end"],
                )
                for r in cur.fetchall()
            ]

    # ----------------------------------------------------------- Replik İşlemleri
    def add_line(
        self,
        text: str,
        character_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        cue_in: float = 0.0,
        cue_out: float = 0.0,
        status: str = "pending",
        notes: str = "",
        emotion: str = "",
        order_index: int = 0,
    ) -> Line:
        proj = self.get_project()
        if not proj:
            raise RuntimeError("Proje bulunamadı")
        line_id = str(uuid.uuid4())
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO lines (id, project_id, character_id, scene_id, text, cue_in, cue_out, status, notes, emotion, order_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (line_id, proj.id, character_id, scene_id, text, cue_in, cue_out, status, notes, emotion, order_index, now),
            )
        return Line(
            id=line_id,
            project_id=proj.id,
            character_id=character_id,
            scene_id=scene_id,
            text=text,
            cue_in=cue_in,
            cue_out=cue_out,
            status=status,
            notes=notes,
            emotion=emotion,
            order_index=order_index,
            created_at=now,
        )

    def list_lines(self, scene_id: Optional[str] = None, character_id: Optional[str] = None) -> List[Line]:
        proj = self.get_project()
        if not proj:
            return []
        query = "SELECT * FROM lines WHERE project_id = ?"
        params: List[Any] = [proj.id]
        if scene_id is not None:
            query += " AND scene_id = ?"
            params.append(scene_id)
        if character_id is not None:
            query += " AND character_id = ?"
            params.append(character_id)
        query += " ORDER BY order_index, cue_in, created_at"

        with self._get_connection() as conn:
            cur = conn.execute(query, params)
            return [
                Line(
                    id=r["id"],
                    project_id=r["project_id"],
                    character_id=r["character_id"],
                    scene_id=r["scene_id"],
                    text=r["text"],
                    cue_in=r["cue_in"],
                    cue_out=r["cue_out"],
                    status=r["status"],
                    notes=r["notes"],
                    emotion=r["emotion"],
                    order_index=r["order_index"],
                    primary_take_id=r["primary_take_id"],
                    created_at=r["created_at"],
                )
                for r in cur.fetchall()
            ]

    def update_line_status(self, line_id: str, status: str) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE lines SET status = ? WHERE id = ?", (status, line_id))

    def set_primary_take(self, line_id: str, take_id: str) -> None:
        with self._get_connection() as conn:
            # Önce bu repliğin tüm take'lerini birincil olmaktan çıkar
            conn.execute("UPDATE takes SET is_primary = 0 WHERE line_id = ?", (line_id,))
            # Seçileni birincil yap
            conn.execute("UPDATE takes SET is_primary = 1 WHERE id = ?", (take_id,))
            # Replik tablosundaki işaretçiyi güncelle
            conn.execute("UPDATE lines SET primary_take_id = ?, status = 'recorded' WHERE id = ?", (take_id, line_id))

    # ------------------------------------------------------------- Take İşlemleri
    def add_take(
        self,
        line_id: str,
        audio_path: str,
        duration: float = 0.0,
        is_favorite: bool = False,
        notes: str = "",
        peak_dbfs: float = -100.0,
        rms_dbfs: float = -100.0,
        quality_score: Optional[int] = None,
    ) -> Take:
        take_id = str(uuid.uuid4())
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COALESCE(MAX(take_number), 0) + 1 AS next_num FROM takes WHERE line_id = ?", (line_id,))
            next_num = cur.fetchone()["next_num"]

            # Eğer bu ilk take ise varsayılan olarak birincil yap
            is_first = (next_num == 1)
            is_primary = 1 if is_first else 0

            conn.execute(
                """
                INSERT INTO takes (id, line_id, take_number, audio_path, duration, is_favorite, is_primary, notes, peak_dbfs, rms_dbfs, quality_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (take_id, line_id, next_num, audio_path, duration, int(is_favorite), is_primary, notes, peak_dbfs, rms_dbfs, quality_score, now),
            )
            if is_first:
                conn.execute("UPDATE lines SET primary_take_id = ?, status = 'recorded' WHERE id = ?", (take_id, line_id))

        return Take(
            id=take_id,
            line_id=line_id,
            take_number=next_num,
            audio_path=audio_path,
            duration=duration,
            is_favorite=is_favorite,
            is_primary=bool(is_primary),
            notes=notes,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            quality_score=quality_score,
            created_at=now,
        )

    def list_takes(self, line_id: str) -> List[Take]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM takes WHERE line_id = ? ORDER BY take_number",
                (line_id,),
            )
            return [
                Take(
                    id=r["id"],
                    line_id=r["line_id"],
                    take_number=r["take_number"],
                    audio_path=r["audio_path"],
                    duration=r["duration"],
                    is_favorite=bool(r["is_favorite"]),
                    is_primary=bool(r["is_primary"]),
                    notes=r["notes"],
                    peak_dbfs=r["peak_dbfs"],
                    rms_dbfs=r["rms_dbfs"],
                    quality_score=r["quality_score"],
                    created_at=r["created_at"],
                )
                for r in cur.fetchall()
            ]

    def set_take_favorite(self, take_id: str, is_favorite: bool) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE takes SET is_favorite = ? WHERE id = ?", (int(is_favorite), take_id))

    def delete_take(self, take_id: str) -> Optional[str]:
        """Take kaydını siler ve ilişkili ses dosyasının yolunu döner (silinmesi için)."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT audio_path, line_id, is_primary FROM takes WHERE id = ?", (take_id,))
            row = cur.fetchone()
            if not row:
                return None
            audio_path = row["audio_path"]
            line_id = row["line_id"]
            was_primary = bool(row["is_primary"])

            conn.execute("DELETE FROM takes WHERE id = ?", (take_id,))

            # Eğer silinen take birincil ise, varsa en son take'i yeni birincil yap
            if was_primary:
                cur2 = conn.execute("SELECT id FROM takes WHERE line_id = ? ORDER BY take_number DESC LIMIT 1", (line_id,))
                new_p = cur2.fetchone()
                if new_p:
                    new_id = new_p["id"]
                    conn.execute("UPDATE takes SET is_primary = 1 WHERE id = ?", (new_id,))
                    conn.execute("UPDATE lines SET primary_take_id = ? WHERE id = ?", (new_id, line_id))
                else:
                    conn.execute("UPDATE lines SET primary_take_id = NULL, status = 'pending' WHERE id = ?", (line_id,))

            return audio_path

    # -------------------------------------------------------------- Yedekleme
    def backup(self, dest_path: str | Path) -> None:
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            bck = sqlite3.connect(str(dest_path))
            with bck:
                conn.backup(bck)
            bck.close()
        logger.info(f"Veritabanı yedeği alındı: {dest_path}")
