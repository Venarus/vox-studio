"""
VOX Studio — Take Yönetim Servisi
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from core.logger import get_logger
from dsp.analysis import analyze_file
from project.database import ProjectDatabase
from project.models import Line, Take

logger = get_logger("takes.manager")


class TakeManager:
    """Replikler ve Take'ler arasındaki seslendirme iş akışını yönetir."""

    def __init__(self, db: ProjectDatabase):
        self.db = db

    def save_take_from_recording(
        self,
        line: Line,
        source_audio_path: str | Path,
        notes: str = "",
        move: bool = False,
    ) -> Take:
        source_path = Path(source_audio_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Kaynak kayıt dosyası bulunamadı: {source_path}")

        proj = self.db.get_project()
        if not proj or not proj.media_dir:
            raise RuntimeError("Proje medya klasörü mevcut değil")

        media_dir = Path(proj.media_dir)
        media_dir.mkdir(parents=True, exist_ok=True)

        # Önceden kayıtlı take sayısını bul
        existing_takes = self.db.list_takes(line.id)
        next_take_num = len(existing_takes) + 1

        # Güvenli hedef dosya adı: Line_{order:03d}_Take_{num:02d}.wav
        target_name = f"Line_{line.order_index:03d}_Take_{next_take_num:02d}.wav"
        target_path = media_dir / target_name

        if move:
            shutil.move(str(source_path), str(target_path))
        else:
            shutil.copy2(str(source_path), str(target_path))

        # DSP analizi yap (peak, rms, score)
        duration = 0.0
        peak_dbfs = -100.0
        rms_dbfs = -100.0
        score = None
        try:
            report = analyze_file(target_path)
            duration = report.duration
            peak_dbfs = report.peak_dbfs
            rms_dbfs = report.rms_dbfs
            score = report.quality_score
        except Exception as e:
            logger.warning(f"Take için ses analizi yapılamadı: {e}")

        take = self.db.add_take(
            line_id=line.id,
            audio_path=str(target_path),
            duration=duration,
            is_favorite=False,
            notes=notes,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            quality_score=score,
        )
        logger.info(f"Yeni take kaydedildi: {take.take_number} ({target_name})")
        return take

    def set_primary(self, line_id: str, take_id: str) -> None:
        self.db.set_primary_take(line_id, take_id)
        logger.info(f"Replik {line_id} için birincil take belirlendi: {take_id}")

    def toggle_favorite(self, take: Take) -> bool:
        new_state = not take.is_favorite
        self.db.set_take_favorite(take.id, new_state)
        take.is_favorite = new_state
        return new_state

    def remove_take(self, take_id: str, delete_file: bool = False) -> None:
        audio_path = self.db.delete_take(take_id)
        if delete_file and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Take dosyası diskten silindi: {audio_path}")
            except Exception as e:
                logger.warning(f"Dosya silinemedi: {e}")
