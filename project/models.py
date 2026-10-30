"""
VOX Studio — Proje ve Take Veri Modelleri
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    LINE_STATUS_PENDING,
)


@dataclass
class Project:
    id: str
    name: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    db_path: Optional[str] = None
    media_dir: Optional[str] = None
    video_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "db_path": self.db_path,
            "media_dir": self.media_dir,
            "video_path": self.video_path,
        }


@dataclass
class Character:
    id: str
    project_id: str
    name: str
    color: str = "#9E2A2B"
    description: str = ""
    voice_actor: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class Scene:
    id: str
    project_id: str
    name: str
    order_index: int = 0
    description: str = ""
    timecode_start: float = 0.0
    timecode_end: float = 0.0


@dataclass
class Line:
    id: str
    project_id: str
    character_id: Optional[str] = None
    scene_id: Optional[str] = None
    text: str = ""
    cue_in: float = 0.0
    cue_out: float = 0.0
    status: str = LINE_STATUS_PENDING
    notes: str = ""
    emotion: str = ""
    order_index: int = 0
    primary_take_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class Take:
    id: str
    line_id: str
    take_number: int
    audio_path: str
    duration: float = 0.0
    is_favorite: bool = False
    is_primary: bool = False
    notes: str = ""
    peak_dbfs: float = -100.0
    rms_dbfs: float = -100.0
    quality_score: Optional[int] = None
    created_at: float = field(default_factory=time.time)



