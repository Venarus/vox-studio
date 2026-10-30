"""
VOX Studio — Merkezi Klavye Kısayolu Yöneticisi (CapCut Uyumlu ve Özelleştirilebilir).

Bu modül:
- CapCut masaüstü düzenleyicisiyle doğrulanmış standart kısayolları tek bir merkezde tanımlar.
- Metin giriş alanlarında (QLineEdit, QTextEdit, SpinBox vb.) ve modal pencerelerde
  harf tuşlarının (B, V, M, Z vb.) düzenleme aracını tetiklemesini engeller.
- Çakışma tespiti (conflict detection) ve varsayılan ayarlara geri dönme özelliği sunar.
- Kısayol özelleştirmelerini ConfigManager aracılığıyla kalıcı olarak saklar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

from core.logger import get_logger

logger = get_logger("core.shortcuts")


@dataclass
class ShortcutAction:
    action_id: str
    name: str
    category: str  # "editing" | "playback" | "tools" | "view" | "project"
    default_key: str
    current_key: str
    description: str
    is_capcut_standard: bool
    context: str = "global"  # "global" | "editor" | "timeline"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "name": self.name,
            "category": self.category,
            "default_key": self.default_key,
            "current_key": self.current_key,
            "description": self.description,
            "is_capcut_standard": self.is_capcut_standard,
            "context": self.context,
        }


# Doğrulanmış CapCut masaüstü standartları ve VOX Studio özgün eşleşmeleri
DEFAULT_SHORTCUTS: List[ShortcutAction] = [
    # 1. Düzenleme & Zaman Çizelgesi
    ShortcutAction(
        action_id="undo",
        name="Geri Al",
        category="editing",
        default_key="Ctrl+Z",
        current_key="Ctrl+Z",
        description="Son işlemi geri alır.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="redo",
        name="Yinele",
        category="editing",
        default_key="Ctrl+Y",
        current_key="Ctrl+Y",
        description="Geri alınan işlemi yineler.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="split_playhead",
        name="İmleçte Böl",
        category="editing",
        default_key="Ctrl+K",
        current_key="Ctrl+K",
        description="Seçili klibi oynatma kafası (playhead) konumundan ikiye böler.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="delete",
        name="Klibi Sil",
        category="editing",
        default_key="Delete",
        current_key="Delete",
        description="Seçili klibi zaman çizelgesinden kaldırır.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="ripple_delete",
        name="Boşluğu Kapatarak Sil (Ripple Delete)",
        category="editing",
        default_key="Shift+Delete",
        current_key="Shift+Delete",
        description="Seçili klibi siler ve ardışık klipleri öne çekerek aradaki boşluğu kapatır.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="duplicate",
        name="Çoğalt",
        category="editing",
        default_key="Ctrl+D",
        current_key="Ctrl+D",
        description="Seçili klibi hemen sonrasına kopyalayarak çoğaltır.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="copy",
        name="Kopyala",
        category="editing",
        default_key="Ctrl+C",
        current_key="Ctrl+C",
        description="Seçili klibi panoya kopyalar.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="paste",
        name="Yapıştır",
        category="editing",
        default_key="Ctrl+V",
        current_key="Ctrl+V",
        description="Panodaki klibi playhead konumuna yapıştırır.",
        is_capcut_standard=True,
    ),

    # 2. Düzenleme Araçları
    ShortcutAction(
        action_id="tool_select",
        name="Seçim Aracı (Select)",
        category="tools",
        default_key="V",
        current_key="V",
        description="Normal seçim ve taşıma modunu etkinleştirir.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="tool_blade",
        name="Jilet / Kesme Aracı (Blade)",
        category="tools",
        default_key="B",
        current_key="B",
        description="Kliplerin üzerine tıklayarak bölmenizi sağlayan kesme modunu etkinleştirir.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="add_marker",
        name="İşaretçi Ekle",
        category="tools",
        default_key="M",
        current_key="M",
        description="Oynatma kafasının bulunduğu zamana işaretçi (marker) ekler.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="toggle_magnetic",
        name="Mıknatıslı Hizalama (Snap)",
        category="tools",
        default_key="N",
        current_key="N",
        description="Klip kenarlarına ve imlece mıknatıslı yapışmayı açar/kapatır.",
        is_capcut_standard=True,
    ),

    # 3. Oynatma ve Gezinme
    ShortcutAction(
        action_id="play_pause",
        name="Oynat / Duraklat",
        category="playback",
        default_key="Space",
        current_key="Space",
        description="Önizleme oynatımını başlatır veya duraklatır.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="step_prev_frame",
        name="1 Kare Geri",
        category="playback",
        default_key="Left",
        current_key="Left",
        description="Playhead'i 1 kare geriye taşır.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="step_next_frame",
        name="1 Kare İleri",
        category="playback",
        default_key="Right",
        current_key="Right",
        description="Playhead'i 1 kare ileriye taşır.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="step_prev_sec",
        name="1 Saniye Geri",
        category="playback",
        default_key="Shift+Left",
        current_key="Shift+Left",
        description="Playhead'i 1 saniye geriye taşır.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="step_next_sec",
        name="1 Saniye İleri",
        category="playback",
        default_key="Shift+Right",
        current_key="Shift+Right",
        description="Playhead'i 1 saniye ileriye taşır.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="jump_home",
        name="Başa Git",
        category="playback",
        default_key="Home",
        current_key="Home",
        description="Playhead'i zaman çizelgesinin başlangıcına (00:00) götürür.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="jump_end",
        name="Sona Git",
        category="playback",
        default_key="End",
        current_key="End",
        description="Playhead'i son klibin bitiş zamanına götürür.",
        is_capcut_standard=True,
    ),

    # 4. Görünüm ve Yakınlaştırma
    ShortcutAction(
        action_id="zoom_in",
        name="Yakınlaştır (Zoom In)",
        category="view",
        default_key="Ctrl+=",
        current_key="Ctrl+=",
        description="Zaman çizelgesini yatayda yakınlaştırır (Ctrl + Fare Tekerleği).",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="zoom_out",
        name="Uzaklaştır (Zoom Out)",
        category="view",
        default_key="Ctrl+-",
        current_key="Ctrl+-",
        description="Zaman çizelgesini yatayda uzaklaştırır.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="fit_project",
        name="Projeyi Ekrana Sığdır",
        category="view",
        default_key="Shift+Z",
        current_key="Shift+Z",
        description="Tüm klipleri tek ekranda gösterecek ölçeğe getirir.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="focus_clip",
        name="Seçili Klibe Odaklan",
        category="view",
        default_key="Z",
        current_key="Z",
        description="Seçili klibi zaman çizelgesinin ortasına alarak yakınlaştırır.",
        is_capcut_standard=True,
    ),
    ShortcutAction(
        action_id="toggle_fullscreen",
        name="Tam Ekran Önizleme",
        category="view",
        default_key="F11",
        current_key="F11",
        description="Video oynatıcıyı tam ekranda açar/kapatır.",
        is_capcut_standard=True,
    ),

    # 5. Proje ve Dışa Aktarma
    ShortcutAction(
        action_id="save_project",
        name="Projeyi Kaydet",
        category="project",
        default_key="Ctrl+S",
        current_key="Ctrl+S",
        description="Mevcut zaman çizelgesi projesini ve ayarlarını kaydeder.",
        is_capcut_standard=False,
    ),
    ShortcutAction(
        action_id="export_project",
        name="Medyayı Dışa Aktar",
        category="project",
        default_key="Ctrl+E",
        current_key="Ctrl+E",
        description="Tamamlanan videoyu veya ses miksajını dışa aktarma penceresini açar.",
        is_capcut_standard=False,
    ),
    ShortcutAction(
        action_id="record_voiceover",
        name="Canlı Dublaj Kaydı",
        category="project",
        default_key="Ctrl+R",
        current_key="Ctrl+R",
        description="İmleç konumundan itibaren mikrofonla ses kaydı başlatır/durdurur.",
        is_capcut_standard=False,
    ),
    ShortcutAction(
        action_id="show_shortcuts_help",
        name="Klavye Kısayolları Rehberi",
        category="project",
        default_key="F1",
        current_key="F1",
        description="CapCut ve VOX Studio klavye kısayolları listesini gösterir.",
        is_capcut_standard=False,
    ),
]


class ShortcutManager:
    """Tekil kısayol yönetim sistemi."""

    _instance: Optional["ShortcutManager"] = None

    def __init__(self, custom_mapping: Optional[Dict[str, str]] = None):
        self._actions: Dict[str, ShortcutAction] = {}
        for item in DEFAULT_SHORTCUTS:
            act = ShortcutAction(
                action_id=item.action_id,
                name=item.name,
                category=item.category,
                default_key=item.default_key,
                current_key=item.default_key,
                description=item.description,
                is_capcut_standard=item.is_capcut_standard,
                context=item.context,
            )
            self._actions[item.action_id] = act

        if custom_mapping:
            self.load_custom_mapping(custom_mapping)

    @classmethod
    def get_instance(cls) -> "ShortcutManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def is_text_input_focused() -> bool:
        """
        Kullanıcı bir metin giriş kutusunda (QLineEdit, QTextEdit vb.)
        yazı yazarken veya aktif bir modal diyalog açıkken tek harfli
        kısayolların çalışmasını engelleyen güvenlik filtresi.
        """
        if QApplication.activeModalWidget() is not None:
            return True
        fw = QApplication.focusWidget()
        if fw is None:
            return False
        from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QLineEdit, QPlainTextEdit
        if isinstance(fw, (QLineEdit, QPlainTextEdit, QComboBox, QAbstractSpinBox)):
            return True
        if fw.inherits("QTextEdit") or fw.inherits("QLineEdit"):
            return True
        return False

    def get_key(self, action_id: str) -> str:
        act = self._actions.get(action_id)
        return act.current_key if act else ""

    def get_action(self, action_id: str) -> Optional[ShortcutAction]:
        return self._actions.get(action_id)

    def all_actions(self) -> List[ShortcutAction]:
        return list(self._actions.values())

    def actions_by_category(self) -> Dict[str, List[ShortcutAction]]:
        grouped: Dict[str, List[ShortcutAction]] = {
            "editing": [],
            "playback": [],
            "tools": [],
            "view": [],
            "project": [],
        }
        for act in self._actions.values():
            grouped.setdefault(act.category, []).append(act)
        return grouped

    def find_conflict(self, key_sequence_str: str, exclude_action_id: Optional[str] = None) -> Optional[ShortcutAction]:
        normalized = self._normalize_key(key_sequence_str)
        for act_id, act in self._actions.items():
            if exclude_action_id and act_id == exclude_action_id:
                continue
            if self._normalize_key(act.current_key) == normalized:
                return act
        return None

    def set_key(self, action_id: str, new_key_str: str) -> Tuple[bool, str]:
        """Kısayolu günceller, çakışma kontrolü yapar."""
        if action_id not in self._actions:
            return False, f"Bilinmeyen eylem: {action_id}"

        clean_key = self._normalize_key(new_key_str)
        if not clean_key:
            return False, "Geçersiz kısayol tuşu."

        conflict = self.find_conflict(clean_key, exclude_action_id=action_id)
        if conflict:
            return False, f"Bu kısayol zaten '{conflict.name}' için tanımlı ({conflict.current_key})."

        self._actions[action_id].current_key = clean_key
        logger.info(f"Kısayol güncellendi: {action_id} -> {clean_key}")
        return True, "Başarılı"

    def reset_to_default(self, action_id: Optional[str] = None) -> None:
        if action_id:
            if action_id in self._actions:
                self._actions[action_id].current_key = self._actions[action_id].default_key
        else:
            for act in self._actions.values():
                act.current_key = act.default_key
        logger.info(f"Kısayollar varsayılana sıfırlandı: action_id={action_id}")

    def load_custom_mapping(self, mapping: Dict[str, str]) -> None:
        for act_id, key in mapping.items():
            if act_id in self._actions and key:
                self._actions[act_id].current_key = self._normalize_key(str(key))

    def export_custom_mapping(self) -> Dict[str, str]:
        diff = {}
        for act in self._actions.values():
            if act.current_key != act.default_key:
                diff[act.action_id] = act.current_key
        return diff

    @staticmethod
    def _normalize_key(key_str: str) -> str:
        s = (key_str or "").strip()
        # QKeySequence ile standart gösterime çevir
        try:
            seq = QKeySequence(s)
            res = seq.toString(QKeySequence.SequenceFormat.PortableText)
            return res or s
        except Exception:
            return s
