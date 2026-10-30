"""
VOX Studio — Klavye Kısayolları ve CapCut Eşleşme Rehberi Penceresi.

Kullanıcıların CapCut masaüstü ile doğrulanmış ve VOX Studio'ya özgü tüm
kısayolları görmesini, aramalarını ve gerektiğinde varsayılanlara sıfırlamasını sağlar.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.shortcuts import ShortcutAction, ShortcutManager
from ui import icons


class ShortcutGuideDialog(QDialog):
    """Klavye kısayolları ve CapCut uyumluluk rehberi penceresi."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.mgr = ShortcutManager.get_instance()
        self.setWindowTitle("VOX Studio — Klavye Kısayolları Rehberi")
        self.resize(760, 560)
        self.setMinimumSize(640, 440)
        self._current_filter_category = "all"
        self._build_ui()
        self._populate_table()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # Başlık ve Açıklama
        top_box = QVBoxLayout()
        top_box.setSpacing(3)
        title = QLabel("Klavye Kısayolları ve Hızlı İş Akışı")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #F5F5F6;")
        top_box.addWidget(title)

        desc = QLabel(
            "CapCut masaüstü ile doğrulanmış temel düzenleme kısayolları ve VOX Studio'ya özgü ses işleme kontrolleri. "
            "Metin kutularında yazarken harf kısayolları güvenli biçimde devre dışı bırakılır."
        )
        desc.setObjectName("MutedSmall")
        desc.setWordWrap(True)
        top_box.addWidget(desc)
        root.addLayout(top_box)

        # Filtre ve Arama Çubuğu
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(10)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Kısayol veya işlem ara (ör. Böl, Kes, Space, B, Sil)...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._on_search_changed)
        filter_bar.addWidget(self.search_edit, 1)

        # Kategori Butonları
        self.cat_group = QButtonGroup(self)
        cats = [
            ("all", "Tümü"),
            ("editing", "Düzenleme"),
            ("tools", "Araçlar"),
            ("playback", "Oynatma"),
            ("view", "Görünüm"),
            ("project", "Proje"),
        ]
        cat_box = QHBoxLayout()
        cat_box.setSpacing(4)
        for idx, (cid, clabel) in enumerate(cats):
            btn = QPushButton(clabel)
            btn.setProperty("compact", "true")
            btn.setCheckable(True)
            if idx == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda _=False, c=cid: self._on_category_changed(c))
            self.cat_group.addButton(btn, idx)
            cat_box.addWidget(btn)
        filter_bar.addLayout(cat_box)
        root.addLayout(filter_bar)

        # Kısayol Tablosu
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["İşlem", "Kısayol Tuşu", "Uyumluluk", "Açıklama"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet(
            "QTableWidget { background-color: #101114; border: 1px solid #28292E; border-radius: 8px; }"
            "QHeaderView::section { background-color: #151619; color: #A1A1AA; font-weight: 600; border: none; padding: 6px; }"
        )
        root.addWidget(self.table, 1)

        # Alt Butonlar
        bot_bar = QHBoxLayout()
        bot_bar.setSpacing(8)

        reset_btn = QPushButton("Varsayılan Kısayollara Sıfırla")
        reset_btn.setProperty("compact", "true")
        reset_btn.setToolTip("Tüm özel kısayolları fabrika ayarlarına geri döndürür")
        reset_btn.clicked.connect(self._reset_defaults)
        bot_bar.addWidget(reset_btn)

        bot_bar.addStretch(1)

        close_btn = QPushButton("Kapat")
        close_btn.setProperty("variant", "primary")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)
        bot_bar.addWidget(close_btn)

        root.addLayout(bot_bar)

    def _populate_table(self) -> None:
        query = self.search_edit.text().strip().lower()
        actions = self.mgr.all_actions()

        filtered: list[ShortcutAction] = []
        for act in actions:
            if self._current_filter_category != "all" and act.category != self._current_filter_category:
                continue
            if query:
                match = (
                    query in act.name.lower()
                    or query in act.current_key.lower()
                    or query in act.description.lower()
                )
                if not match:
                    continue
            filtered.append(act)

        self.table.setRowCount(len(filtered))
        for row, act in enumerate(filtered):
            # 1. İşlem Adı
            item_name = QTableWidgetItem(f"  {act.name}")
            item_name.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            self.table.setItem(row, 0, item_name)

            # 2. Kısayol Rozeti
            item_key = QTableWidgetItem(f" {act.current_key} ")
            item_key.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_key.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            self.table.setItem(row, 1, item_key)

            # 3. Uyumluluk
            comp_txt = "✓ CapCut Standardı" if act.is_capcut_standard else "VOX Studio"
            item_comp = QTableWidgetItem(comp_txt)
            item_comp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_comp.setForeground(Qt.GlobalColor.green if act.is_capcut_standard else Qt.GlobalColor.cyan)
            self.table.setItem(row, 2, item_comp)

            # 4. Açıklama
            item_desc = QTableWidgetItem(act.description)
            self.table.setItem(row, 3, item_desc)

    def _on_search_changed(self, text: str) -> None:
        self._populate_table()

    def _on_category_changed(self, category: str) -> None:
        self._current_filter_category = category
        self._populate_table()

    def _reset_defaults(self) -> None:
        self.mgr.reset_to_default()
        self._populate_table()
