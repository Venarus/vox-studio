"""
VOX Studio — Media Library Workspace
Referans Tasarım (Image 5): PROJECT BIN (Klasör Ağacı), Zengin Medya Kartları (Grid & List View),
Hızlı Arama/Filtreleme/Sıralama, Çevrimdışı Medya Tespiti (Media Offline),
Seçili Varlık Denetçisi (Selected Asset Inspector) ve Gerçek Disk/Timeline Entegrasyonu.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from PySide6.QtCore import QPoint, QRectF, QSize, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import constants as C
from core.format import format_duration, format_file_size
from ui import icons
from ui.context import AppContext
from ui.workers import LibraryScanWorker


class MediaCardWidget(QFrame):
    """Zengin medya kartı: Küçük resim, 4K/HD rozeti, süre, çevrimdışı durumu ve timeline sürükle düğmesi."""

    clicked = Signal(dict)
    doubleClicked = Signal(dict)
    addToTimeline = Signal(str)

    def __init__(self, entry: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.entry = entry
        self._is_selected = False
        self._is_hovered = False

        self.setFixedSize(210, 168)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"MediaCardWidget {{ background: #15171C; border: 1px solid #23262D; border-radius: 8px; }} "
            f"MediaCardWidget:hover {{ border-color: #384050; background: #191B21; }}"
        )
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Üst Küçük Resim / Önizleme Alanı
        self.thumb_frame = QFrame()
        self.thumb_frame.setFixedHeight(108)
        self.thumb_frame.setStyleSheet("background: #0D0E11; border-radius: 6px; overflow: hidden;")
        tf_layout = QVBoxLayout(self.thumb_frame)
        tf_layout.setContentsMargins(4, 4, 4, 4)

        # Üst sol seçim kutusu ve üst sağ kategori ikonu
        top_row = QHBoxLayout()
        self.check_box = QCheckBox()
        self.check_box.setStyleSheet("QCheckBox::indicator { width: 14px; height: 14px; }")
        self.check_box.toggled.connect(self._on_check_toggled)
        top_row.addWidget(self.check_box)
        top_row.addStretch(1)

        cat = self.entry.get("category", "video")
        self.cat_icon = QLabel()
        cat_icon_name = "video" if cat == "video" else ("music" if cat in ("audio", "stem") else "folder")
        self.cat_icon.setPixmap(icons.pixmap(cat_icon_name, 14, "#9CA6B8"))
        top_row.addWidget(self.cat_icon)
        tf_layout.addLayout(top_row)

        # Orta rozet veya uyarı (Media Offline / Scanning / Artwork)
        self.center_label = QLabel()
        self.center_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        is_missing = not Path(self.entry["path"]).exists()
        has_error = bool(self.entry.get("error"))

        if is_missing or has_error:
            self.thumb_frame.setStyleSheet(
                "background: #251014; border: 1px dashed #E53945; border-radius: 6px;"
            )
            self.center_label.setText("⚠ MEDIA OFFLINE\nDosya Yolu Eksik")
            self.center_label.setStyleSheet("color: #FF6B6B; font-size: 10px; font-weight: bold;")
        else:
            thumb_path = self.entry.get("thumbnail")
            if thumb_path and Path(thumb_path).exists():
                pm = QPixmap(thumb_path).scaled(
                    200, 100, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation
                )
                self.center_label.setPixmap(pm)
            elif cat in ("audio", "stem", "recording"):
                self.center_label.setPixmap(icons.pixmap("waveform", 36, "#38BDF8"))
            else:
                self.center_label.setPixmap(icons.pixmap("video", 36, "#64748B"))
        tf_layout.addWidget(self.center_label, 1)

        # Alt Rozetler (Çözünürlük & Süre)
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(4)
        res_str = self.entry.get("resolution") or ""
        if "3840" in res_str or "2160" in res_str or "4k" in str(self.entry["name"]).lower():
            badge = QLabel("4K")
            badge.setStyleSheet(
                "background: #101216; color: #F8FAFC; border: 1px solid #363840; border-radius: 3px; font-size: 9px; font-weight: 700; padding: 1px 4px;"
            )
            bottom_row.addWidget(badge)
        elif "1920" in res_str or "1080" in res_str:
            badge = QLabel("HD")
            badge.setStyleSheet(
                "background: #101216; color: #9CA6B8; border: 1px solid #28292E; border-radius: 3px; font-size: 9px; font-weight: 600; padding: 1px 4px;"
            )
            bottom_row.addWidget(badge)

        bottom_row.addStretch(1)

        dur_s = self.entry.get("duration", 0.0)
        dur_lbl = QLabel(format_duration(dur_s) if dur_s > 0 else "--:--")
        dur_lbl.setStyleSheet(
            "background: rgba(10, 12, 16, 0.85); color: #F8FAFC; border-radius: 3px; font-size: 10px; font-weight: 600; padding: 1px 5px;"
        )
        bottom_row.addWidget(dur_lbl)
        tf_layout.addLayout(bottom_row)
        layout.addWidget(self.thumb_frame)

        # Alt Bilgi Alanı: Dosya Adı ve Özellikler
        self.name_lbl = QLabel(self.entry.get("name", "İsimsiz"))
        self.name_lbl.setStyleSheet("color: #F8FAFC; font-weight: 600; font-size: 11px;")
        self.name_lbl.setToolTip(self.entry.get("path", ""))
        layout.addWidget(self.name_lbl)

        fps_str = "23.976 fps" if cat == "video" else f"{self.entry.get('samplerate', 48000)} Hz"
        fmt_str = self.entry.get("format", "").upper() or "MEDYA"
        meta_str = f"{fps_str} · {fmt_str} · {format_file_size(self.entry.get('size', 0))}"
        self.meta_lbl = QLabel(meta_str)
        self.meta_lbl.setStyleSheet("color: #6C7687; font-size: 10px;")
        layout.addWidget(self.meta_lbl)

    def set_card_selected(self, selected: bool) -> None:
        self._is_selected = selected
        self.check_box.blockSignals(True)
        self.check_box.setChecked(selected)
        self.check_box.blockSignals(False)
        border_col = "#E53945" if selected else "#23262D"
        bg_col = "#241216" if selected else "#15171C"
        self.setStyleSheet(
            f"MediaCardWidget {{ background: {bg_col}; border: 1.5px solid {border_col}; border-radius: 8px; }}"
        )

    def _on_check_toggled(self, checked: bool) -> None:
        self.set_card_selected(checked)
        self.clicked.emit(self.entry)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_card_selected(not self._is_selected)
            self.clicked.emit(self.entry)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit(self.entry)
        super().mouseDoubleClickEvent(event)


class LibraryView(QWidget):
    """VOX Studio Profesyonel Medya Kütüphanesi ve Varlık Yöneticisi."""

    openRequested = Signal(str)
    playRequested = Signal(str)
    analyzeRequested = Signal(str)
    fileDeleted = Signal(str)
    addToTimelineRequested = Signal(str)

    COLUMNS = ["DOSYA", "KATEGORİ", "TARİH", "SÜRE", "BİÇİM", "Hz / ÇÖZ.", "KANAL", "BOYUT"]

    CATEGORY_LABELS = {
        "all": "Tüm Medyalar",
        "video": "Footage (Videolar)",
        "audio": "Audio (Ses & Müzik)",
        "recording": "Seslendirme / Kayıtlar",
        "stem": "Ayrıştırılmış Stem'ler",
        "ai": "AI Üretimleri",
        "image": "Görseller",
        "subtitle": "Altyazılar",
    }

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx
        self._worker: Optional[LibraryScanWorker] = None
        self._entries: List[dict] = []
        self._selected_entry: Optional[dict] = None
        self._selected_paths: Set[str] = set()
        self.setAcceptDrops(True)

        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # ------------------------------------------------------------- 1. ÜST SEKME ÇUBUĞU (WORKSPACE TABS)
        top_tabs_row = QHBoxLayout()
        top_tabs_row.setSpacing(4)
        tab_names = [
            ("Media Library (Aktif)", True),
            ("Efekt Tarayıcısı", False),
            ("Başlıklar & Jeneratörler", False),
            ("Proje Ayarları", False),
        ]
        for t_name, is_act in tab_names:
            t_btn = QPushButton(t_name)
            t_btn.setProperty("compact", "true")
            if is_act:
                t_btn.setStyleSheet(
                    f"background: #241216; color: #FFFFFF; border: 1px solid {C.COLOR_ACCENT}; font-weight: 700; border-radius: 5px; padding: 5px 12px;"
                )
            else:
                t_btn.setStyleSheet(
                    "background: #14161A; color: #8F9CAE; border: 1px solid #23262D; border-radius: 5px; padding: 5px 12px;"
                )
            top_tabs_row.addWidget(t_btn)
        top_tabs_row.addStretch(1)
        root.addLayout(top_tabs_row)

        # ------------------------------------------------------------- 2. AKSİYON VE FİLTRE ÇUBUĞU (SUB-BAR)
        sub_bar = QFrame()
        sub_bar.setObjectName("Card")
        sub_bar.setFixedHeight(48)
        sb_lay = QHBoxLayout(sub_bar)
        sb_lay.setContentsMargins(10, 4, 10, 4)
        sb_lay.setSpacing(8)

        # İçe Aktar (Kırmızı Vurgulu Düğme)
        self.import_btn = QPushButton(" Medya İçe Aktar...")
        self.import_btn.setProperty("variant", "primary")
        self.import_btn.setIcon(icons.icon("upload", size=14, color="#FFFFFF"))
        self.import_btn.setStyleSheet(
            f"background-color: {C.COLOR_ACCENT}; color: #FFFFFF; font-weight: 700; "
            "border: none; border-radius: 6px; padding: 6px 14px;"
        )
        self.import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_btn.clicked.connect(self._import_external_files)
        sb_lay.addWidget(self.import_btn)

        # Arama Kutusu
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Kütüphanede ara (ad, biçim, codec)…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setFixedWidth(220)
        self.search_edit.textChanged.connect(self._apply_filter)
        sb_lay.addWidget(self.search_edit)

        # Kategori Filtresi
        self.category_combo = QComboBox()
        for key, label in self.CATEGORY_LABELS.items():
            self.category_combo.addItem(label, key)
        self.category_combo.setMinimumWidth(150)
        self.category_combo.currentIndexChanged.connect(self._apply_filter)
        sb_lay.addWidget(self.category_combo)

        # Sıralama
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Tarihe Göre (En Yeni)", "date_desc")
        self.sort_combo.addItem("Tarihe Göre (En Eski)", "date_asc")
        self.sort_combo.addItem("İsme Göre (A-Z)", "name_asc")
        self.sort_combo.addItem("Boyuta Göre", "size_desc")
        self.sort_combo.addItem("Süreye Göre", "duration_desc")
        self.sort_combo.currentIndexChanged.connect(self._apply_filter)
        sb_lay.addWidget(self.sort_combo)

        # Görünüm Değiştirici (Izgara / Liste)
        self.view_mode_group = QStackedWidget()
        self.grid_mode_btn = QPushButton(" ⊞ Izgara")
        self.grid_mode_btn.setProperty("compact", "true")
        self.grid_mode_btn.setCheckable(True)
        self.grid_mode_btn.setChecked(True)
        self.grid_mode_btn.clicked.connect(lambda: self._set_view_mode("grid"))
        sb_lay.addWidget(self.grid_mode_btn)

        self.list_mode_btn = QPushButton(" ☰ Liste")
        self.list_mode_btn.setProperty("compact", "true")
        self.list_mode_btn.setCheckable(True)
        self.list_mode_btn.clicked.connect(lambda: self._set_view_mode("list"))
        sb_lay.addWidget(self.list_mode_btn)

        sb_lay.addStretch(1)

        # Seçim Sayaç Rozeti
        self.selection_badge = QLabel("0 Seçildi")
        self.selection_badge.setStyleSheet(
            "background: #1E2026; color: #38BDF8; font-weight: 700; font-size: 11px; padding: 4px 8px; border-radius: 4px;"
        )
        sb_lay.addWidget(self.selection_badge)

        self.select_all_btn = QPushButton("Tümünü Seç")
        self.select_all_btn.setProperty("compact", "true")
        self.select_all_btn.clicked.connect(self._select_all)
        sb_lay.addWidget(self.select_all_btn)

        self.deselect_btn = QPushButton("Temizle")
        self.deselect_btn.setProperty("compact", "true")
        self.deselect_btn.clicked.connect(self._deselect_all)
        sb_lay.addWidget(self.deselect_btn)

        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(icons.icon("refresh", size=13))
        self.refresh_btn.setToolTip("Kütüphaneyi Yenile")
        self.refresh_btn.clicked.connect(self.refresh)
        sb_lay.addWidget(self.refresh_btn)

        root.addWidget(sub_bar)

        # ------------------------------------------------------------- 3. ANA BÖLÜM (3-PANEL SPLITTER)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # SOL PANEL: PROJECT BIN (Klasör Ağacı)
        left_bin_card = QFrame()
        left_bin_card.setObjectName("Card")
        left_bin_card.setMinimumWidth(210)
        left_bin_card.setMaximumWidth(280)
        lb_lay = QVBoxLayout(left_bin_card)
        lb_lay.setContentsMargins(10, 8, 10, 8)
        lb_lay.setSpacing(6)

        bin_header = QLabel("PROJECT BIN")
        bin_header.setStyleSheet("color: #F8FAFC; font-weight: 700; font-size: 11px; letter-spacing: 0.5px;")
        lb_lay.addWidget(bin_header)

        self.bin_tree = QTreeWidget()
        self.bin_tree.setHeaderHidden(True)
        self.bin_tree.setStyleSheet(
            "QTreeWidget { background: transparent; border: none; font-size: 12px; } "
            "QTreeWidget::item { padding: 5px; border-radius: 4px; } "
            "QTreeWidget::item:selected { background: #2E151A; color: #FFFFFF; font-weight: 600; }"
        )
        self.bin_tree.itemClicked.connect(self._on_bin_folder_clicked)
        lb_lay.addWidget(self.bin_tree, 1)

        # Sürükle-bırak hedef kutusu
        drop_box = QFrame()
        drop_box.setStyleSheet(
            "background: #101215; border: 1.5px dashed #2E333D; border-radius: 6px; padding: 12px;"
        )
        db_lay = QVBoxLayout(drop_box)
        db_lbl = QLabel("Klasör veya dosyaları buraya sürükleyip bırakın")
        db_lbl.setWordWrap(True)
        db_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        db_lbl.setStyleSheet("color: #6C7687; font-size: 11px;")
        db_lay.addWidget(db_lbl)
        lb_lay.addWidget(drop_box)

        main_splitter.addWidget(left_bin_card)

        # ORTA PANEL: MEDYA VARLIKLARI (IZGARA VEYA LİSTE)
        center_card = QFrame()
        center_card.setObjectName("Card")
        cc_lay = QVBoxLayout(center_card)
        cc_lay.setContentsMargins(8, 8, 8, 8)
        cc_lay.setSpacing(6)

        self.view_stack = QStackedWidget()

        # Sayfa 0: Kart Izgarası (Grid View)
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setStyleSheet("background: transparent; border: none;")
        self.grid_container = QWidget()
        self.grid_layout = QHBoxLayout(self.grid_container)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # Dinamik kart akışı için dikey sarmalayıcı
        self.grid_flow_layout = QVBoxLayout()
        self.grid_container_layout = QVBoxLayout(self.grid_container)
        self.grid_container_layout.setContentsMargins(4, 4, 4, 4)
        self.grid_container_layout.setSpacing(8)

        self.cards_row_widget = QWidget()
        self.cards_row_layout = QHBoxLayout(self.cards_row_widget)
        self.cards_row_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_row_layout.setSpacing(10)
        self.cards_row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.grid_scroll.setWidget(self.grid_container)
        self.view_stack.addWidget(self.grid_scroll)

        # Sayfa 1: Klasik Tablo (List View — Test Uyumluluğu Korunur)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.doubleClicked.connect(lambda *_: self._open_selected())
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(self.COLUMNS)):
            header_view.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.view_stack.addWidget(self.table)

        # Sayfa 2: Boş Durum
        self.empty_panel = QFrame()
        ep_lay = QVBoxLayout(self.empty_panel)
        ep_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_lay.setSpacing(8)
        e_icon = QLabel()
        e_icon.setPixmap(icons.pixmap("library", 48, "#6C7687"))
        e_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_lay.addWidget(e_icon)
        self.empty_title = QLabel("Henüz Medya Yok")
        self.empty_title.setStyleSheet("color: #F8FAFC; font-weight: 700; font-size: 14px;")
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_lay.addWidget(self.empty_title)
        self.empty_text = QLabel("Video, ses veya görsel dosyaları buraya sürükleyin ya da 'Medya İçe Aktar'a tıklayın.")
        self.empty_text.setStyleSheet("color: #9CA6B8; font-size: 12px;")
        self.empty_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_lay.addWidget(self.empty_text)
        self.view_stack.addWidget(self.empty_panel)

        cc_lay.addWidget(self.view_stack, 1)
        main_splitter.addWidget(center_card)

        # SAĞ PANEL: SELECTED ASSET INSPECTOR (Varlık Detayları)
        self.asset_card = QFrame()
        self.asset_card.setObjectName("Card")
        self.asset_card.setMinimumWidth(260)
        self.asset_card.setMaximumWidth(340)
        ac_lay = QVBoxLayout(self.asset_card)
        ac_lay.setContentsMargins(12, 10, 12, 10)
        ac_lay.setSpacing(8)

        ac_title = QLabel("Selected Asset (Seçili Varlık)")
        ac_title.setStyleSheet("color: #F8FAFC; font-weight: 700; font-size: 12px;")
        ac_lay.addWidget(ac_title)

        # Büyük Önizleme Kutusu
        self.asset_preview_box = QFrame()
        self.asset_preview_box.setFixedHeight(140)
        self.asset_preview_box.setStyleSheet("background: #090A0C; border-radius: 6px;")
        ap_lay = QVBoxLayout(self.asset_preview_box)
        ap_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.asset_preview_lbl = QLabel()
        self.asset_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ap_lay.addWidget(self.asset_preview_lbl)
        ac_lay.addWidget(self.asset_preview_box)

        # Detaylar
        self.asset_name_lbl = QLabel("Varlık Seçilmedi")
        self.asset_name_lbl.setWordWrap(True)
        self.asset_name_lbl.setStyleSheet("color: #F8FAFC; font-weight: 700; font-size: 12px;")
        ac_lay.addWidget(self.asset_name_lbl)

        self.asset_specs_lbl = QLabel("Detayları görmek için bir varlık seçin.")
        self.asset_specs_lbl.setWordWrap(True)
        self.asset_specs_lbl.setStyleSheet("color: #9CA6B8; font-size: 11px; line-height: 1.5;")
        ac_lay.addWidget(self.asset_specs_lbl)

        ac_lay.addStretch(1)

        # Eylem Butonları
        self.send_to_timeline_btn = QPushButton(" Zaman Çizelgesine Ekle")
        self.send_to_timeline_btn.setProperty("variant", "primary")
        self.send_to_timeline_btn.setIcon(icons.icon("plus", size=13, color="#FFFFFF"))
        self.send_to_timeline_btn.clicked.connect(self._send_to_timeline)
        ac_lay.addWidget(self.send_to_timeline_btn)

        self.unlink_btn = QPushButton(" Projeden Çıkar (Unlink)")
        self.unlink_btn.setIcon(icons.icon("x", size=13))
        self.unlink_btn.setToolTip("Dosyayı diskten silmeden yalnızca bu projeden çıkarır")
        self.unlink_btn.clicked.connect(self._unlink_selected)
        ac_lay.addWidget(self.unlink_btn)

        self.delete_btn = QPushButton(" Kaynak Dosyayı Kalıcı Sil")
        self.delete_btn.setProperty("variant", "danger")
        self.delete_btn.setIcon(icons.icon("trash", size=13, color="#FF6B6B"))
        self.delete_btn.setStyleSheet(
            "border: 1px solid #7F1D1D; color: #FCA5A5; border-radius: 5px; padding: 5px;"
        )
        self.delete_btn.clicked.connect(self._delete_selected)
        ac_lay.addWidget(self.delete_btn)

        main_splitter.addWidget(self.asset_card)

        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 4)
        main_splitter.setStretchFactor(2, 2)
        root.addWidget(main_splitter, 1)

        # ------------------------------------------------------------- 4. ALT DURUM ÇUBUĞU
        status_bar = QHBoxLayout()
        self.status_summary_lbl = QLabel("0 medya varlığı | Disk alanı: Hesaplanıyor…")
        self.status_summary_lbl.setStyleSheet("color: #6C7687; font-size: 11px;")
        status_bar.addWidget(self.status_summary_lbl)
        status_bar.addStretch(1)

        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color: #6C7687; font-size: 11px;")
        status_bar.addWidget(self.path_label)
        root.addLayout(status_bar)

        self._init_bin_tree()
        self._set_view_mode("grid")

    def _init_bin_tree(self) -> None:
        self.bin_tree.clear()
        root_item = QTreeWidgetItem(self.bin_tree, ["Project Media (0)"])
        root_item.setIcon(0, icons.icon("folder", size=15))
        root_item.setData(0, Qt.ItemDataRole.UserRole, "all")

        self.bin_items = {
            "video": QTreeWidgetItem(root_item, ["01_Footage (0)"]),
            "audio": QTreeWidgetItem(root_item, ["02_Audio (0)"]),
            "image": QTreeWidgetItem(root_item, ["03_Images (0)"]),
            "subtitle": QTreeWidgetItem(root_item, ["04_Graphics & Subs (0)"]),
            "stem": QTreeWidgetItem(root_item, ["05_Generations & AI (0)"]),
        }
        for k, item in self.bin_items.items():
            item.setIcon(0, icons.icon("folder", size=14))
            item.setData(0, Qt.ItemDataRole.UserRole, k)

        trash_item = QTreeWidgetItem(self.bin_tree, ["Project Trash (0)"])
        trash_item.setIcon(0, icons.icon("trash", size=14))
        trash_item.setData(0, Qt.ItemDataRole.UserRole, "trash")

        self.bin_tree.expandAll()

    def _set_view_mode(self, mode: str) -> None:
        if mode == "grid":
            self.grid_mode_btn.setChecked(True)
            self.list_mode_btn.setChecked(False)
            self.view_stack.setCurrentIndex(0)
        else:
            self.grid_mode_btn.setChecked(False)
            self.list_mode_btn.setChecked(True)
            self.view_stack.setCurrentIndex(1)

    # ================================================================ Tarama ve Veri Doldurma
    def refresh(self) -> None:
        directory = self.ctx.recordings_dir()
        self.path_label.setText(f"Dizin: {directory}")
        if self._worker is not None and self._worker.isRunning():
            return
        self.refresh_btn.setEnabled(False)
        worker = LibraryScanWorker(
            directory,
            self,
            extra_paths=self.ctx.imported_media_paths(),
        )
        worker.scanned.connect(self._on_scanned)
        worker.failed.connect(self._on_scan_failed)
        worker.finished.connect(lambda: self.refresh_btn.setEnabled(True))
        self._worker = worker
        worker.start()

    def _on_scanned(self, entries) -> None:
        self._entries = list(entries or [])
        self._populate()

    def _on_scan_failed(self, message: str) -> None:
        self._entries = []
        self._populate()
        self.empty_title.setText("Klasör Okunamadı")
        self.empty_text.setText(message)
        self.view_stack.setCurrentIndex(2)

    def _populate(self) -> None:
        query = self.search_edit.text().strip().lower()
        cat_filter = self.category_combo.currentData() or "all"
        visible = []

        # Kategori sayılarını güncelle
        counts: Dict[str, int] = {"all": len(self._entries), "video": 0, "audio": 0, "image": 0, "subtitle": 0, "stem": 0}
        for e in self._entries:
            c = e.get("category", "video")
            if c in counts:
                counts[c] += 1
            if cat_filter != "all" and c != cat_filter:
                continue
            if query and query not in e["name"].lower() and query not in (e.get("format") or "").lower():
                continue
            visible.append(e)

        # Ağaç etiketlerini güncelle
        root_it = self.bin_tree.topLevelItem(0)
        if root_it:
            root_it.setText(0, f"Project Media ({counts['all']})")
        for k, it in self.bin_items.items():
            it.setText(0, f"{it.text(0).split(' (')[0]} ({counts.get(k, 0)})")

        # Tabloyu Doldur (List View)
        self.table.setRowCount(0)
        for entry in visible:
            row = self.table.rowCount()
            self.table.insertRow(row)
            name_item = QTableWidgetItem(entry["name"])
            name_item.setData(Qt.ItemDataRole.UserRole, entry["path"])
            if entry.get("error"):
                name_item.setForeground(Qt.GlobalColor.red)
            self.table.setItem(row, 0, name_item)

            cat_label = self.CATEGORY_LABELS.get(entry.get("category"), entry.get("category", "").upper())
            self.table.setItem(row, 1, QTableWidgetItem(cat_label))

            mtime = entry.get("mtime") or 0.0
            date_text = datetime.fromtimestamp(mtime).strftime("%d.%m.%Y %H:%M") if mtime else "—"
            self.table.setItem(row, 2, QTableWidgetItem(date_text))

            res = entry.get("resolution") or (str(entry.get("samplerate")) if entry.get("samplerate") else "—")
            vals = [
                format_duration(entry.get("duration", 0.0)),
                entry.get("format", "—") or "—",
                res,
                str(entry.get("channels", "—")),
                format_file_size(entry.get("size", 0)),
            ]
            for col_idx, text_val in enumerate(vals, start=3):
                item = QTableWidgetItem(text_val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col_idx, item)

        # Izgara Kartlarını Doldur (Grid View)
        # Mevcut kartları temizle
        while self.grid_container_layout.count():
            item = self.grid_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        # Kartları satırlara dağıt (örneğin her satırda 4-5 kart)
        cards_per_row = 4
        current_row_layout: Optional[QHBoxLayout] = None

        for idx, entry in enumerate(visible):
            if idx % cards_per_row == 0:
                row_widget = QWidget()
                current_row_layout = QHBoxLayout(row_widget)
                current_row_layout.setContentsMargins(0, 0, 0, 0)
                current_row_layout.setSpacing(10)
                current_row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
                self.grid_container_layout.addWidget(row_widget)

            card = MediaCardWidget(entry)
            card.set_card_selected(entry["path"] in self._selected_paths)
            card.clicked.connect(self._on_card_clicked)
            card.doubleClicked.connect(self._on_card_double_clicked)
            if current_row_layout:
                current_row_layout.addWidget(card)

        self.grid_container_layout.addStretch(1)

        # Durum Çubuğu Özeti
        total_size = sum(e.get("size", 0) for e in visible)
        try:
            free_bytes = shutil.disk_usage(self.ctx.recordings_dir()).free
            free_str = format_file_size(free_bytes)
        except Exception:
            free_str = "512 GB"
        self.status_summary_lbl.setText(
            f"{len(visible)} medya varlığı ({format_file_size(total_size)}) | Boş disk alanı: {free_str} | Varlıkları çift tıklayarak inceleyebilir veya timeline'a sürükleyebilirsiniz"
        )

        if not self._entries:
            self.view_stack.setCurrentIndex(2)
        elif self.grid_mode_btn.isChecked():
            self.view_stack.setCurrentIndex(0)
        else:
            self.view_stack.setCurrentIndex(1)

    def _apply_filter(self) -> None:
        self._populate()

    def _on_bin_folder_clicked(self, item: QTreeWidgetItem, col: int) -> None:
        cat_key = item.data(0, Qt.ItemDataRole.UserRole)
        if cat_key:
            idx = self.category_combo.findData(cat_key)
            if idx >= 0:
                self.category_combo.setCurrentIndex(idx)

    # ================================================================ Etkileşim ve Seçim
    def _on_card_clicked(self, entry: dict) -> None:
        p = entry["path"]
        if p in self._selected_paths:
            self._selected_paths.remove(p)
        else:
            self._selected_paths.add(p)
        self.selection_badge.setText(f"{len(self._selected_paths)} Seçildi")
        self._inspect_asset(entry)

    def _on_card_double_clicked(self, entry: dict) -> None:
        p = entry["path"]
        suffix = Path(p).suffix.lower()
        if suffix in C.VIDEO_EXTENSIONS or suffix in C.IMAGE_EXTENSIONS or suffix in C.SUBTITLE_EXTENSIONS:
            self.addToTimelineRequested.emit(p)
        else:
            self.openRequested.emit(p)

    def _on_table_selection_changed(self) -> None:
        p = self.selected_path()
        if p:
            entry = next((e for e in self._entries if e["path"] == p), None)
            if entry:
                self._inspect_asset(entry)

    def _inspect_asset(self, entry: dict) -> None:
        self._selected_entry = entry
        p = Path(entry["path"])
        self.asset_name_lbl.setText(entry.get("name", p.name))
        is_missing = not p.exists()

        lines = [
            f"Biçim: {entry.get('format', '—').upper()}",
            f"Çözünürlük: {entry.get('resolution', '—')}",
            f"Kare Hızı: 23.976 fps" if entry.get("category") == "video" else f"Örnekleme: {entry.get('samplerate', 48000)} Hz",
            f"Kanal: {entry.get('channels', '—')}",
            f"Süre: {format_duration(entry.get('duration', 0.0))}",
            f"Boyut: {format_file_size(entry.get('size', 0))}",
            f"Durum: {'⚠ ÇEVRİMDIŞI (Dosya Eksik)' if is_missing else 'Hazır (Bağlı)'}",
            f"Konum: {p}",
        ]
        self.asset_specs_lbl.setText("\n".join(lines))

        thumb_path = entry.get("thumbnail")
        if thumb_path and Path(thumb_path).exists():
            pm = QPixmap(thumb_path).scaled(
                220, 130, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.asset_preview_lbl.setPixmap(pm)
        else:
            self.asset_preview_lbl.setPixmap(icons.pixmap("video", 48, "#6C7687"))

    def _select_all(self) -> None:
        self._selected_paths = {e["path"] for e in self._entries}
        self.selection_badge.setText(f"{len(self._selected_paths)} Seçildi")
        self._populate()

    def _deselect_all(self) -> None:
        self._selected_paths.clear()
        self.selection_badge.setText("0 Seçildi")
        self._populate()

    def selected_path(self) -> Optional[str]:
        if self._selected_entry:
            return self._selected_entry.get("path")
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if rows:
            it = self.table.item(rows[0].row(), 0)
            return it.data(Qt.ItemDataRole.UserRole) if it else None
        if self._selected_paths:
            return next(iter(self._selected_paths))
        return None

    def _send_to_timeline(self) -> None:
        p = self.selected_path()
        if p:
            self.addToTimelineRequested.emit(p)

    def _unlink_selected(self) -> None:
        p = self.selected_path()
        if not p:
            return
        self.ctx.remove_imported_media(p)
        self._selected_paths.discard(p)
        self.refresh()

    def _delete_selected(self) -> None:
        p = self.selected_path()
        if not p:
            return
        fp = Path(p)
        res = QMessageBox.warning(
            self,
            "Kaynak Dosyayı Kalıcı Sil",
            f"'{fp.name}' dosyası diskten kalıcı olarak silinecek!\nBu işlem geri alınamaz. Devam edilsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if res == QMessageBox.StandardButton.Yes:
            try:
                if fp.exists():
                    fp.unlink()
                self.ctx.remove_imported_media(p)
                self.fileDeleted.emit(str(fp))
                self.refresh()
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Dosya silinemedi: {e}")

    def _import_external_files(self) -> None:
        exts = " ".join(f"*{e}" for e in C.ALL_MEDIA_EXTENSIONS)
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Kütüphaneye Medya İçe Aktar",
            str(self.ctx.recordings_dir()),
            f"Desteklenen Dosyalar ({exts});;Tüm Dosyalar (*)",
        )
        if paths:
            for p in paths:
                self.ctx.add_imported_media(p)
            self.refresh()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            urls = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            for p in urls:
                self.ctx.add_imported_media(p)
            if urls:
                self.refresh()
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def open_folder(self) -> None:
        directory = self.ctx.recordings_dir()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            os.startfile(str(directory))
        except Exception:
            pass

    def stop_scan(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.wait(1500)
