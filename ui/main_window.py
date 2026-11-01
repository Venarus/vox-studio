"""
VOX Studio ana penceresi.

Yerleşim: sol kenar çubuğu, üst çubuk, modül alanı ve alt taşıma çubuğu.
Kayıt/oynatma motorları Qt'den bağımsızdır; bu pencere yalnızca orkestrasyon
ve durum gösteriminden sorumludur. Kayıt sürerken arayüz donmaz.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QFontMetrics, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from audio.player import PlaybackState
from core import constants as C
from core.format import format_duration
from core.logger import get_logger
from ui import icons
from ui.context import AppContext
from ui.theme import FONT_MONO
from ui.views.recording_studio import RecordingStudioView
from ui.views.processing import ProcessingView
from ui.views.settings import SettingsView
from ui.widgets.level_meter import LevelMeter
from ui.workers import AnalysisWorker, PeaksWorker

logger = get_logger("ui.main_window")


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx

        self._recorder_state = "idle"
        self._player_state = "idle"
        self._current_path: Optional[str] = None
        self._current_view = ""
        self._countdown_active = False
        self._countdown_remaining = 0
        self._tick_count = 0
        self._workers = set()
        self._analysis_worker: Optional[AnalysisWorker] = None
        self._last_cpu_time = time.process_time()
        self._last_wall_time = time.monotonic()

        self.setWindowTitle(f"{C.APP_NAME} — {C.APP_SUBTITLE}")
        self.setMinimumSize(QSize(*C.MIN_WINDOW_SIZE))
        self.resize(QSize(*C.DEFAULT_WINDOW_SIZE))
        self.setAcceptDrops(True)

        self._build_ui()
        self._wire_signals()
        self._init_timers()

        self._show_view("studio")
        self._refresh_device_labels()
        self._update_badge()
        self._sync_controls()
        self._set_time(0.0, 0.0)
        self._set_status("Hazır. Mikrofonunuzu seçip giriş kontrolü yapabilirsiniz.")

    # ================================================================ yerleşim
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("AppRoot")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_topbar())

        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        outer.addLayout(middle, 1)

        middle.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        middle.addWidget(self.stack, 1)
        self._build_views()

        outer.addWidget(self._build_transport())

    # ------------------------------------------------------------------ sidebar
    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(C.SIDEBAR_WIDTH)
        self._sidebar_frame = sidebar
        self._sidebar_collapsed = False
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        brand = QWidget()
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(14, 13, 10, 13)
        brand_layout.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(icons.app_icon().pixmap(28, 28))
        logo.setFixedSize(28, 28)
        brand_layout.addWidget(logo)
        text_box = QVBoxLayout()
        text_box.setSpacing(0)
        self._brand_name_lbl = QLabel("VOX Studio")
        self._brand_name_lbl.setObjectName("BrandTitle")
        self._brand_sub_lbl = QLabel("AUDIO ENGINE V1")
        self._brand_sub_lbl.setObjectName("BrandSub")
        text_box.addWidget(self._brand_name_lbl)
        text_box.addWidget(self._brand_sub_lbl)
        brand_layout.addLayout(text_box)
        brand_layout.addStretch(1)

        self.sidebar_toggle_btn = QPushButton()
        self.sidebar_toggle_btn.setProperty("variant", "ghost")
        self.sidebar_toggle_btn.setFixedSize(26, 26)
        self.sidebar_toggle_btn.setIcon(icons.icon("chevron-left", C.COLOR_TEXT_MUTED, size=15))
        self.sidebar_toggle_btn.setToolTip("Kenar çubuğunu daralt / genişlet")
        self.sidebar_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sidebar_toggle_btn.clicked.connect(self._toggle_sidebar)
        brand_layout.addWidget(self.sidebar_toggle_btn)
        layout.addWidget(brand)

        divider = QFrame()
        divider.setObjectName("Divider")
        layout.addWidget(divider)

        nav = QWidget()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(8, 10, 8, 8)
        nav_layout.setSpacing(3)

        self.nav_buttons: Dict[str, QPushButton] = {}
        self._nav_icons: Dict[str, str] = {}
        self._nav_labels: Dict[str, str] = {}
        entries = [
            ("studio", "Kayıt Stüdyosu", "headphones"),
            ("processing", "AI Restorasyon Lab", "sliders"),
            ("settings", "Ayarlar", "gear"),
        ]
        for key, label, icon_name in entries:
            nav_layout.addWidget(self._nav_button(key, label, icon_name))

        # Görevler (Arka Plan İşlemleri) satırı + gerçek görev sayısı rozeti
        tasks_row_w = QWidget()
        tasks_row = QHBoxLayout(tasks_row_w)
        tasks_row.setContentsMargins(0, 0, 0, 0)
        tasks_row.setSpacing(4)
        self.tasks_nav_btn = QPushButton("  Görevler")
        self.tasks_nav_btn.setProperty("nav", "true")
        self.tasks_nav_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tasks_nav_btn.setIconSize(QSize(18, 18))
        self.tasks_nav_btn.setIcon(icons.icon("tasks", C.COLOR_TEXT_SECONDARY))
        self.tasks_nav_btn.setToolTip("Arka plan görevlerini, sistem tanılamayı ve önbelleği görüntüle")
        self.tasks_nav_btn.clicked.connect(self._open_tasks_dialog)
        tasks_row.addWidget(self.tasks_nav_btn, 1)

        self.tasks_badge_lbl = QLabel("0")
        self.tasks_badge_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tasks_badge_lbl.setFixedSize(22, 18)
        self.tasks_badge_lbl.setStyleSheet(
            f"background: {C.COLOR_ACCENT}; color: #FFFFFF; font-size: 10px; "
            "font-weight: 700; border-radius: 9px;"
        )
        self.tasks_badge_lbl.setVisible(False)
        tasks_row.addWidget(self.tasks_badge_lbl)
        nav_layout.addWidget(tasks_row_w)

        settings_btn = self._nav_button("settings", "Ayarlar", "gear")
        nav_layout.addWidget(settings_btn)

        nav_layout.addStretch(1)

        # Sol alt hızlı tanıtım / erişim kartı (referans tasarımla uyumlu, sade)
        self._sidebar_promo_card = QFrame()
        self._sidebar_promo_card.setObjectName("Card")
        self._sidebar_promo_card.setStyleSheet(
            f"#Card {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, "
            f"stop:0 #1C1215, stop:1 {C.COLOR_BG_SURFACE}); "
            f"border: 1px solid #351C21; border-radius: 12px; }}"
        )
        promo_lay = QVBoxLayout(self._sidebar_promo_card)
        promo_lay.setContentsMargins(12, 12, 12, 12)
        promo_lay.setSpacing(6)

        promo_icon_box = QLabel()
        promo_icon_box.setFixedSize(32, 32)
        promo_icon_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        promo_icon_box.setObjectName("IconBadge")
        promo_icon_box.setPixmap(icons.pixmap("waveform", 18, C.COLOR_ACCENT_HOVER))
        promo_lay.addWidget(promo_icon_box, 0, Qt.AlignmentFlag.AlignHCenter)

        promo_title = QLabel("Profesyonel Ses\nAyrıştırma")
        promo_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        promo_title.setStyleSheet(f"color: {C.COLOR_TEXT_PRIMARY}; font-weight: 700; font-size: 12px;")
        promo_lay.addWidget(promo_title)

        promo_sub = QLabel("Video ve ses dosyalarından temiz stem'ler oluştur.")
        promo_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        promo_sub.setWordWrap(True)
        promo_sub.setStyleSheet(f"color: {C.COLOR_TEXT_MUTED}; font-size: 10px;")
        promo_lay.addWidget(promo_sub)

        promo_btn = QPushButton("Hemen Başla  ›")
        promo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        promo_btn.setStyleSheet(
            f"QPushButton {{ background: {C.COLOR_BG_PANEL}; border: 1px solid #3D1F25; "
            f"border-radius: 7px; padding: 5px 10px; font-size: 11px; font-weight: 600; color: {C.COLOR_TEXT_PRIMARY}; }}"
            f"QPushButton:hover {{ border-color: {C.COLOR_ACCENT}; background: {C.COLOR_BG_HOVER}; }}"
        )
        promo_btn.clicked.connect(lambda: self._show_view("stems"))
        promo_lay.addWidget(promo_btn)
        nav_layout.addWidget(self._sidebar_promo_card)

        layout.addWidget(nav, 1)

        self._sidebar_footer = QWidget()
        footer_layout = QVBoxLayout(self._sidebar_footer)
        footer_layout.setContentsMargins(16, 6, 16, 12)
        footer_layout.setSpacing(1)
        stage = QLabel(f"Tam Stüdyo · v{C.APP_VERSION}")
        stage.setObjectName("MutedSmall")
        local = QLabel("Yerel çalışır · Telemetri yok")
        local.setObjectName("MutedSmall")
        footer_layout.addWidget(stage)
        footer_layout.addWidget(local)
        layout.addWidget(self._sidebar_footer)
        return sidebar

    def _toggle_sidebar(self) -> None:
        self._sidebar_collapsed = not self._sidebar_collapsed
        if self._sidebar_collapsed:
            self._sidebar_frame.setFixedWidth(C.SIDEBAR_COLLAPSED_WIDTH)
            self._brand_name_lbl.setVisible(False)
            self._brand_sub_lbl.setVisible(False)
            self._sidebar_promo_card.setVisible(False)
            self._sidebar_footer.setVisible(False)
            self.sidebar_toggle_btn.setIcon(icons.icon("chevron-right", C.COLOR_TEXT_MUTED, size=15))
            for k, btn in self.nav_buttons.items():
                btn.setText("")
                btn.setToolTip(self._nav_labels.get(k, k))
            self.tasks_nav_btn.setText("")
        else:
            self._sidebar_frame.setFixedWidth(C.SIDEBAR_WIDTH)
            self._brand_name_lbl.setVisible(True)
            self._brand_sub_lbl.setVisible(True)
            self._sidebar_promo_card.setVisible(True)
            self._sidebar_footer.setVisible(True)
            self.sidebar_toggle_btn.setIcon(icons.icon("chevron-left", C.COLOR_TEXT_MUTED, size=15))
            for k, btn in self.nav_buttons.items():
                btn.setText(f"  {self._nav_labels.get(k, k)}")
            self.tasks_nav_btn.setText("  Görevler")

    def _nav_button(self, key: str, label: str, icon_name: str) -> QPushButton:
        btn = QPushButton(f"  {label}")
        btn.setCheckable(True)
        btn.setProperty("nav", "true")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setIconSize(QSize(18, 18))
        btn.setIcon(icons.icon(icon_name, C.COLOR_TEXT_SECONDARY))
        btn.clicked.connect(lambda _=False, k=key: self._show_view(k))
        btn.toggled.connect(
            lambda checked, b=btn, name=icon_name: b.setIcon(
                icons.icon(name, C.COLOR_ACCENT_HOVER if checked else C.COLOR_TEXT_SECONDARY)
            )
        )
        self.nav_buttons[key] = btn
        self._nav_icons[key] = icon_name
        self._nav_labels[key] = label
        return btn

    # ------------------------------------------------------------------ üst çubuk
    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(C.TOPBAR_HEIGHT)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(14)

        # Global arama kutusu (Bölüm 3.B)
        search_wrap = QFrame()
        search_wrap.setObjectName("TopSearchEdit")
        search_wrap.setFixedWidth(320)
        search_wrap.setFixedHeight(34)
        sw_lay = QHBoxLayout(search_wrap)
        sw_lay.setContentsMargins(10, 0, 8, 0)
        sw_lay.setSpacing(8)
        s_icon = QLabel()
        s_icon.setPixmap(icons.pixmap("search", 14, C.COLOR_TEXT_MUTED))
        sw_lay.addWidget(s_icon)
        self.global_search_edit = QLineEdit()
        self.global_search_edit.setPlaceholderText("Proje, dosya veya özellik ara...")
        self.global_search_edit.setStyleSheet("background: transparent; border: none; padding: 0; font-size: 12px;")
        self.global_search_edit.textChanged.connect(self._on_global_search_text_changed)
        self.global_search_edit.returnPressed.connect(self._on_global_search_return)
        sw_lay.addWidget(self.global_search_edit, 1)
        layout.addWidget(search_wrap)

        divider0 = QFrame()
        divider0.setObjectName("VDivider")
        divider0.setFixedHeight(22)
        layout.addWidget(divider0)

        project_caption = QLabel("AKTİF PROJE")
        project_caption.setObjectName("MetricLabel")
        self.project_value_label = QLabel("Proje yok")
        self.project_value_label.setObjectName("TopValue")
        self.project_value_label.setToolTip("Ana Sayfa'dan yeni proje oluşturabilir veya mevcut bir .voxproj açabilirsiniz")
        box = QVBoxLayout()
        box.setSpacing(1)
        box.addWidget(project_caption)
        box.addWidget(self.project_value_label)
        layout.addLayout(box)

        layout.addStretch(1)

        self.record_badge = QLabel("● HAZIR")
        self.record_badge.setObjectName("RecordBadge")
        self.record_badge.setToolTip("Gerçek kayıt motorunun anlık durumu")
        layout.addWidget(self.record_badge)

        divider = QFrame()
        divider.setObjectName("VDivider")
        divider.setFixedHeight(22)
        layout.addWidget(divider)

        device_box = QHBoxLayout()
        device_box.setSpacing(6)
        mic_icon = QLabel()
        mic_icon.setPixmap(icons.pixmap("mic", 15, C.COLOR_TEXT_SECONDARY))
        device_box.addWidget(mic_icon)
        self.device_label = QLabel("Giriş cihazı: —")
        self.device_label.setObjectName("TopValue")
        device_box.addWidget(self.device_label)
        layout.addLayout(device_box)

        divider2 = QFrame()
        divider2.setObjectName("VDivider")
        divider2.setFixedHeight(22)
        layout.addWidget(divider2)

        self.cpu_label = QLabel("CPU —")
        self.cpu_label.setObjectName("TopValue")
        self.cpu_label.setToolTip(
            "Uygulama sürecinin işlemci zamanı / geçen süre. Sistemin toplam CPU "
            "kullanımı değildir."
        )
        layout.addWidget(self.cpu_label)

        top_shortcuts_btn = QPushButton()
        top_shortcuts_btn.setProperty("variant", "ghost")
        top_shortcuts_btn.setFixedSize(32, 32)
        top_shortcuts_btn.setIcon(icons.icon("help-circle", C.COLOR_TEXT_SECONDARY, size=16))
        top_shortcuts_btn.setToolTip("Klavye Kısayolları ve CapCut Eşleşme Rehberi (F1)")
        top_shortcuts_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        top_shortcuts_btn.clicked.connect(self._open_shortcuts_guide)
        layout.addWidget(top_shortcuts_btn)

        top_settings_btn = QPushButton()
        top_settings_btn.setProperty("variant", "ghost")
        top_settings_btn.setFixedSize(32, 32)
        top_settings_btn.setIcon(icons.icon("gear", C.COLOR_TEXT_SECONDARY, size=16))
        top_settings_btn.setToolTip("Ayarları aç")
        top_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        top_settings_btn.clicked.connect(lambda: self._show_view("settings"))
        layout.addWidget(top_settings_btn)
        return bar

    # ------------------------------------------------------------------ görünümler
    def _build_views(self) -> None:
        self.studio = RecordingStudioView(self.ctx)
        self.processing = ProcessingView(self.ctx)
        self.settings_view = SettingsView(self.ctx)

        self._views: Dict[str, QWidget] = {
            "studio": self.studio,
            "processing": self.processing,
            "settings": self.settings_view,
        }
        for view in self._views.values():
            self.stack.addWidget(view)

    # ------------------------------------------------------------------ taşıma
    def _build_transport(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TransportBar")
        bar.setFixedHeight(C.TRANSPORT_HEIGHT)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(10)

        self.record_btn = QPushButton("  Kayıt")
        self.record_btn.setObjectName("RecordButton")
        self.record_btn.setIcon(icons.icon("record", size=14, color="#FFFFFF"))
        self.record_btn.setIconSize(QSize(14, 14))
        self.record_btn.setMinimumSize(QSize(108, 40))
        self.record_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_btn.setToolTip("Kaydı başlat / bitir (Ctrl+R)")
        self.record_btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_btn)

        self.cancel_btn = QPushButton("İptal")
        self.cancel_btn.setProperty("variant", "danger")
        self.cancel_btn.setToolTip("Süren kaydı iptal et ve yarım dosyayı sil")
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.clicked.connect(self.cancel_recording)
        self.cancel_btn.setVisible(False)
        layout.addWidget(self.cancel_btn)

        self.play_btn = QPushButton()
        self.play_btn.setObjectName("TransportButton")
        self.play_btn.setIcon(icons.icon("play", C.COLOR_TEXT_PRIMARY))
        self.play_btn.setIconSize(QSize(18, 18))
        self.play_btn.setFixedSize(QSize(40, 40))
        self.play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_btn.setToolTip("Oynat / duraklat (Ctrl+P)")
        self.play_btn.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton()
        self.stop_btn.setObjectName("TransportButton")
        self.stop_btn.setIcon(icons.icon("stop", C.COLOR_TEXT_PRIMARY))
        self.stop_btn.setIconSize(QSize(16, 16))
        self.stop_btn.setFixedSize(QSize(40, 40))
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setToolTip("Durdur (Esc)")
        self.stop_btn.clicked.connect(self.stop_transport)
        layout.addWidget(self.stop_btn)

        divider = QFrame()
        divider.setObjectName("VDivider")
        divider.setFixedHeight(34)
        layout.addWidget(divider)

        time_box = QVBoxLayout()
        time_box.setSpacing(0)
        self.time_label = QLabel("00:00.0")
        self.time_label.setObjectName("TimeDisplay")
        self.time_sub = QLabel("HAZIR")
        self.time_sub.setObjectName("TimeSub")
        time_box.addWidget(self.time_label)
        time_box.addWidget(self.time_sub)
        layout.addLayout(time_box)

        self.meter = LevelMeter(bar, show_scale=False, show_clip=True)
        self.meter.setFixedSize(QSize(230, 18))
        layout.addWidget(self.meter)

        layout.addStretch(1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("StatusText")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.status_label.setMinimumWidth(0)
        self.status_label.setMaximumWidth(720)
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self.status_label, 1)

        hint = QLabel("Ctrl+R · Ctrl+P · Esc")
        hint.setObjectName("MutedSmall")
        hint.setToolTip("Ctrl+R: kayıt · Ctrl+P: oynat/duraklat · Esc: durdur")
        hint.setFixedWidth(130)
        hint.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(hint)
        return bar

    # ================================================================ sinyaller
    def _wire_signals(self) -> None:
        bridge = self.ctx.bridge
        bridge.recorderState.connect(self._on_recorder_state)
        bridge.recorderError.connect(self._on_recorder_error)
        bridge.playerState.connect(self._on_player_state)
        bridge.playerError.connect(self._on_player_error)
        bridge.monitorError.connect(self._on_monitor_error)

        self.studio.openFileRequested.connect(self._open_in_restoration_lab)
        self.studio.analyzeRequested.connect(self._start_analysis)
        self.studio.seekRequested.connect(self.seek_playback)
        self.studio.settingsChanged.connect(self._on_studio_settings_changed)
        self.studio.monitorStateChanged.connect(self._on_monitor_state_changed)
        if hasattr(self.studio, "sendToTimelineRequested"):
            self.studio.sendToTimelineRequested.connect(self._open_in_restoration_lab)

        self.settings_view.settingsChanged.connect(self._on_settings_changed)
        self.settings_view.recordingsChanged.connect(self._on_recordings_changed)

        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.toggle_recording)
        QShortcut(QKeySequence("Ctrl+P"), self, activated=self.toggle_playback)
        QShortcut(QKeySequence("Esc"), self, activated=self.stop_transport)
        QShortcut(QKeySequence("F1"), self, activated=self._open_shortcuts_guide)
        self._apply_ui_language()

    def _open_shortcuts_guide(self) -> None:
        from ui.widgets.shortcut_guide import ShortcutGuideDialog
        dlg = ShortcutGuideDialog(self)
        dlg.exec()

    def _apply_ui_language(self, _lang_code: Optional[str] = None) -> None:
        if not hasattr(self.ctx, "i18n"):
            return
        i18n = self.ctx.i18n
        is_rtl = i18n.is_rtl()
        self.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft if is_rtl else Qt.LayoutDirection.LeftToRight
        )
        nav_key_map = {
            "projects": "nav.home",
            "editor": "nav.editor",
            "stems": "nav.stems",
            "dubbing": "nav.dubbing",
            "library": "nav.library",
            "studio": "nav.studio",
            "processing": "nav.processing",
            "transcription": "nav.transcription",
            "settings": "nav.settings",
        }
        for key, t_key in nav_key_map.items():
            translated = i18n.t(t_key)
            self._nav_labels[key] = translated
            btn = self.nav_buttons.get(key)
            if btn is not None:
                if self._sidebar_collapsed:
                    btn.setText("")
                else:
                    btn.setText(f"  {translated}")
                btn.setToolTip(translated)
        if hasattr(self, "tasks_nav_btn"):
            t_tasks = i18n.t("nav.tasks")
            self.tasks_nav_btn.setText("" if self._sidebar_collapsed else f"  {t_tasks}")
            self.tasks_nav_btn.setToolTip(t_tasks)
        if hasattr(self, "global_search_edit"):
            self.global_search_edit.setPlaceholderText(i18n.t("top.search_placeholder"))
        self._sync_controls()

    def _init_timers(self) -> None:
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._tick)

        self._cpu_timer = QTimer(self)
        self._cpu_timer.setInterval(1000)
        self._cpu_timer.timeout.connect(self._update_cpu)
        self._cpu_timer.start()

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setSingleShot(True)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._countdown_tick)

    # ================================================================ görünüm
    def _show_view(self, key: str) -> None:
        if key not in self._views:
            return
        if self._current_view == "studio" and key != "studio":
            self.studio.stop_monitor()
        self._current_view = key
        self.stack.setCurrentWidget(self._views[key])
        for name, btn in self.nav_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(name == key)
            btn.blockSignals(False)
            btn.setIcon(
                icons.icon(
                    self._nav_icons[name],
                    C.COLOR_ACCENT_HOVER if name == key else C.COLOR_TEXT_SECONDARY,
                )
            )

        if key == "studio":
            self.studio.refresh_devices()
            self.studio.refresh_recent()
            self.studio.set_settings_from_config()
        elif key == "library":
            self.library.refresh()
        elif key == "settings":
            self.settings_view.refresh()
        elif key == "projects":
            self.projects.refresh_recent()
        self._refresh_device_labels()
        self._sync_controls()

    def _refresh_tasks_badge(self) -> None:
        if not hasattr(self, "tasks_badge_lbl") or not hasattr(self.ctx, "task_manager"):
            return
        running = len(self.ctx.task_manager.running_tasks())
        if running > 0:
            self.tasks_badge_lbl.setText(str(running))
            self.tasks_badge_lbl.setVisible(True)
        else:
            self.tasks_badge_lbl.setVisible(False)
        if hasattr(self, "projects") and hasattr(self.projects, "refresh_tasks_panel"):
            self.projects.refresh_tasks_panel()

    def _open_tasks_dialog(self) -> None:
        from ui.views.video_editor_view import DiagnosticsAndTasksDialog

        dlg = DiagnosticsAndTasksDialog(self.ctx, self.editor, self)
        dlg.exec()

    def _on_global_search_text_changed(self, text: str) -> None:
        q = text.strip()
        if self._current_view == "projects" and hasattr(self.projects, "set_project_search_query"):
            self.projects.set_project_search_query(q)
        elif self._current_view == "library" and hasattr(self.library, "search_edit"):
            self.library.search_edit.setText(q)
        elif self._current_view == "editor" and hasattr(self.editor, "bin_search_edit"):
            self.editor.bin_search_edit.setText(q)

    def _on_global_search_return(self) -> None:
        q = self.global_search_edit.text().strip().lower()
        if not q:
            return
        feature_map = [
            (("stem", "ayrıştır", "vokal", "müzik ayır"), "stems"),
            (("dublaj", "replik", "karakter"), "dubbing"),
            (("altyazı", "transkrip", "srt", "vtt", "metin"), "transcription"),
            (("video", "editör", "timeline", "kurgu"), "editor"),
            (("kayıt", "mikrofon"), "studio"),
            (("efekt", "işleme", "eq", "kompresör"), "processing"),
            (("kütüphane", "medya"), "library"),
            (("ayar", "cihaz", "klasör"), "settings"),
        ]
        for keywords, target_view in feature_map:
            if any(k in q for k in keywords):
                self._show_view(target_view)
                return
        # Özellik eşleşmediyse Ana Sayfa'da proje aramasına yönlendir
        if self._current_view != "projects":
            self._show_view("projects")
        if hasattr(self.projects, "set_project_search_query"):
            self.projects.set_project_search_query(self.global_search_edit.text().strip())

    # ================================================================ cihaz/üst
    def _refresh_device_labels(self) -> None:
        device = self.ctx.input_device()
        if device is None:
            self.device_label.setText("Giriş cihazı yok")
            self.device_label.setStyleSheet(f"color: {C.COLOR_VU_RED};")
            self.device_label.setToolTip(
                "Ses giriş cihazı bulunamadı. Mikrofonu bağlayıp Kayıt Stüdyosu'ndaki "
                "'Yenile' düğmesini kullanın."
            )
            return
        metrics = QFontMetrics(self.device_label.font())
        text = metrics.elidedText(f"Giriş: {device.name}", Qt.TextElideMode.ElideRight, 210)
        self.device_label.setText(text)
        self.device_label.setStyleSheet("")
        rate = self.ctx.config.get_int("sample_rate", C.DEFAULT_SAMPLE_RATE)
        channels = self.ctx.config.get_int("channels", 1)
        subtype = self.ctx.config.get_str("recording_subtype", "PCM_24")
        self.device_label.setToolTip(
            f"{device.name}\n{device.hostapi_name} · cihaz #{device.index}\n"
            f"Kanallar: {device.max_input_channels} giriş · varsayılan {device.default_samplerate:.0f} Hz\n"
            f"Seçili kayıt: {rate} Hz · {channels} kanal · {subtype}"
        )

    def _on_settings_changed(self) -> None:
        self._refresh_device_labels()
        self.studio.set_settings_from_config()
        self._sync_controls()

    def _on_studio_settings_changed(self) -> None:
        self._refresh_device_labels()
        if self.ctx.monitor.active:
            self.studio.stop_monitor()
            self.studio.monitor_check.setChecked(True)
        self._sync_controls()

    def _on_monitor_state_changed(self, active: bool) -> None:
        if active:
            self._ui_timer.start()
            self._set_status("Giriş dinleniyor — seviye ölçülüyor, kayıt yapılmıyor.")
        else:
            self.meter.reset()
            self._maybe_stop_ui_timer()

    def _on_monitor_error(self, message: str) -> None:
        self._set_status(message, "error")

    # ================================================================ durum
    def _update_badge(self) -> None:
        state = self._recorder_state
        if state == "recording":
            text, prop = "● KAYIT", "recording"
        elif state == "paused":
            text, prop = "● DURAKLATILDI", "paused"
        elif state in ("starting", "stopping"):
            text, prop = "● HAZIRLANIYOR", "busy"
        elif state == "error":
            text, prop = "● HATA", "error"
        else:
            text, prop = "● HAZIR", "idle"
        self.record_badge.setText(text)
        self.record_badge.setProperty("state", prop)
        self.record_badge.style().unpolish(self.record_badge)
        self.record_badge.style().polish(self.record_badge)

    def _set_status(self, text: str, kind: str = "info") -> None:
        colors = {
            "info": C.COLOR_TEXT_SECONDARY,
            "ok": C.COLOR_VU_GREEN,
            "warn": C.COLOR_VU_YELLOW,
            "error": C.COLOR_VU_RED,
        }
        metrics = QFontMetrics(self.status_label.font())
        available = max(200, self.status_label.width() - 8)
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, available)
        self.status_label.setText(elided)
        self.status_label.setToolTip(text)
        self.status_label.setStyleSheet(f"color: {colors.get(kind, C.COLOR_TEXT_SECONDARY)};")

    def _sync_controls(self) -> None:
        rec_active = self.ctx.recorder.is_active
        state = self.ctx.player.state
        playing = state is PlaybackState.PLAYING
        paused = state is PlaybackState.PAUSED
        countdown = self._countdown_active
        can_play = self._current_path is not None and not rec_active and not countdown

        if countdown:
            self.record_btn.setText("  İptal")
            self.record_btn.setIcon(icons.icon("stop", size=14, color="#FFFFFF"))
            self.record_btn.setEnabled(True)
        elif rec_active:
            self.record_btn.setText("  Kaydı Bitir")
            self.record_btn.setIcon(icons.icon("stop", size=14, color="#FFFFFF"))
            self.record_btn.setEnabled(True)
        else:
            self.record_btn.setText("  Kayıt")
            self.record_btn.setIcon(icons.icon("record", size=14, color="#FFFFFF"))
            self.record_btn.setEnabled(self.ctx.input_device() is not None)

        self.record_btn.setProperty("recording", "true" if rec_active else "false")
        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)

        self.cancel_btn.setVisible(rec_active)
        self.cancel_btn.setEnabled(rec_active)

        self.play_btn.setEnabled(can_play)
        self.play_btn.setIcon(
            icons.icon("pause" if playing else "play", C.COLOR_TEXT_PRIMARY)
        )
        self.play_btn.setToolTip(
            "Duraklat (Ctrl+P)" if playing else "Oynat (Ctrl+P)" if can_play else "Oynatılacak dosya yok"
        )

        self.stop_btn.setEnabled(rec_active or playing or paused or countdown)

    # ================================================================ kayıt
    def _record_from_projects(self) -> None:
        self._show_view("studio")
        self.start_recording()

    def _on_record_cue_requested(self, text: str, char_name: str) -> None:
        self._show_view("studio")
        label = f"{char_name}_{text[:20]}".replace(" ", "_")
        self.studio.set_label_text(label)

    def _on_transfer_dubbing_takes(
        self, records: list, lang: str = "Türkçe (TR)", offset_ms: float = 0.0
    ) -> None:
        self._show_view("editor")
        self.editor.receive_dubbing_takes(records, lang, offset_ms)

    def _on_active_project_changed(self, proj) -> None:
        if proj is not None:
            self.project_value_label.setText(proj.name)
            db = self.ctx.project_manager.active_db
            self.project_value_label.setToolTip(str(db.db_path) if db else proj.name)
        else:
            self.project_value_label.setText("Proje yok")
            self.project_value_label.setToolTip("Aktif bir .voxproj projesi açık değil")

    def _on_add_to_timeline_from_library(self, path: str) -> None:
        self._show_view("editor")
        self.editor.import_files([path])

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            urls = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            if urls:
                first_ext = Path(urls[0]).suffix.lower()
                if first_ext == ".voxproj":
                    self.projects._open_project_path(urls[0])
                elif self._current_view == "editor" or first_ext in C.VIDEO_EXTENSIONS or first_ext in C.IMAGE_EXTENSIONS or first_ext in C.SUBTITLE_EXTENSIONS:
                    self._show_view("editor")
                    self.editor.import_files(urls)
                else:
                    self.open_file(urls[0])
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def toggle_recording(self) -> None:
        if self._countdown_active:
            self._cancel_countdown()
            return
        if self.ctx.recorder.is_active:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self.ctx.recorder.is_active or self._countdown_active:
            return
        if self.ctx.player.state is not PlaybackState.IDLE:
            self.ctx.player.stop()
            self.ctx.player.seek(0.0)
        device = self.ctx.input_device()
        if device is None:
            QMessageBox.warning(
                self,
                "Giriş cihazı yok",
                "Ses giriş cihazı bulunamadı.\n\n"
                "Mikrofonun bağlı olduğundan ve Windows gizlilik ayarlarında mikrofon "
                "izninin verildiğinden emin olun. Ardından Kayıt Stüdyosu'ndaki "
                "'Yenile' düğmesine basın.",
            )
            return
        rate = self.ctx.config.get_int("sample_rate", C.DEFAULT_SAMPLE_RATE)
        channels = self.ctx.config.get_int("channels", 1)
        ok, message = self.ctx.devices.validate_input(device, rate, channels)
        if not ok:
            QMessageBox.warning(self, "Kayıt başlatılamadı", message)
            self._set_status("Seçili cihaz/hız kombinasyonu doğrulanamadı.", "error")
            return

        self._show_view("studio")
        self.studio.stop_monitor()
        seconds = self.ctx.config.get_int("countdown_sec", 0)
        if seconds > 0:
            self._begin_countdown(seconds)
        else:
            self._begin_recording()

    def _begin_countdown(self, seconds: int) -> None:
        self._countdown_active = True
        self._countdown_remaining = int(seconds)
        self._render_countdown()
        self._countdown_timer.start()
        self._set_status(f"Kayıt {seconds} saniye içinde başlayacak.", "warn")
        self._sync_controls()

    def _render_countdown(self) -> None:
        self.time_label.setText(format_duration(float(self._countdown_remaining)))
        self.time_sub.setText("GERİ SAYIM")

    def _countdown_tick(self) -> None:
        if not self._countdown_active:
            return
        self._countdown_remaining -= 1
        if self._countdown_remaining <= 0:
            self._countdown_timer.stop()
            self._countdown_active = False
            self._begin_recording()
        else:
            self._render_countdown()

    def _cancel_countdown(self) -> None:
        self._countdown_timer.stop()
        self._countdown_active = False
        self._set_status("Geri sayım iptal edildi.", "warn")
        self._set_time(0.0, 0.0)
        self._sync_controls()

    def _begin_recording(self) -> None:
        if self.ctx.recorder.is_active:
            return
        label = self.studio.label_edit.text().strip()
        try:
            path = self.ctx.new_recording_path(label)
            cfg = self.ctx.build_recording_config(label)
            cfg.path = path
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Kayıt hazırlanamadı",
                f"Kayıt dosyası yolu hazırlanamadı:\n{exc}\n\n"
                "Kayıt klasörünün yazılabilir olduğunu kontrol edin.",
            )
            return

        self._show_view("studio")
        self.studio.stop_monitor()
        self.studio.waveform.clear("Kayıt başlıyor…")
        self.studio.clear_analysis()
        self.studio.duration_label.setText("—")
        self.studio._set_file_label("Kayıt hazırlanıyor…")

        try:
            self.ctx.recorder.start(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Kayıt başlatılamadı", str(exc))
            self._set_status("Kayıt başlatılamadı.", "error")
            self._sync_controls()
            return

        self._tick_count = 0
        self._ui_timer.start()
        self.studio.set_analysis_busy(False)
        self._set_time(0.0, 0.0)
        self._set_status(f"Kayıt başladı: {path.name}", "ok")
        self._update_badge()
        self._sync_controls()

    def _on_recorder_state(self, state: str) -> None:
        self._recorder_state = state
        self.studio.set_recording_state(state)
        self._update_badge()
        self._sync_controls()
        if state in ("finished", "cancelled", "error", "idle"):
            self._maybe_stop_ui_timer()

    def _on_recorder_error(self, message: str) -> None:
        self._set_status(message, "error")
        logger.error("Kayıt hatası: %s", message)

    def stop_recording(self) -> None:
        rec = self.ctx.recorder
        if not rec.is_active:
            return
        try:
            mins, maxs = rec.waveform_view(2000)
        except Exception:
            mins, maxs = None, None

        try:
            result = rec.stop()
        except Exception as exc:
            QMessageBox.critical(
                self, "Kayıt durdurulamadı", f"Kayıt güvenli biçimde kapatılamadı:\n{exc}"
            )
            self._set_status("Kayıt durdurulamadı.", "error")
            return

        self.ctx.last_recording = result
        if result.path:
            self._current_path = result.path
            self.ctx.config.add_recent_recording(result.path)
            self.studio.show_recording_result(result, mins, maxs)
            self.studio.refresh_recent()
            self.projects.refresh_recent()
            try:
                info = self.ctx.load_for_playback(result.path)
                self.ctx.config.set("recent_recordings", self.ctx.config.get("recent_recordings", []))
                self._set_time(0.0, info.duration)
            except Exception as exc:
                self._set_status(
                    f"Kayıt kaydedildi ancak oynatma için açılamadı: {exc}", "error"
                )
            self._start_analysis(result.path)
            self._set_status(
                f"Kayıt tamamlandı: {Path(result.path).name} · {format_duration(result.duration)}",
                "ok",
            )
            if result.clip_count:
                self._set_status(
                    f"Kayıt tamamlandı ancak {result.clip_count} örnek kırpıldı. "
                    "Kazancı düşürüp yeniden kayıt önerilir.",
                    "warn",
                )
            if result.overflow_frames:
                QMessageBox.warning(
                    self,
                    "Kayıt tamponu taştı",
                    f"Kayıt sırasında {result.overflow_frames} frame tampona sığmadı. "
                    "Disk yavaş olabilir; kayıtta kısa boşluklar oluşmuş olabilir.\n\n"
                    "Dosya yine de kaydedildi. Farklı bir kayıt klasörü (ör. SSD) deneyin.",
                )
            if rec.error_message:
                QMessageBox.warning(
                    self, "Kayıt uyarısı", f"Kayıt sırasında sorun oluştu:\n{rec.error_message}"
                )
        else:
            detail = rec.error_message or "Bilinmeyen hata."
            QMessageBox.critical(
                self,
                "Kayıt tamamlanamadı",
                f"Kayıt dosyası oluşturulamadı.\n\n{detail}\n\n"
                f"Geçici dosya şurada olabilir: {self.ctx.recordings_dir()}",
            )
            self._set_status("Kayıt dosyası oluşturulamadı.", "error")

        self._sync_controls()
        self._maybe_stop_ui_timer()

    def cancel_recording(self) -> None:
        if not self.ctx.recorder.is_active:
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Kaydı iptal et")
        box.setText("Süren kayıt iptal edilsin mi?")
        box.setInformativeText(
            "Bu kayıt silinecek ve kurtarılamayacak. Daha önce tamamlanan kayıtlar etkilenmez."
        )
        cancel_btn = box.addButton("Kaydı sil", QMessageBox.ButtonRole.DestructiveRole)
        keep_btn = box.addButton("Kayda devam et", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep_btn)
        box.exec()
        if box.clickedButton() is not cancel_btn:
            return

        self.ctx.recorder.cancel()
        self.studio.waveform.clear("Kayıt iptal edildi")
        self.studio.duration_label.setText("—")
        self.studio.clear_analysis()
        self._set_time(0.0, 0.0)
        self._set_status("Kayıt iptal edildi; yarım dosya silindi.", "warn")
        self._sync_controls()

    # ================================================================ oynatma
    def toggle_playback(self) -> None:
        if self._countdown_active or self.ctx.recorder.is_active:
            return
        player = self.ctx.player
        if player.state is PlaybackState.PLAYING:
            player.pause()
            self._set_time(player.position, player.duration)
            self._set_status("Oynatma duraklatıldı.")
            return
        if player.state is PlaybackState.PAUSED:
            player.resume()
            self._ui_timer.start()
            self._set_status("Oynatma devam ediyor.")
            return
        if not self._current_path:
            self._set_status("Oynatılacak dosya yok. Bir kayıt seçin.", "warn")
            return
        if player.info is None:
            try:
                self.ctx.load_for_playback(self._current_path)
            except Exception as exc:
                QMessageBox.critical(self, "Dosya açılamadı", str(exc))
                return
        try:
            start = player.position
            if player.info is not None and start >= player.info.duration - 1e-6:
                start = 0.0
            player.play(start)
        except Exception as exc:
            QMessageBox.critical(self, "Oynatılamadı", str(exc))
            return
        self._ui_timer.start()
        self._set_status(f"Oynatılıyor: {Path(self._current_path).name}")

    def stop_transport(self) -> None:
        if self._countdown_active:
            self._cancel_countdown()
            return
        if self.ctx.recorder.is_active:
            self.stop_recording()
            return
        player = self.ctx.player
        if player.state is not PlaybackState.IDLE:
            player.stop()
            player.seek(0.0)
            self.studio.set_playback_position(0.0)
            self._set_time(0.0, player.duration)
            self._set_status("Oynatma durduruldu.")
        self._sync_controls()
        self._maybe_stop_ui_timer()

    def seek_playback(self, seconds: float) -> None:
        if self.ctx.recorder.is_active:
            return
        player = self.ctx.player
        if player.info is None:
            return
        player.seek(seconds)
        self.studio.set_playback_position(player.position)
        self._set_time(player.position, player.duration)
        if player.state is PlaybackState.PLAYING:
            self._ui_timer.start()

    def _on_player_state(self, state: str) -> None:
        self._player_state = state
        self.studio.playback_state_changed(state)
        if state == "playing":
            self._ui_timer.start()
        elif state == "paused":
            self._set_time(self.ctx.player.position, self.ctx.player.duration)
        elif state == "idle":
            if self.ctx.current_info is not None:
                self._set_time(self.ctx.player.position, self.ctx.current_info.duration)
            self._maybe_stop_ui_timer()
        self._sync_controls()

    def _on_player_error(self, message: str) -> None:
        self._set_status(message, "error")

    # ================================================================ dosyalar
    def open_file(self, path: str, autoplay: bool = False) -> None:
        if self.ctx.recorder.is_active:
            QMessageBox.information(
                self,
                "Kayıt sürüyor",
                "Kayıt sürerken başka bir dosya açılmaz. Önce kaydı bitirin.",
            )
            return
        self._show_view("studio")
        self.ctx.player.stop()
        try:
            info = self.ctx.load_for_playback(path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Dosya açılamadı",
                f"{exc}\n\nDosyanın bozuk olmadığını ve desteklenen bir biçimde "
                "olduğunu kontrol edin.",
            )
            self._set_status("Dosya açılamadı.", "error")
            return

        self._current_path = info.path
        self.studio.show_file(info)
        self.studio.clear_analysis()
        self._set_time(0.0, info.duration)

    def _open_in_restoration_lab(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        self.processing.load_file(p)
        self._show_view("processing")
        self._set_status(f"Kayıt AI Restorasyon Laboratuvarına aktarıldı: {p.name}")
        self._sync_controls()

    def _open_and_analyze(self, path: str) -> None:
        self.open_file(path)

    def _start_peaks(self, path: str) -> None:
        worker = PeaksWorker(Path(path), self)
        worker.done.connect(self._on_peaks_done)
        worker.failed.connect(self._on_peaks_failed)
        self._start_worker(worker)

    def _on_peaks_done(self, path: str, payload) -> None:
        if path != self._current_path:
            return
        mins, maxs, duration = payload
        self.studio.set_waveform_peaks(mins, maxs, duration)

    def _on_peaks_failed(self, message: str) -> None:
        self.studio.waveform.clear("Waveform oluşturulamadı")
        self._set_status(f"Waveform oluşturulamadı — {message}", "warn")

    # ================================================================ analiz
    def _start_analysis(self, path: str) -> None:
        if not path:
            return
        if self._analysis_worker is not None:
            if str(self._analysis_worker.path) == str(path):
                return
            self._set_status("Başka bir dosya analiz ediliyor; bitmesini bekleyin.", "warn")
            return
        worker = AnalysisWorker(Path(path), self)
        worker.done.connect(self._on_analysis_done)
        worker.failed.connect(self._on_analysis_failed)
        worker.finished.connect(lambda w=worker: self._clear_analysis_worker(w))
        self._analysis_worker = worker
        self.studio.set_analysis_busy(True, str(path))
        self._start_worker(worker)

    def _clear_analysis_worker(self, worker: AnalysisWorker) -> None:
        if self._analysis_worker is worker:
            self._analysis_worker = None
        self.studio.analyze_btn.setEnabled(self.studio.current_path is not None)

    def _on_analysis_done(self, path: str, analysis) -> None:
        if self._current_path and str(Path(path)) != str(Path(self._current_path)):
            return
        self.studio.show_analysis(analysis)
        score = analysis.score if analysis.score is not None else "—"
        self._set_status(f"Analiz tamamlandı: {Path(path).name} · puan {score}/100", "ok")

    def _on_analysis_failed(self, message: str) -> None:
        self.studio.show_analysis_error(message)
        self._set_status("Analiz başarısız — ayrıntı analiz panelinde.", "error")

    # ================================================================ işçiler
    def _start_worker(self, worker) -> None:
        self._workers.add(worker)
        worker.finished.connect(lambda w=worker: self._finish_worker(w))
        worker.start()

    def _finish_worker(self, worker) -> None:
        self._workers.discard(worker)
        worker.deleteLater()

    # ================================================================ dosya silme
    def _on_file_deleted(self, path: str) -> None:
        remaining = [
            p for p in (self.ctx.config.get("recent_recordings", []) or []) if p != str(path)
        ]
        self.ctx.config.set("recent_recordings", remaining)
        self.studio.refresh_recent()
        self.projects.refresh_recent()

        if self._current_path == str(path):
            self.ctx.player.stop()
            self.ctx.current_info = None
            self._current_path = None
            self.studio.waveform.clear("Dosya silindi")
            self.studio.duration_label.setText("—")
            self.studio._set_file_label("Kayıt bekleniyor")
            self.studio.clear_analysis()
            self._set_time(0.0, 0.0)
            self._set_status(f"Silindi: {Path(path).name}", "warn")
        self._sync_controls()

    def _on_recordings_changed(self) -> None:
        self.studio.refresh_recent()
        self.projects.refresh_recent()
        if self._current_view == "library":
            self.library.refresh()

    # ================================================================ zamanlayıcı
    def _tick(self) -> None:
        recorder = self.ctx.recorder
        if recorder.is_active:
            duration = recorder.duration
            self._set_time(duration, duration, mode="record")
            snap = recorder.level_snapshot()
            self.studio.show_level(snap)
            self.meter.set_level(snap.peak_db, snap.hold_db, snap.clip_count > 0)
            self._tick_count += 1
            if self._tick_count % 6 == 0:
                mins, maxs = recorder.waveform_view(1600)
                if maxs.size:
                    self.studio.show_recording_waveform(mins, maxs, duration)
            return

        if self.ctx.monitor.active:
            snap = self.ctx.monitor.snapshot()
            self.studio.show_level(snap)
            self.meter.set_level(snap.peak_db, snap.hold_db, snap.clip_count > 0)
            return

        if self.ctx.player.state is PlaybackState.PLAYING:
            position = self.ctx.player.position
            self._set_time(position, self.ctx.player.duration, mode="play")
            self.studio.set_playback_position(position)
            return

        self._maybe_stop_ui_timer()

    def _set_time(self, position: float, duration: float, mode: str = "idle") -> None:
        self.time_label.setText(format_duration(max(0.0, position)))
        if mode == "record":
            self.time_sub.setText(f"KAYIT · {format_duration(max(0.0, duration))}")
        elif mode == "play":
            remaining = max(0.0, duration - position)
            self.time_sub.setText(f"Kalan {format_duration(remaining)}")
        elif duration > 0:
            self.time_sub.setText(f"TOPLAM {format_duration(duration)}")
        else:
            self.time_sub.setText("HAZIR")

    def _maybe_stop_ui_timer(self) -> None:
        if self.ctx.recorder.is_active or self.ctx.monitor.active:
            return
        if self.ctx.player.state is not PlaybackState.IDLE:
            return
        self._ui_timer.stop()

    def _update_cpu(self) -> None:
        now_wall = time.monotonic()
        now_cpu = time.process_time()
        elapsed = max(1e-6, now_wall - self._last_wall_time)
        used = max(0.0, now_cpu - self._last_cpu_time)
        percent = used / elapsed * 100.0
        self._last_wall_time = now_wall
        self._last_cpu_time = now_cpu
        self.cpu_label.setText(f"CPU {percent:.0f}%")

    # ================================================================ kapanış
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._countdown_active:
            self._countdown_timer.stop()
            self._countdown_active = False

        if self.ctx.recorder.is_active:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Kayıt sürüyor")
            box.setText("Kayıt sürüyor. Durdurup kaydederek çıkmak ister misiniz?")
            box.setInformativeText(
                "Kaydedilen bölüm kalıcı dosyaya taşınır. Vazgeçerseniz uygulama açık kalır."
            )
            save_btn = box.addButton("Kaydı bitir ve çık", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Vazgeç", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is not save_btn:
                event.ignore()
                return
            try:
                result = self.ctx.recorder.stop()
                if result.path:
                    self.ctx.config.add_recent_recording(result.path)
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Kayıt kapatılamadı",
                    f"Kayıt güvenli biçimde kapatılamadı:\n{exc}\n\n"
                    "Uygulama kapatılmadı; veri kaybını önlemek için tekrar deneyin.",
                )
                event.ignore()
                return

        if hasattr(self.ctx, "task_manager") and self.ctx.task_manager.active_tasks():
            active = self.ctx.task_manager.active_tasks()
            names = ", ".join(t.title for t in active[:3])
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Aktif arka plan görevi sürüyor")
            box.setText(f"{len(active)} arka plan görevi hâlâ çalışıyor ({names}).")
            box.setInformativeText("Görevleri iptal edip çıkmak ister misiniz?")
            quit_btn = box.addButton("Görevleri iptal et ve çık", QMessageBox.ButtonRole.DestructiveRole)
            stay_btn = box.addButton("Açık kal", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(stay_btn)
            box.exec()
            if box.clickedButton() is not quit_btn:
                event.ignore()
                return
            for t in active:
                self.ctx.task_manager.cancel_task(t.id)

        self.studio.stop_monitor()
        self._ui_timer.stop()
        self._cpu_timer.stop()

        if hasattr(self, "processing") and hasattr(self.processing, "ab_player"):
            try:
                self.processing.ab_player.stop()
            except Exception:
                pass

        if hasattr(self, "library"):
            try:
                self.library.stop_scan()
            except Exception:
                logger.exception("Kütüphane taraması beklenirken hata")

        pending = list(self._workers)
        for worker in pending:
            try:
                worker.wait(5000)
            except RuntimeError:
                continue
        still_running = [w for w in pending if w.isRunning()]
        if still_running:
            QMessageBox.information(
                self,
                "Arka plan işlemi sürüyor",
                "Analiz veya tarama hâlâ sürüyor. Veri kaybı olmaması için uygulama "
                "kapanmadı; işlem bitince tekrar deneyin.",
            )
            event.ignore()
            return

        self.ctx.shutdown()
        event.accept()

    def shutdown_workers(self) -> None:
        for worker in list(self._workers):
            try:
                worker.wait(2000)
            except RuntimeError:
                continue
