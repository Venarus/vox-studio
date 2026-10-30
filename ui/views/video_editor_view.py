"""
VOX Studio — CapCut Kolaylığında, Profesyonel Video Düzenleme, Dublaj ve AI Ses Ayrıştırma İş İstasyonu
Bölüm 2, 4, 5, 6, 7, 11, 12, 15 ve 16 kapsamında:
- CapCut tarzı Üst Araç Çubuğu (Medya, Ses, Metin & Altyazı, AI Araçları, Efekt & Geçiş, Kayıt, Tanılama, Dışa Aktar)
- Seçilen kategoriye göre değişen bağlama duyarlı Sol İçerik Paneli
- Orta Video Önizleme, Sağ Proje Detayları / Klip Özellikleri (Inspector) ve Alt Akıllı Timeline
- Kaynak kliple tam zaman senkronlu AI Ses Ayrıştırma, Otomatik Konuşma/Sessizlik Tespiti ve Canlı Dublaj.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ai.exporter import StemExporter
from ai.media_extractor import MediaExtractor, MediaInfo, parse_subtitles
from ai.stem_separator import ModelCatalog, StemEnhancer, StemSeparationResult, StemSeparator
from ai.transcription import WhisperTranscriber, detect_speech_and_silence
from core import constants as C
from core.diagnostics import SystemDiagnostics
from core.format import format_duration, format_file_size
from core.logger import get_logger
from ui import icons
from ui.context import AppContext
from ui.widgets.clip_inspector import ClipInspectorWidget
from ui.widgets.multi_track_timeline import MultiTrackTimeline
from ui.widgets.shortcut_guide import ShortcutGuideDialog
from ui.widgets.toast import ToastNotification
from ui.widgets.video_preview import VideoPreviewWidget
from ui.workers import classify_media_category
from video_editor.engine import TimelineEngine
from video_editor.playback import PlaybackController
from video_editor.models import ClipType, TimelineClip, TimelineProject, TrackType
from video_editor.renderer import RenderSettings, VideoRenderer

logger = get_logger("ui.views.editor")


class DiagnosticsAndTasksDialog(QDialog):
    """Bölüm 15.6, 15.8 ve 15.9: Sistem Tanılama, Görev Yöneticisi ve Depolama/Önbellek Kontrolü."""

    def __init__(self, ctx: AppContext, editor: "VideoEditorView", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx
        self.editor = editor
        self.setWindowTitle("Sistem Tanılama, Görev Yöneticisi ve Depolama Kontrolü")
        self.resize(840, 560)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        tabs = QTabWidget()

        # 1. Sistem Tanılama Sekmesi
        diag_tab = QWidget()
        d_lay = QVBoxLayout(diag_tab)
        self.diag_table = QTableWidget(0, 4)
        self.diag_table.setHorizontalHeaderLabels(["Bileşen", "Durum", "Özet", "Neden / Çözüm Önerisi"])
        self.diag_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.diag_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        d_lay.addWidget(self.diag_table, 1)

        refresh_diag_btn = QPushButton(" Tanılama Kontrollerini Yenile")
        refresh_diag_btn.setIcon(icons.icon("refresh", size=14))
        refresh_diag_btn.clicked.connect(self._populate_diagnostics)
        d_lay.addWidget(refresh_diag_btn)
        tabs.addTab(diag_tab, "1. Sistem Tanılama (Donanım & Modeller)")

        # 2. Görev Yöneticisi Sekmesi
        task_tab = QWidget()
        t_lay = QVBoxLayout(task_tab)
        self.task_table = QTableWidget(0, 5)
        self.task_table.setHorizontalHeaderLabels(["Görev", "Kategori", "Durum", "İlerleme", "Süre / Mesaj"])
        self.task_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.task_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        t_lay.addWidget(self.task_table, 1)
        tabs.addTab(task_tab, "2. Arka Plan Görev Yöneticisi")

        # 3. Depolama, Önbellek ve Eksik Medya Sekmesi
        store_tab = QWidget()
        s_lay = QVBoxLayout(store_tab)
        self.storage_lbl = QLabel("")
        self.storage_lbl.setWordWrap(True)
        self.storage_lbl.setStyleSheet("color: #F2F4F8; font-size: 12px;")
        s_lay.addWidget(self.storage_lbl)

        self.missing_lbl = QLabel("")
        self.missing_lbl.setWordWrap(True)
        self.missing_lbl.setStyleSheet("color: #FBBF24; font-size: 12px;")
        s_lay.addWidget(self.missing_lbl)

        s_btn_row = QHBoxLayout()
        clr_cache_btn = QPushButton(" Önbelleği Güvenle Temizle (Thumbnails & Geçici Sesler)")
        clr_cache_btn.setIcon(icons.icon("trash", size=14))
        clr_cache_btn.clicked.connect(self._clear_cache)
        s_btn_row.addWidget(clr_cache_btn)
        s_btn_row.addStretch(1)
        s_lay.addLayout(s_btn_row)
        s_lay.addStretch(1)
        tabs.addTab(store_tab, "3. Depolama & Eksik Medya")

        # 4. Günlükler (Logs)
        log_tab = QWidget()
        l_lay = QVBoxLayout(log_tab)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        l_lay.addWidget(self.log_view, 1)
        tabs.addTab(log_tab, "4. Uygulama Günlükleri (Logs)")

        root.addWidget(tabs, 1)

        close_btn = QPushButton("Kapat")
        close_btn.clicked.connect(self.accept)
        root.addWidget(close_btn)

        self._populate_diagnostics()
        self._populate_tasks()
        self._populate_storage()
        self.log_view.setPlainText(SystemDiagnostics.read_recent_logs(60))

    def _populate_diagnostics(self) -> None:
        checks = SystemDiagnostics.run_all_checks(self.ctx)
        self.diag_table.setRowCount(0)
        for item in checks:
            r = self.diag_table.rowCount()
            self.diag_table.insertRow(r)
            self.diag_table.setItem(r, 0, QTableWidgetItem(item.component))
            st_txt = "HAZIR" if item.status == "ok" else ("UYARI" if item.status == "warning" else "HATA")
            self.diag_table.setItem(r, 1, QTableWidgetItem(st_txt))
            self.diag_table.setItem(r, 2, QTableWidgetItem(item.summary))
            note = " ".join(p for p in [item.cause, item.remedy] if p) or "Sorun yok."
            self.diag_table.setItem(r, 3, QTableWidgetItem(note))

    def _populate_tasks(self) -> None:
        tasks = self.ctx.task_manager.list_tasks()
        self.task_table.setRowCount(0)
        for t in tasks:
            r = self.task_table.rowCount()
            self.task_table.insertRow(r)
            self.task_table.setItem(r, 0, QTableWidgetItem(t.title))
            self.task_table.setItem(r, 1, QTableWidgetItem(t.category.upper()))
            self.task_table.setItem(r, 2, QTableWidgetItem(t.status.upper()))
            self.task_table.setItem(r, 3, QTableWidgetItem(f"%{t.progress}"))
            self.task_table.setItem(r, 4, QTableWidgetItem(f"{t.elapsed_sec:.1f}s · {t.message}"))

    def _populate_storage(self) -> None:
        usage = SystemDiagnostics.storage_usage()
        self.storage_lbl.setText(
            f"Küçük Resim (Thumbnail) Önbelleği: {format_file_size(usage['thumbnails_bytes'])}\n"
            f"Çözülmüş Geçici Ses Önbelleği: {format_file_size(usage['decoded_audio_bytes'])}\n"
            f"Toplam Önbellek Boyutu: {format_file_size(usage['total_cache_bytes'])}"
        )
        missing = self.ctx.project_manager.find_missing_media(
            self.editor.engine.project.to_dict(),
            list(self.editor._bin_entries.keys()),
        )
        if missing:
            self.missing_lbl.setText("Eksik Medya Dosyaları:\n" + "\n".join(f"• {m}" for m in missing))
        else:
            self.missing_lbl.setText("Projedeki tüm medya bağlantıları doğrulandı (eksik medya yok).")

    def _clear_cache(self) -> None:
        removed = SystemDiagnostics.clear_cache_safely()
        self._populate_storage()
        QMessageBox.information(
            self, "Önbellek Temizlendi", f"{format_file_size(removed)} geçici önbellek dosyası güvenli şekilde silindi."
        )


class RenderWorker(QThread):
    progressSignal = Signal(float, str)
    finishedSignal = Signal(str)
    errorSignal = Signal(str)

    def __init__(self, renderer: VideoRenderer):
        super().__init__()
        self.renderer = renderer

    def run(self):
        try:
            out_file = self.renderer.render(
                progress_cb=lambda p, msg: self.progressSignal.emit(p, msg)
            )
            self.finishedSignal.emit(out_file)
        except Exception as e:
            logger.exception("Render iş parçacığı hatası")
            self.errorSignal.emit(str(e))


class EditorStemWorker(QThread):
    """Video editörü içinden çağrılan arka plan AI stem ayrıştırma işçisi (arayüzü dondurmaz)."""

    progressSignal = Signal(int, str)
    finishedSignal = Signal(object, dict)
    errorSignal = Signal(str)

    def __init__(self, media_path: str, out_dir: Path, model_id: str = "vox-dubbing-dialogue"):
        super().__init__()
        self.media_path = media_path
        self.out_dir = out_dir
        self.model_id = model_id
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            self.progressSignal.emit(10, "Video ses akışı çözülüyor...")
            audio_arr, fs = MediaExtractor.extract_audio(
                self.media_path, target_sample_rate=48000, target_channels=2
            )
            if self._cancelled:
                raise RuntimeError("İşlem kullanıcı tarafından iptal edildi.")

            self.progressSignal.emit(30, "AI modeli ile diyalog, müzik ve ortam ayrıştırılıyor...")
            sep = StemSeparator(model_id=self.model_id)

            def _prog(p: float, msg: str) -> None:
                if self._cancelled:
                    raise RuntimeError("İşlem kullanıcı tarafından iptal edildi.")
                self.progressSignal.emit(30 + int(p * 50), msg)

            res = sep.separate(
                audio_arr,
                fs,
                progress_cb=_prog,
                cancel_check=lambda: self._cancelled,
            )
            if self._cancelled:
                raise RuntimeError("İşlem kullanıcı tarafından iptal edildi.")

            self.progressSignal.emit(85, "Ayrıştırılmış stem dosyaları doğrulanarak diske yazılıyor...")
            report = StemExporter.export_all_stems_detailed(
                res.stems,
                res.sample_rate,
                self.out_dir,
                project_name=Path(self.media_path).stem,
                format_type="wav",
                create_zip=False,
                overwrite=True,
                model_name=self.model_id,
            )
            stem_paths = {k: str(v) for k, v in report.succeeded.items()}
            self.progressSignal.emit(100, "Tamamlandı.")
            self.finishedSignal.emit(res, stem_paths)
        except Exception as exc:
            logger.error(f"EditorStemWorker hatası: {exc}")
            self.errorSignal.emit(str(exc))


class VideoEditorView(QWidget):
    """CapCut Kolaylığında Ana Video Düzenleme, Dublaj ve AI Ses Ayrıştırma Çalışma İstasyonu."""

    homeRequested = Signal()

    CATEGORY_FILTERS = [
        ("all", "Tümü"),
        ("video", "Videolar"),
        ("audio", "Ses && Müzik"),
        ("recording", "Seslendirme / Kayıtlar"),
        ("image", "Görseller"),
        ("subtitle", "Altyazılar"),
        ("stem", "AI && Stem'ler"),
    ]

    TOP_TOOL_TABS = [
        ("media", "Medya", "video"),
        ("audio", "Ses", "music"),
        ("text", "Metin && Altyazı", "subtitle"),
        ("ai", "AI Araçları", "sparkles"),
        ("effects", "Efekt && Geçiş", "sliders"),
    ]

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx
        self.engine = TimelineEngine()
        self.playback_controller = PlaybackController(engine=self.engine)
        self._is_dubbing_recording = False
        self._recording_start_t = 0.0
        self._current_rec_path: Optional[Path] = None
        self.render_worker: Optional[RenderWorker] = None
        self.stem_worker: Optional[EditorStemWorker] = None
        self._bin_entries: Dict[str, dict] = {}
        self._is_dirty: bool = False
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._build_ui()
        self._connect_signals()

        # Otomatik Kayıt Zamanlayıcısı (30 saniye)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(30_000)
        self._autosave_timer.timeout.connect(self._auto_save_tick)
        self._autosave_timer.start()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        # ------------------------------------------------------------- SADELEŞTİRİLMİŞ 3 BÖLGELİ ÜST ARAÇ ÇUBUĞU (Bölüm 5)
        top_bar = QFrame()
        top_bar.setObjectName("Card")
        top_bar.setFixedHeight(48)
        tb_layout = QHBoxLayout(top_bar)
        tb_layout.setContentsMargins(10, 4, 10, 4)
        tb_layout.setSpacing(6)

        # 1) SOL GRUP: Ana Sayfa + Proje Adı + Kaydedilme Durumu
        self.home_btn = QPushButton(" Ana Sayfa")
        self.home_btn.setProperty("compact", "true")
        self.home_btn.setIcon(icons.icon("home", size=14))
        self.home_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.home_btn.setToolTip("Ana Sayfa / Proje Panosuna Dön")
        self.home_btn.clicked.connect(self.homeRequested.emit)
        tb_layout.addWidget(self.home_btn)

        self.proj_title_edit = QLineEdit(self.engine.project.name)
        self.proj_title_edit.setFixedWidth(145)
        self.proj_title_edit.setToolTip("Aktif proje / sekans adı")
        self.proj_title_edit.setStyleSheet("font-weight: 600; font-size: 12px;")
        self.proj_title_edit.textChanged.connect(self._on_project_title_edited)
        tb_layout.addWidget(self.proj_title_edit)

        self.save_status_lbl = QLabel("● Kaydedildi")
        self.save_status_lbl.setStyleSheet("color: #10B981; font-size: 11px; font-weight: 600;")
        self.save_status_lbl.setToolTip("Proje kayıt durumu")
        tb_layout.addWidget(self.save_status_lbl)

        sep_left = QFrame()
        sep_left.setFrameShape(QFrame.Shape.VLine)
        sep_left.setStyleSheet("color: #28292E;")
        tb_layout.addWidget(sep_left)

        # 2) ORTA GRUP: Yalnızca Temel Düzenleme Araçları (Geri Al, Yinele, Seçim V, Kes B, Böl, Sil, Snap, İşaretçi)
        self.top_undo_btn = QPushButton()
        self.top_undo_btn.setProperty("compact", "true")
        self.top_undo_btn.setIcon(icons.icon("undo", size=14))
        self.top_undo_btn.setFixedSize(28, 28)
        self.top_undo_btn.setToolTip("Geri Al (Ctrl+Z)")
        self.top_undo_btn.clicked.connect(self.engine.undo)
        tb_layout.addWidget(self.top_undo_btn)

        self.top_redo_btn = QPushButton()
        self.top_redo_btn.setProperty("compact", "true")
        self.top_redo_btn.setIcon(icons.icon("redo", size=14))
        self.top_redo_btn.setFixedSize(28, 28)
        self.top_redo_btn.setToolTip("Yinele (Ctrl+Y)")
        self.top_redo_btn.clicked.connect(self.engine.redo)
        tb_layout.addWidget(self.top_redo_btn)

        self.top_select_tool_btn = QPushButton(" Seçim (V)")
        self.top_select_tool_btn.setProperty("compact", "true")
        self.top_select_tool_btn.setCheckable(True)
        self.top_select_tool_btn.setChecked(True)
        self.top_select_tool_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.top_select_tool_btn.setToolTip("Seçim Aracı (Kısayol: V)")
        self.top_select_tool_btn.clicked.connect(lambda: self.set_editor_tool_mode("select"))
        tb_layout.addWidget(self.top_select_tool_btn)

        self.top_blade_tool_btn = QPushButton(" Kes (B)")
        self.top_blade_tool_btn.setProperty("compact", "true")
        self.top_blade_tool_btn.setCheckable(True)
        self.top_blade_tool_btn.setChecked(False)
        self.top_blade_tool_btn.setIcon(icons.icon("scissors", size=13, color="#E53945"))
        self.top_blade_tool_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.top_blade_tool_btn.setToolTip(
            "Kesme / Bölme Aracı (Kısayol: B) — Kliplerin üzerine tıklayarak istediğiniz noktadan bölün"
        )
        self.top_blade_tool_btn.clicked.connect(lambda: self.set_editor_tool_mode("blade"))
        tb_layout.addWidget(self.top_blade_tool_btn)

        self.top_split_btn = QPushButton(" Böl")
        self.top_split_btn.setProperty("compact", "true")
        self.top_split_btn.setIcon(icons.icon("scissors", size=13))
        self.top_split_btn.setToolTip("İmleç (Playhead) konumundan klibi böl (Ctrl+K)")
        self.top_split_btn.clicked.connect(lambda: self.timeline_widget._split_selected())
        tb_layout.addWidget(self.top_split_btn)

        self.top_delete_btn = QPushButton(" Sil")
        self.top_delete_btn.setProperty("compact", "true")
        self.top_delete_btn.setIcon(icons.icon("trash", size=13))
        self.top_delete_btn.setToolTip("Seçili klibi sil (Delete)")
        self.top_delete_btn.clicked.connect(lambda: self.timeline_widget._delete_selected())
        tb_layout.addWidget(self.top_delete_btn)

        self.top_snap_btn = QPushButton(" Snap")
        self.top_snap_btn.setProperty("compact", "true")
        self.top_snap_btn.setCheckable(True)
        self.top_snap_btn.setChecked(self.engine.magnetic_snap_enabled)
        self.top_snap_btn.setIcon(icons.icon("magnet", size=13))
        self.top_snap_btn.setToolTip("Mıknatıslı hizalama (Snap) aç/kapat")
        self.top_snap_btn.toggled.connect(self._on_top_snap_toggled)
        tb_layout.addWidget(self.top_snap_btn)

        self.top_marker_btn = QPushButton(" İşaretçi")
        self.top_marker_btn.setProperty("compact", "true")
        self.top_marker_btn.setIcon(icons.icon("bookmark", size=13))
        self.top_marker_btn.setToolTip("İmleç konumuna işaretleyici ekle (M)")
        self.top_marker_btn.clicked.connect(
            lambda: self.timeline_widget.canvas._add_marker(self.engine.project.playhead_pos)
        )
        tb_layout.addWidget(self.top_marker_btn)

        tb_layout.addStretch(1)

        # 3) SAĞ GRUP: Önizleme Kalitesi + Kaydet + Proje Ayarları/Tanılama + Dışa Aktar
        self.preview_quality_combo = QComboBox()
        self.preview_quality_combo.addItem("Önizleme: Tam (100%)", "full")
        self.preview_quality_combo.addItem("Önizleme: Yarım (1/2)", "half")
        self.preview_quality_combo.addItem("Önizleme: Çeyrek (1/4)", "quarter")
        self.preview_quality_combo.setToolTip("Video önizleme çözünürlük/performans ölçeği")
        tb_layout.addWidget(self.preview_quality_combo)

        self.save_proj_btn = QPushButton(" Kaydet")
        self.save_proj_btn.setProperty("compact", "true")
        self.save_proj_btn.setIcon(icons.icon("save", size=13))
        self.save_proj_btn.setToolTip("Proje ve timeline durumunu kaydet (Ctrl+S)")
        self.save_proj_btn.clicked.connect(self.save_project_state)
        tb_layout.addWidget(self.save_proj_btn)

        self.diag_btn = QPushButton(" Proje Ayarları")
        self.diag_btn.setProperty("compact", "true")
        self.diag_btn.setIcon(icons.icon("activity", size=13))
        self.diag_btn.setToolTip(
            "Proje Ayarları, Sistem Tanılama, Arka Plan Görev Yöneticisi ve Önbellek Kontrolü"
        )
        self.diag_btn.clicked.connect(self._open_diagnostics_dialog)
        tb_layout.addWidget(self.diag_btn)

        self.export_btn = QPushButton(" Dışa Aktar")
        self.export_btn.setProperty("variant", "primary")
        self.export_btn.setProperty("compact", "true")
        self.export_btn.setIcon(icons.icon("download", size=13, color="#FFFFFF"))
        self.export_btn.setStyleSheet(
            f"background-color: {C.COLOR_ACCENT_PRIMARY}; color: #FFFFFF; font-weight: 700; "
            "border: none; border-radius: 6px; padding: 6px 14px;"
        )
        self.export_btn.setToolTip("Düzenlenen videoyu, dublajı veya ses miksajını dışa aktar")
        self.export_btn.clicked.connect(self._open_export_dialog)
        tb_layout.addWidget(self.export_btn)

        root.addWidget(top_bar)

        # ------------------------------------------------------------- ORTA BÖLÜM (3-PANE SPLITTER)
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 1. Sol: Bağlama Duyarlı İçerik Paneli (Medya Havuzu + Kategori Sekmeleri + Hızlı Aksiyonlar)
        media_card = QFrame()
        media_card.setObjectName("Card")
        media_card.setMinimumWidth(285)
        m_layout = QVBoxLayout(media_card)
        m_layout.setContentsMargins(10, 8, 10, 8)
        m_layout.setSpacing(6)

        # Sol İçerik Paneli Kategori Sekmeleri (Medya, Ses, Metin & Altyazı, AI Araçları, Efekt & Geçiş)
        tabs_row = QHBoxLayout()
        tabs_row.setSpacing(3)
        self.tool_tab_group = QButtonGroup(self)
        self.tool_tab_group.setExclusive(True)
        self.tool_tab_buttons: Dict[str, QPushButton] = {}
        for idx, (tab_key, tab_label, tab_icon) in enumerate(self.TOP_TOOL_TABS):
            btn = QPushButton(f" {tab_label}")
            btn.setProperty("compact", "true")
            btn.setCheckable(True)
            btn.setIcon(icons.icon(tab_icon, size=12))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            if idx == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda _, k=tab_key: self._switch_top_tool_tab(k))
            self.tool_tab_group.addButton(btn, idx)
            self.tool_tab_buttons[tab_key] = btn
            tabs_row.addWidget(btn)
        m_layout.addLayout(tabs_row)

        # Hızlı Aksiyon Çubuğu (İçe Aktar / Ses Ayrıştır / Dublaj)
        quick_act_row = QHBoxLayout()
        quick_act_row.setSpacing(4)
        self.import_btn = QPushButton(" İçe Aktar")
        self.import_btn.setProperty("compact", "true")
        self.import_btn.setIcon(icons.icon("upload", size=12))
        self.import_btn.setToolTip("Video, Ses, Görsel veya Altyazı dosyalarını içe aktar")
        self.import_btn.clicked.connect(self._import_media)
        quick_act_row.addWidget(self.import_btn)

        self.stem_btn = QPushButton(" Ses Ayrıştır")
        self.stem_btn.setProperty("compact", "true")
        self.stem_btn.setIcon(icons.icon("sparkles", size=12))
        self.stem_btn.setToolTip("Seçili klibin sesini AI ile Diyalog, Müzik ve Ortam katmanlarına ayır")
        self.stem_btn.clicked.connect(self._on_quick_stem_separation)
        quick_act_row.addWidget(self.stem_btn)

        self.dub_btn = QPushButton(" Dublaj")
        self.dub_btn.setProperty("variant", "primary")
        self.dub_btn.setProperty("compact", "true")
        self.dub_btn.setIcon(icons.icon("record", size=12, color="#FFFFFF"))
        self.dub_btn.setToolTip("Videoyu oynatarak playhead konumundan gerçek mikrofonla dublaj kaydı al")
        self.dub_btn.clicked.connect(self._toggle_dubbing_record)
        quick_act_row.addWidget(self.dub_btn)
        m_layout.addLayout(quick_act_row)

        m_top = QHBoxLayout()
        self.drawer_title_label = QLabel("MEDYA HAVUZU")
        self.drawer_title_label.setObjectName("SectionTitle")
        m_top.addWidget(self.drawer_title_label)
        m_top.addStretch(1)
        self.category_combo = QComboBox()
        for cat_key, cat_label in self.CATEGORY_FILTERS:
            self.category_combo.addItem(cat_label.replace("&&", "&"), cat_key)
        self.category_combo.setMinimumWidth(120)
        self.category_combo.setToolTip("Medya türüne göre filtrele")
        self.category_combo.currentIndexChanged.connect(self._refresh_media_bin_view)
        m_top.addWidget(self.category_combo)
        m_layout.addLayout(m_top)

        # Bağlama Duyarlı Üst Araç Çekmecesi (Seçilen üst sekmeye göre değişir)
        self.context_drawer = QStackedWidget()
        self.context_drawer.setMaximumHeight(112)

        # Sayfa 0: Medya & Ses Hızlı İçe Aktarma Çekmecesi
        page_media = QWidget()
        pm_lay = QVBoxLayout(page_media)
        pm_lay.setContentsMargins(0, 0, 0, 0)
        pm_lay.setSpacing(4)
        self.bin_search_edit = QLineEdit()
        self.bin_search_edit.setPlaceholderText("Medya ara (ad veya biçim)…")
        self.bin_search_edit.setClearButtonEnabled(True)
        self.bin_search_edit.textChanged.connect(self._refresh_media_bin_view)
        pm_lay.addWidget(self.bin_search_edit)

        pm_btns = QHBoxLayout()
        pm_btns.setSpacing(5)
        imp_quick_btn = QPushButton(" Dosya Seç")
        imp_quick_btn.setProperty("compact", "true")
        imp_quick_btn.setIcon(icons.icon("upload", size=12))
        imp_quick_btn.clicked.connect(self._import_media)
        pm_btns.addWidget(imp_quick_btn)

        self.add_to_tl_btn = QPushButton(" Timeline'a Ekle")
        self.add_to_tl_btn.setProperty("compact", "true")
        self.add_to_tl_btn.setIcon(icons.icon("plus", size=12))
        self.add_to_tl_btn.clicked.connect(self._add_selected_bin_item_to_timeline)
        pm_btns.addWidget(self.add_to_tl_btn)
        pm_lay.addLayout(pm_btns)
        self.context_drawer.addWidget(page_media)

        # Sayfa 1: Metin & Altyazı Çekmecesi
        page_text = QWidget()
        pt_lay = QVBoxLayout(page_text)
        pt_lay.setContentsMargins(0, 0, 0, 0)
        pt_lay.setSpacing(4)
        self.quick_sub_edit = QLineEdit()
        self.quick_sub_edit.setPlaceholderText("Altyazı veya başlık metni yazın...")
        pt_lay.addWidget(self.quick_sub_edit)
        pt_btns = QHBoxLayout()
        self.add_sub_btn = QPushButton(" Altyazı Ekle")
        self.add_sub_btn.setProperty("compact", "true")
        self.add_sub_btn.setIcon(icons.icon("subtitle", size=12))
        self.add_sub_btn.clicked.connect(self._add_subtitle_clip)
        pt_btns.addWidget(self.add_sub_btn)

        exp_srt_quick_btn = QPushButton(" SRT İndir")
        exp_srt_quick_btn.setProperty("compact", "true")
        exp_srt_quick_btn.setIcon(icons.icon("download", size=12))
        exp_srt_quick_btn.clicked.connect(self._export_timeline_subtitles_srt)
        pt_btns.addWidget(exp_srt_quick_btn)
        pt_lay.addLayout(pt_btns)
        self.context_drawer.addWidget(page_text)

        # Sayfa 2: AI Araçları Çekmecesi (Bölüm 5.2 & 15.1)
        page_ai = QWidget()
        pai_lay = QVBoxLayout(page_ai)
        pai_lay.setContentsMargins(0, 0, 0, 0)
        pai_lay.setSpacing(4)
        self.editor_ai_model_combo = QComboBox()
        for m in ModelCatalog.get_available_models():
            if m.is_downloaded:
                self.editor_ai_model_combo.addItem(m.name, m.id)
        pai_lay.addWidget(self.editor_ai_model_combo)
        pai_btns = QHBoxLayout()
        run_ai_sep_btn = QPushButton(" Sesi Ayrıştır")
        run_ai_sep_btn.setProperty("variant", "primary")
        run_ai_sep_btn.setProperty("compact", "true")
        run_ai_sep_btn.setIcon(icons.icon("sparkles", size=12, color="#FFFFFF"))
        run_ai_sep_btn.clicked.connect(self._on_quick_stem_separation)
        pai_btns.addWidget(run_ai_sep_btn)

        vad_mark_btn = QPushButton(" Sessizlik İşaretle")
        vad_mark_btn.setProperty("compact", "true")
        vad_mark_btn.setToolTip("Seçili klipteki konuşma ve sessizlik sınırlarını timeline üzerinde işaretler")
        vad_mark_btn.clicked.connect(self._mark_clip_silences_on_timeline)
        pai_btns.addWidget(vad_mark_btn)
        pai_lay.addLayout(pai_btns)
        self.context_drawer.addWidget(page_ai)

        # Sayfa 3: Efektler & Geçişler Çekmecesi
        page_fx = QWidget()
        pfx_lay = QVBoxLayout(page_fx)
        pfx_lay.setContentsMargins(0, 0, 0, 0)
        pfx_lay.setSpacing(4)
        self.fx_preset_combo = QComboBox()
        self.fx_preset_combo.addItem("Yumuşak Ses/Video Geçişi (0.8s Fade In/Out)", "fade_smooth")
        self.fx_preset_combo.addItem("Sinematik Kontrast (+Kontrast & Dengeli Parlaklık)", "cinema_contrast")
        self.fx_preset_combo.addItem("Canlı / Parlak Görünüm (+Parlaklık)", "bright_vivid")
        self.fx_preset_combo.addItem("Stüdyo Konuşma Netliği (+Presence Ses)", "voice_clarity")
        self.fx_preset_combo.addItem("Tüm Klip Efektlerini Sıfırla", "reset_fx")
        pfx_lay.addWidget(self.fx_preset_combo)
        apply_fx_btn = QPushButton(" Seçili Klibe Efekti / Geçişi Uygula")
        apply_fx_btn.setProperty("compact", "true")
        apply_fx_btn.setIcon(icons.icon("sliders", size=12))
        apply_fx_btn.clicked.connect(self._apply_quick_effect_preset)
        pfx_lay.addWidget(apply_fx_btn)
        self.context_drawer.addWidget(page_fx)

        m_layout.addWidget(self.context_drawer)

        # Medya Listesi
        self.media_list = QListWidget()
        self.media_list.setIconSize(QSize(42, 26))
        self.media_list.setToolTip(
            "Çift tıklayın: Önizlemede aç · Dosyaları buraya sürükleyip bırakabilirsiniz"
        )
        self.media_list.itemClicked.connect(self._on_media_item_clicked)
        self.media_list.itemDoubleClicked.connect(self._on_media_item_double_clicked)
        m_layout.addWidget(self.media_list, 1)

        self.bin_info_label = QLabel("Dosya eklemek için sürükleyin veya 'İçe Aktar'a tıklayın.")
        self.bin_info_label.setObjectName("MutedSmall")
        self.bin_info_label.setWordWrap(True)
        m_layout.addWidget(self.bin_info_label)

        top_splitter.addWidget(media_card)

        # 2. Orta: Video Önizleme
        self.preview_widget = VideoPreviewWidget()
        top_splitter.addWidget(self.preview_widget)

        # 3. Sağ: Klip Inspector & Proje Detayları Paneli
        self.inspector_widget = ClipInspectorWidget()
        self.inspector_widget.setMinimumWidth(255)
        self.inspector_widget.update_project_summary(self.engine.project)
        top_splitter.addWidget(self.inspector_widget)

        top_splitter.setCollapsible(0, False)
        top_splitter.setCollapsible(1, False)
        top_splitter.setCollapsible(2, False)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 3)
        top_splitter.setStretchFactor(2, 1)

        main_splitter.addWidget(top_splitter)

        # ------------------------------------------------------------- ALT BÖLÜM: AKILLI TIMELINE
        self.timeline_widget = MultiTrackTimeline(self.engine)
        self.timeline_widget.setMinimumHeight(220)
        main_splitter.addWidget(self.timeline_widget)

        main_splitter.setCollapsible(0, False)
        main_splitter.setCollapsible(1, False)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)

        root.addWidget(main_splitter, 1)

    def _switch_top_tool_tab(self, tab_key: str) -> None:
        if tab_key == "media":
            self.drawer_title_label.setText("MEDYA HAVUZU")
            self.context_drawer.setCurrentIndex(0)
            self.category_combo.setCurrentIndex(0)  # Tümü
        elif tab_key == "audio":
            self.drawer_title_label.setText("SES && STEM KÜTÜPHANESİ")
            self.context_drawer.setCurrentIndex(0)
            idx = self.category_combo.findData("audio")
            if idx >= 0:
                self.category_combo.setCurrentIndex(idx)
        elif tab_key == "text":
            self.drawer_title_label.setText("METİN && ALTYAZILAR")
            self.context_drawer.setCurrentIndex(1)
        elif tab_key == "ai":
            self.drawer_title_label.setText("AI STÜDYO ARAÇLARI")
            self.context_drawer.setCurrentIndex(2)
        elif tab_key == "effects":
            self.drawer_title_label.setText("EFEKTLER && GEÇİŞLER")
            self.context_drawer.setCurrentIndex(3)

    def _connect_signals(self):
        self.timeline_widget.playheadMoved.connect(self._on_timeline_seek)
        self.preview_widget.positionChanged.connect(self._on_video_pos_changed)
        self.preview_widget.set_playback_controller(self.playback_controller)
        self.playback_controller.timelinePositionChanged.connect(self._on_playback_clock_pos_changed)
        self.timeline_widget.clipSelected.connect(self._on_clip_selected_for_inspector)
        self.timeline_widget.clipModified.connect(self._on_timeline_clip_modified)
        self.timeline_widget.recordVoiceoverRequested.connect(self._toggle_dubbing_record)
        self.timeline_widget.toolModeChanged.connect(self._sync_top_tool_mode_buttons)
        self.inspector_widget.clipChanged.connect(self._on_inspector_clip_modified)
        self.inspector_widget.separateAudioRequested.connect(self._separate_clip_audio)
        self.timeline_widget.filesDropped.connect(self.import_files)
        self.ctx.projectChanged.connect(self._on_active_project_changed)
        self.ctx.stemsToTimelineRequested.connect(self.receive_stems_from_studio)
        self.ctx.transcriptionToTimelineRequested.connect(self.receive_transcription_from_studio)
        self.engine.on_changed(self._on_engine_state_changed)

        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._safe_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._safe_redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._safe_redo)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_project_state)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.timeline_widget._split_selected)
        QShortcut(QKeySequence("Ctrl+D"), self, activated=self.timeline_widget._duplicate_selected)
        QShortcut(QKeySequence("Delete"), self, activated=self._safe_delete)
        QShortcut(QKeySequence("Shift+Delete"), self, activated=self._safe_ripple_delete)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self._safe_copy)
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self._safe_paste)
        QShortcut(QKeySequence("Space"), self, activated=self._safe_toggle_play)
        QShortcut(QKeySequence("Ctrl++"), self, activated=self.timeline_widget.zoom_in)
        QShortcut(QKeySequence("Ctrl+="), self, activated=self.timeline_widget.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, activated=self.timeline_widget.zoom_out)
        QShortcut(QKeySequence("Shift+Z"), self, activated=self.timeline_widget.fit_to_project)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self._open_export_dialog)
        QShortcut(QKeySequence("F1"), self, activated=self._show_shortcuts_dialog)
        self._sync_top_tool_mode_buttons(self.timeline_widget.tool_mode)

    def _on_clip_selected_for_inspector(self, clip: Optional[TimelineClip]) -> None:
        self._inspector_clean_state = self.engine.project.to_dict()
        self.inspector_widget.set_clip(clip)

    def _safe_undo(self) -> None:
        if self._is_text_input_focused():
            return
        if self.engine.undo():
            self._inspector_clean_state = self.engine.project.to_dict()
            if self.inspector_widget.current_clip:
                found = self.engine.project.find_clip(self.inspector_widget.current_clip.id)
                self.inspector_widget.set_clip(found)
            self.timeline_widget.canvas.update()
            self._mark_dirty(True)

    def _safe_redo(self) -> None:
        if self._is_text_input_focused():
            return
        if self.engine.redo():
            self._inspector_clean_state = self.engine.project.to_dict()
            if self.inspector_widget.current_clip:
                found = self.engine.project.find_clip(self.inspector_widget.current_clip.id)
                self.inspector_widget.set_clip(found)
            self.timeline_widget.canvas.update()
            self._mark_dirty(True)

    def _safe_delete(self) -> None:
        if self._is_text_input_focused():
            return
        self.timeline_widget._delete_selected()

    def _safe_ripple_delete(self) -> None:
        if self._is_text_input_focused():
            return
        self.timeline_widget._ripple_delete_selected()
        ToastNotification.show_toast(self, "Klip silindi ve boşluk kapatıldı", "info")

    def _safe_copy(self) -> None:
        if self._is_text_input_focused():
            return
        if self.timeline_widget._copy_selected():
            ToastNotification.show_toast(self, "Klip panoya kopyalandı", "info")

    def _safe_paste(self) -> None:
        if self._is_text_input_focused():
            return
        pasted = self.timeline_widget._paste_at_playhead()
        if pasted:
            ToastNotification.show_toast(self, "Klip zaman çizelgesine yapıştırıldı", "success")

    def _show_shortcuts_dialog(self) -> None:
        dlg = ShortcutGuideDialog(self)
        dlg.exec()

    def _safe_toggle_play(self) -> None:
        if self._is_text_input_focused():
            return
        if hasattr(self, "playback_controller") and self.playback_controller is not None:
            self.playback_controller.toggle_play()
        else:
            self.preview_widget.toggle_play()

    # ------------------------------------------------------------- Araç Modu & B Tuşu Kısayol Koruması
    @staticmethod
    def _is_text_input_focused() -> bool:
        """
        Ek Gereksinim 2 — Metin kutusu, arama alanı, altyazı düzenleyici, proje adı veya
        modal pencere aktifken tek harfli kısayolların (B, V, M) tetiklenmesini engeller.
        """
        from PySide6.QtWidgets import QAbstractSpinBox, QApplication

        if QApplication.activeModalWidget() is not None:
            return True
        fw = QApplication.focusWidget()
        if fw is None:
            return False
        return isinstance(
            fw,
            (QLineEdit, QPlainTextEdit, QComboBox, QAbstractSpinBox),
        ) or fw.inherits("QTextEdit")

    def set_editor_tool_mode(self, mode: str) -> None:
        self.timeline_widget.set_tool_mode(mode)
        self._sync_top_tool_mode_buttons(self.timeline_widget.tool_mode)

    def _sync_top_tool_mode_buttons(self, mode: str) -> None:
        is_blade = mode == "blade"
        self.top_select_tool_btn.blockSignals(True)
        self.top_blade_tool_btn.blockSignals(True)
        self.top_select_tool_btn.setChecked(not is_blade)
        self.top_blade_tool_btn.setChecked(is_blade)
        self.top_select_tool_btn.blockSignals(False)
        self.top_blade_tool_btn.blockSignals(False)
        self.top_select_tool_btn.setStyleSheet(
            "background-color: #1C1D21; color: #F5F5F6; font-weight: 700; border: 1px solid #E53945; border-radius: 4px;"
            if not is_blade
            else ""
        )
        self.top_blade_tool_btn.setStyleSheet(
            "background-color: rgba(229, 57, 69, 0.24); color: #FF4652; font-weight: 700; border: 1px solid #E53945; border-radius: 4px;"
            if is_blade
            else ""
        )

    def _on_top_snap_toggled(self, checked: bool) -> None:
        self.timeline_widget.snap_btn.setChecked(checked)

    def trigger_key_shortcut(self, key_text: str) -> bool:
        """
        B / V / M kısayollarını metin kutusu odağını kontrol ederek güvenli şekilde çalıştırır.
        Eğer odak bir metin giriş alanındaysa False döner ve hiçbir kesme/mod değişimi yapmaz.
        """
        if self._is_text_input_focused():
            return False
        k = (key_text or "").strip().lower()
        if k == "b":
            self.set_editor_tool_mode("blade")
            # Eğer seçili bir klip varsa ve playhead klibin içindeyse anında imleçten de bölebilir
            clip = self.timeline_widget.canvas.get_selected_clip()
            ph = self.engine.project.playhead_pos
            if clip and (clip.timeline_in + 0.04 < ph < clip.timeline_out - 0.04):
                self.timeline_widget._split_selected()
            return True
        if k == "v":
            self.set_editor_tool_mode("select")
            return True
        if k == "m":
            self.timeline_widget.canvas._add_marker(self.engine.project.playhead_pos)
            return True
        return False

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if not self._is_text_input_focused() and not (
            event.modifiers()
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.AltModifier
                | Qt.KeyboardModifier.MetaModifier
            )
        ):
            if event.key() == Qt.Key.Key_B:
                if self.trigger_key_shortcut("b"):
                    event.accept()
                    return
            elif event.key() in (Qt.Key.Key_V, Qt.Key.Key_Escape):
                if self.trigger_key_shortcut("v"):
                    event.accept()
                    return
            elif event.key() == Qt.Key.Key_Space:
                self._safe_toggle_play()
                event.accept()
                return
            elif event.key() == Qt.Key.Key_M:
                if self.trigger_key_shortcut("m"):
                    event.accept()
                    return
            elif event.key() == Qt.Key.Key_N:
                self.timeline_widget.snap_btn.toggle()
                st = "Açık" if self.timeline_widget.snap_btn.isChecked() else "Kapalı"
                ToastNotification.show_toast(self, f"Mıknatıslı Hizalama: {st}", "info")
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Z:
                self.timeline_widget.focus_selected_clip()
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Left:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.timeline_widget.step_seconds(-1.0)
                else:
                    self.timeline_widget.step_frame(-1)
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Right:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.timeline_widget.step_seconds(1.0)
                else:
                    self.timeline_widget.step_frame(1)
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Home:
                self.timeline_widget.jump_home()
                event.accept()
                return
            elif event.key() == Qt.Key.Key_End:
                self.timeline_widget.jump_end()
                event.accept()
                return
            elif event.key() == Qt.Key.Key_F1:
                self._show_shortcuts_dialog()
                event.accept()
                return
        super().keyPressEvent(event)

    def _on_engine_state_changed(self) -> None:
        self.inspector_widget.update_project_summary(self.engine.project)
        self._mark_dirty(True)

    def _mark_dirty(self, dirty: bool) -> None:
        self._is_dirty = bool(dirty)
        if hasattr(self, "save_status_lbl"):
            if self._is_dirty:
                self.save_status_lbl.setText("● Kaydedilmemiş *")
                self.save_status_lbl.setStyleSheet("color: #F59E0B; font-size: 11px; font-weight: 600;")
            else:
                self.save_status_lbl.setText("● Kaydedildi")
                self.save_status_lbl.setStyleSheet("color: #10B981; font-size: 11px; font-weight: 600;")

    def _on_inspector_clip_modified(self, clip: TimelineClip) -> None:
        if getattr(self, "_inspector_clean_state", None) is not None:
            self.engine._undo_stack.append(self._inspector_clean_state)
            if len(self.engine._undo_stack) > self.engine._max_undo_depth:
                self.engine._undo_stack.pop(0)
            self.engine._redo_stack.clear()
            self._inspector_clean_state = None
        else:
            self.engine.snapshot()

        if clip and clip.clip_type == ClipType.VIDEO:
            self.preview_widget.audio_output.setMuted(clip.is_muted)
        self.timeline_widget.canvas.update()
        self._mark_dirty(True)

    def _on_timeline_clip_modified(self, clip: TimelineClip) -> None:
        if clip and self.inspector_widget.current_clip and self.inspector_widget.current_clip.id == clip.id:
            self.inspector_widget.sync_clip_values(clip)
        self._mark_dirty(True)

    def _open_diagnostics_dialog(self) -> None:
        dlg = DiagnosticsAndTasksDialog(self.ctx, self, self)
        dlg.exec()

    # ------------------------------------------------------------- Otomatik Kayıt ve Proje Durumu
    def _auto_save_tick(self) -> None:
        try:
            self.ctx.project_manager.save_autosave_snapshot(
                self.engine.project.to_dict(),
                media_bin=list(self._bin_entries.keys()),
            )
        except Exception:
            pass

    def _on_project_title_edited(self, text: str) -> None:
        self.engine.project.name = text.strip() or "Adsız Proje"
        self.inspector_widget.update_project_summary(self.engine.project)
        self._mark_dirty(True)

    def _on_active_project_changed(self, proj) -> None:
        if proj is None:
            return
        self.load_project_state()

    def save_project_state(self, show_toast: bool = True) -> Optional[Path]:
        pm = self.ctx.project_manager
        if pm.active_project is None:
            pm.create_project(name=self.engine.project.name or "Yeni Dublaj Projesi")
        saved_path = pm.save_timeline_state(
            self.engine.project.to_dict(),
            media_bin=list(self._bin_entries.keys()),
        )
        self._auto_save_tick()
        self._mark_dirty(False)
        if show_toast and saved_path:
            ToastNotification.show_toast(
                self, f"Proje kaydedildi: {Path(saved_path).name}", "success"
            )
        return saved_path

    def load_project_state(self) -> bool:
        pm = self.ctx.project_manager
        tl_dict, bin_paths = pm.load_timeline_state()
        if tl_dict:
            self.engine.project = TimelineProject.from_dict(tl_dict)
            self.proj_title_edit.blockSignals(True)
            self.proj_title_edit.setText(self.engine.project.name)
            self.proj_title_edit.blockSignals(False)
            self.timeline_widget.rebuild_headers()
            self.timeline_widget.canvas.update()
            if hasattr(self, "playback_controller") and self.playback_controller is not None:
                self.playback_controller.set_engine(self.engine)
                self._sync_preview_to_timeline(self.engine.project.playhead_pos)
        elif pm.active_project:
            self.engine.project.name = pm.active_project.name
            self.proj_title_edit.blockSignals(True)
            self.proj_title_edit.setText(pm.active_project.name)
            self.proj_title_edit.blockSignals(False)

        self._bin_entries.clear()
        for bp in bin_paths:
            if Path(bp).exists():
                self._register_media_in_bin(bp)
        self._refresh_media_bin_view()
        self.inspector_widget.update_project_summary(self.engine.project)
        self._mark_dirty(False)
        return tl_dict is not None

    # ------------------------------------------------------------- Senkronizasyon
    def _on_timeline_seek(self, t: float):
        self.engine.project.playhead_pos = t
        if hasattr(self, "playback_controller") and self.playback_controller is not None:
            self.playback_controller.seek(t)
        self._sync_preview_to_timeline(t)

    def _on_video_pos_changed(self, t: float):
        if hasattr(self, "playback_controller") and self.playback_controller is not None and self.playback_controller.is_playing:
            return
        self.timeline_widget.set_playhead(t, auto_scroll=False, emit_seek=False)

    def _on_playback_clock_pos_changed(self, t: float):
        self.timeline_widget.set_playhead(t, auto_scroll=True, emit_seek=False)
        self._sync_preview_to_timeline(t)

    def _sync_preview_to_timeline(self, t: float) -> None:
        """Playhead konumundaki en üst görünür video/görsel/altyazı klibini önizleme oynatıcısına yükler ve kareyi senkronize eder."""
        if hasattr(self.preview_widget, "set_timeline_time"):
            self.preview_widget.set_timeline_time(t, self.engine.project.duration)

        # 1. Video veya görsel klibi bul (en üst görünür kanaldan aşağıya)
        active_video_clip: Optional[TimelineClip] = None
        if hasattr(self, "playback_controller") and self.playback_controller is not None:
            active_video_clip = self.playback_controller.get_active_video_clip(t)
        else:
            for track in reversed(self.engine.project.tracks):
                if track.is_muted or track.track_type not in (TrackType.VIDEO, TrackType.GRAPHICS):
                    continue
                for clip in track.clips:
                    if clip.timeline_in <= t < clip.timeline_out:
                        active_video_clip = clip
                        break
                if active_video_clip:
                    break

        # 2. Altyazı klibi bul
        active_sub_clip: Optional[TimelineClip] = None
        for track in self.engine.project.tracks:
            if track.is_muted or track.track_type != TrackType.SUBTITLE:
                continue
            for clip in track.clips:
                if clip.timeline_in <= t < clip.timeline_out:
                    active_sub_clip = clip
                    break
            if active_sub_clip:
                break

        # Subtitle overlay güncelle
        sub_text = active_sub_clip.text_content if active_sub_clip else ""
        if hasattr(self.preview_widget, "set_subtitle_overlay"):
            self.preview_widget.set_subtitle_overlay(sub_text)

        if active_video_clip and active_video_clip.media_path and Path(active_video_clip.media_path).exists():
            offset = t - active_video_clip.timeline_in
            source_t = active_video_clip.source_in + (offset * max(0.01, active_video_clip.speed))
            if self.preview_widget.current_media_path != active_video_clip.media_path:
                self.preview_widget.load_media(active_video_clip.media_path)
            if hasattr(self.preview_widget, "seek_source_seconds"):
                self.preview_widget.seek_source_seconds(source_t)
            else:
                self.preview_widget.seek_seconds(source_t)
        else:
            # Boşluk (gap) durumunda temiz siyah kare göster
            if hasattr(self.preview_widget, "show_blank_frame"):
                self.preview_widget.show_blank_frame()
            else:
                self.preview_widget.seek_seconds(t)


    # ------------------------------------------------------------- Medya Havuzu ve İçe Aktarma
    def _import_media(self):
        v_exts = " ".join(f"*{e}" for e in C.VIDEO_EXTENSIONS)
        a_exts = " ".join(f"*{e}" for e in C.AUDIO_EXTENSIONS)
        i_exts = " ".join(f"*{e}" for e in C.IMAGE_EXTENSIONS)
        s_exts = " ".join(f"*{e}" for e in C.SUBTITLE_EXTENSIONS)
        all_exts = f"{v_exts} {a_exts} {i_exts} {s_exts}"

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Medya Dosyalarını İçe Aktar",
            str(self.ctx.recordings_dir()),
            f"Tüm Desteklenen Medyalar ({all_exts});;"
            f"Videolar ({v_exts});;"
            f"Ses ve Müzik ({a_exts});;"
            f"Görseller ({i_exts});;"
            f"Altyazılar ({s_exts})",
        )
        if paths:
            self.import_files(paths)

    def import_files(self, file_paths: List[str | Path], show_errors: bool = True) -> int:
        imported_count = 0
        errors: List[str] = []
        for raw_p in file_paths:
            try:
                self._add_media_to_bin_and_timeline(str(raw_p), raise_on_error=True)
                imported_count += 1
            except Exception as exc:
                logger.warning(f"Medya içe aktarılamadı ({raw_p}): {exc}")
                errors.append(f"• {Path(raw_p).name}: {exc}")

        if errors and show_errors:
            QMessageBox.warning(
                self,
                "Bazı Dosyalar İçe Aktarılamadı",
                "Aşağıdaki dosyalar bozuk, boş veya desteklenmeyen biçimde olduğu için "
                "içe aktarılamadı:\n\n" + "\n".join(errors),
            )
        return imported_count

    def _register_media_in_bin(self, media_path: str | Path, info: Optional[MediaInfo] = None) -> dict:
        p = Path(media_path)
        if info is None:
            info = MediaExtractor.probe(p)

        cat = classify_media_category(p)
        if info.has_video:
            cat = "video"
        elif info.media_category in ("image", "subtitle"):
            cat = info.media_category

        thumb_path = ""
        if cat in ("video", "image"):
            tp = MediaExtractor.extract_thumbnail(p, size=(96, 54))
            if tp:
                thumb_path = str(tp)

        res_str = f"{info.video_resolution[0]}x{info.video_resolution[1]}" if info.video_resolution else ""
        stream_summary = ""
        if info.streams:
            s0 = info.streams[0]
            stream_summary = f"{s0.codec_name.upper()} · {s0.sample_rate} Hz · {s0.channels} kanal"
            if len(info.streams) > 1:
                stream_summary += f" (+{len(info.streams) - 1} ses akışı)"
        elif info.media_category == "subtitle":
            stream_summary = f"{info.subtitle_count} altyazı repliği"

        entry = {
            "path": str(p),
            "name": p.name,
            "category": cat,
            "duration": info.duration_sec,
            "format": info.format_name.upper(),
            "resolution": res_str,
            "video_codec": (info.video_codec or "").upper(),
            "stream_summary": stream_summary,
            "size": info.file_size,
            "thumbnail": thumb_path,
            "info": info,
        }
        self._bin_entries[str(p)] = entry
        self.ctx.add_imported_media(p)
        return entry

    def _refresh_media_bin_view(self) -> None:
        selected_cat = self.category_combo.currentData() or "all"
        query = self.bin_search_edit.text().strip().lower()

        self.media_list.clear()
        cat_icon_map = {
            "video": "video",
            "audio": "music",
            "recording": "mic",
            "image": "image",
            "subtitle": "subtitle",
            "stem": "layers",
            "ai": "sparkles",
        }

        for path_str, entry in self._bin_entries.items():
            if selected_cat != "all" and entry["category"] != selected_cat:
                continue
            if query and query not in entry["name"].lower() and query not in entry["format"].lower():
                continue

            dur_str = format_duration(entry["duration"]) if entry["duration"] > 0 else "—"
            meta_parts = [entry["format"], dur_str]
            if entry["resolution"]:
                meta_parts.append(entry["resolution"])
            subtitle_line = " · ".join(meta_parts)

            item = QListWidgetItem(f"{entry['name']}\n{subtitle_line}")
            item.setData(Qt.ItemDataRole.UserRole, path_str)

            thumb = entry.get("thumbnail")
            if thumb and Path(thumb).exists():
                pm = QPixmap(thumb)
                if not pm.isNull():
                    item.setIcon(QIcon(pm))
                else:
                    item.setIcon(icons.icon(cat_icon_map.get(entry["category"], "folder"), size=20))
            else:
                item.setIcon(icons.icon(cat_icon_map.get(entry["category"], "folder"), size=20))

            tooltip_lines = [
                f"Dosya: {entry['name']}",
                f"Yol: {path_str}",
                f"Kategori: {entry['category'].upper()} | Biçim: {entry['format']}",
                f"Süre: {dur_str} | Boyut: {format_file_size(entry['size'])}",
            ]
            if entry["resolution"]:
                tooltip_lines.append(f"Çözünürlük: {entry['resolution']} ({entry['video_codec']})")
            if entry["stream_summary"]:
                tooltip_lines.append(f"Akış: {entry['stream_summary']}")
            item.setToolTip("\n".join(tooltip_lines))

            self.media_list.addItem(item)

        total = len(self._bin_entries)
        shown = self.media_list.count()
        self.bin_info_label.setText(
            f"Havuzda {total} medya ({shown} gösteriliyor)"
            if total
            else "Dosya eklemek için sürükleyin veya 'İçe Aktar'a tıklayın."
        )

    def _add_media_to_bin_and_timeline(self, media_path: str, raise_on_error: bool = False):
        p = Path(media_path)
        if not p.exists():
            if raise_on_error:
                raise FileNotFoundError(f"Dosya bulunamadı: {p}")
            return

        try:
            info = MediaExtractor.probe(p)
            entry = self._register_media_in_bin(p, info=info)
            self._refresh_media_bin_view()
        except Exception as exc:
            if raise_on_error:
                raise
            QMessageBox.warning(self, "Medya Açılamadı", f"{p.name} açılamadı:\n{exc}")
            return

        self._insert_entry_into_timeline(entry)

    def _insert_entry_into_timeline(self, entry: dict) -> None:
        p = Path(entry["path"])
        info: MediaInfo = entry["info"]
        dur = max(0.1, info.duration_sec)
        insert_t = self.engine.project.playhead_pos

        if info.media_category == "subtitle":
            cues = parse_subtitles(p)
            sub_track = next(
                (t for t in self.engine.project.tracks if t.track_type == TrackType.SUBTITLE),
                None,
            )
            if not sub_track:
                sub_track = self.engine.add_track(name="Altyazı", track_type=TrackType.SUBTITLE, color="#F39C12")
            self.engine.snapshot()
            for cue in cues:
                clip = TimelineClip(
                    name=f"{p.stem} #{cue.index}",
                    media_path=str(p),
                    clip_type=ClipType.SUBTITLE,
                    track_id=sub_track.id,
                    timeline_in=cue.start_sec,
                    timeline_out=cue.end_sec,
                    source_in=0.0,
                    source_out=cue.end_sec - cue.start_sec,
                    text_content=cue.text,
                    color="#F39C12",
                )
                sub_track.clips.append(clip)
            sub_track.clips.sort(key=lambda c: c.timeline_in)
            self.engine._notify()
            self.preview_widget.load_media(p)
            return

        if info.media_category == "image":
            v_track = next(
                (t for t in self.engine.project.tracks if t.track_type == TrackType.VIDEO),
                None,
            )
            if not v_track:
                v_track = self.engine.add_track(name="Video 1", track_type=TrackType.VIDEO, color="#2980B9")
            clip = TimelineClip(
                name=p.stem,
                media_path=str(p),
                clip_type=ClipType.IMAGE,
                track_id=v_track.id,
                timeline_in=insert_t,
                timeline_out=insert_t + dur,
                source_in=0.0,
                source_out=dur,
                color="#16A085",
            )
            self.engine.add_clip(v_track.id, clip)
            self.preview_widget.load_media(p)
            return

        if info.has_video:
            v_track = next(
                (t for t in self.engine.project.tracks if t.track_type == TrackType.VIDEO),
                None,
            )
            if not v_track:
                v_track = self.engine.add_track(name="Video 1", track_type=TrackType.VIDEO, color="#2980B9")

            if info.video_fps and info.video_fps > 0:
                self.engine.project.fps = info.video_fps
                self.preview_widget.fps = info.video_fps
            if info.video_resolution:
                self.engine.project.resolution = info.video_resolution

            clip = TimelineClip(
                name=p.stem,
                media_path=str(p),
                clip_type=ClipType.VIDEO,
                track_id=v_track.id,
                timeline_in=insert_t,
                timeline_out=insert_t + dur,
                source_in=0.0,
                source_out=dur,
                color="#2980B9",
            )
            self.engine.add_clip(v_track.id, clip)
            self.preview_widget.load_media(p)
            self.preview_widget.audio_output.setMuted(False)
        else:
            cat = entry.get("category", "audio")
            preferred_type = TrackType.DUBBING if cat == "recording" else TrackType.MUSIC
            a_track = next(
                (t for t in self.engine.project.tracks if t.track_type == preferred_type),
                None,
            )
            if not a_track:
                a_track = next(
                    (
                        t
                        for t in self.engine.project.tracks
                        if t.track_type in (TrackType.MUSIC, TrackType.DIALOGUE, TrackType.AMBIENT, TrackType.DUBBING)
                    ),
                    None,
                )
            if not a_track:
                a_track = self.engine.add_track(name="Ses 1", track_type=TrackType.MUSIC, color="#8E44AD")

            clip = TimelineClip(
                name=p.stem,
                media_path=str(p),
                clip_type=ClipType.AUDIO,
                track_id=a_track.id,
                timeline_in=insert_t,
                timeline_out=insert_t + dur,
                source_in=0.0,
                source_out=dur,
                color="#C0392B" if cat == "recording" else "#8E44AD",
            )
            self.engine.add_clip(a_track.id, clip)
            self.preview_widget.load_media(p)

    def _on_media_item_clicked(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and path in self._bin_entries:
            e = self._bin_entries[path]
            dur_str = format_duration(e["duration"])
            extra = f" · {e['resolution']}" if e["resolution"] else ""
            stream = f" · {e['stream_summary']}" if e["stream_summary"] else ""
            self.bin_info_label.setText(f"{e['name']} ({e['format']} · {dur_str}{extra}{stream})")

    def _on_media_item_double_clicked(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.preview_widget.load_media(path)

    def _add_selected_bin_item_to_timeline(self):
        item = self.media_list.currentItem()
        if not item:
            QMessageBox.information(self, "Bilgi", "Lütfen önce medya havuzundan bir öğe seçin.")
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and path in self._bin_entries:
            self._insert_entry_into_timeline(self._bin_entries[path])

    def _add_subtitle_clip(self):
        sub_track = None
        for t in self.engine.project.tracks:
            if t.track_type == TrackType.SUBTITLE:
                sub_track = t
                break
        if not sub_track:
            sub_track = self.engine.add_track(name="Altyazı", track_type=TrackType.SUBTITLE, color="#F39C12")

        t_in = self.engine.project.playhead_pos
        custom_txt = self.quick_sub_edit.text().strip() if hasattr(self, "quick_sub_edit") else ""
        clip = TimelineClip(
            name="Yeni Altyazı",
            clip_type=ClipType.SUBTITLE,
            track_id=sub_track.id,
            timeline_in=t_in,
            timeline_out=t_in + 3.0,
            text_content=custom_txt or "Replik metnini sağdaki inspector'dan düzenleyin.",
            color="#F39C12",
        )
        self.engine.add_clip(sub_track.id, clip)
        if hasattr(self, "quick_sub_edit"):
            self.quick_sub_edit.clear()

    def _export_timeline_subtitles_srt(self) -> None:
        from ai.transcription import TranscriptSegment

        segs: List[TranscriptSegment] = []
        for t in self.engine.project.tracks:
            if t.track_type == TrackType.SUBTITLE:
                for idx, c in enumerate(t.clips, start=1):
                    if c.text_content:
                        segs.append(
                            TranscriptSegment(
                                id=idx,
                                start=c.timeline_in,
                                end=c.timeline_out,
                                text=c.text_content,
                            )
                        )
        if not segs:
            QMessageBox.information(self, "Altyazı Yok", "Timeline üzerinde dışa aktarılacak altyazı klibi bulunamadı.")
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Timeline Altyazılarını SRT Olarak Kaydet", "", "SRT Altyazı (*.srt)")
        if dest:
            WhisperTranscriber.export_srt(segs, dest)
            QMessageBox.information(self, "Başarılı", f"Altyazılar SRT olarak kaydedildi:\n{dest}")

    def _mark_clip_silences_on_timeline(self) -> None:
        clip = self.timeline_widget.canvas.get_selected_clip()
        if not clip or not clip.media_path or not Path(clip.media_path).exists():
            QMessageBox.information(self, "Klip Seçin", "Sessizlik tespiti için lütfen timeline'dan bir video veya ses klibi seçin.")
            return
        try:
            speech_iv, silences, _ = detect_speech_and_silence(clip.media_path)
            for s in silences:
                if clip.source_in <= s.start <= clip.source_out:
                    tl_t = clip.timeline_in + (s.start - clip.source_in) / max(0.1, clip.speed)
                    self.engine.project.markers.append((tl_t, f"Sessizlik ({s.duration:.1f}s)", "#38BDF8"))
            self.timeline_widget.canvas.update()
            QMessageBox.information(
                self,
                "Sessizlik Tespiti Tamamlandı",
                f"{len(speech_iv)} konuşma bölümü ve {len(silences)} sessizlik aralığı tespit edilerek timeline cetvelinde işaretlendi.",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Sessizlik Tespit Hatası", str(exc))

    def _apply_quick_effect_preset(self) -> None:
        clip = self.timeline_widget.canvas.get_selected_clip()
        if not clip:
            QMessageBox.information(self, "Klip Seçin", "Efekt veya geçiş uygulamak için lütfen timeline'dan bir klip seçin.")
            return
        preset = self.fx_preset_combo.currentData()
        self.engine.snapshot()
        if preset == "fade_smooth":
            dur_half = max(0.1, clip.duration * 0.35)
            clip.fade_in_sec = min(0.8, dur_half)
            clip.fade_out_sec = min(0.8, dur_half)
        elif preset == "cinema_contrast":
            clip.contrast = 1.25
            clip.brightness = -0.04
        elif preset == "bright_vivid":
            clip.brightness = 0.12
            clip.contrast = 1.10
        elif preset == "voice_clarity" and clip.media_path and Path(clip.media_path).exists():
            self.inspector_widget.set_clip(clip)
            idx = self.inspector_widget.enhance_combo.findData("speech_clarity")
            if idx >= 0:
                self.inspector_widget.enhance_combo.setCurrentIndex(idx)
                self.inspector_widget._apply_clip_enhancement()
        elif preset == "reset_fx":
            clip.fade_in_sec = 0.0
            clip.fade_out_sec = 0.0
            clip.brightness = 0.0
            clip.contrast = 1.0
            clip.scale_x = 1.0
            clip.scale_y = 1.0
            clip.rotation = 0
        self.inspector_widget.set_clip(clip)
        self.timeline_widget.canvas.update()

    # ------------------------------------------------------------- Stüdyo Modüllerinden Gelen Aktarımlar (Bölüm 16.5)
    def receive_stems_from_studio(
        self,
        res: StemSeparationResult,
        stem_paths: Dict[str, str],
        source_media_path: str,
    ) -> None:
        """StemStudioView üzerinden ayrıştırılan stem'leri kaynak klibin tam zaman koduna yerleştirir."""
        matched_clip: Optional[TimelineClip] = None
        norm_src = str(Path(source_media_path).resolve()) if os.path.exists(source_media_path) else source_media_path
        for t in self.engine.project.tracks:
            for c in t.clips:
                if c.media_path and os.path.exists(c.media_path):
                    if str(Path(c.media_path).resolve()) == norm_src:
                        matched_clip = c
                        break
            if matched_clip:
                break

        insert_t = matched_clip.timeline_in if matched_clip else self.engine.project.playhead_pos
        self.engine.add_separated_stems_to_timeline(
            res, stem_paths, insert_time=insert_t, source_clip=matched_clip
        )
        if matched_clip and matched_clip.clip_type == ClipType.VIDEO:
            self.preview_widget.audio_output.setMuted(True)

        for sp in stem_paths.values():
            if Path(sp).exists():
                self._register_media_in_bin(sp)
        self._refresh_media_bin_view()

    def receive_transcription_from_studio(self, trans_payload) -> None:
        if isinstance(trans_payload, dict) and "result" in trans_payload:
            res = trans_payload["result"]
            mode = str(trans_payload.get("mode", "subtitle_track"))
            offset = float(trans_payload.get("offset_time", 0.0))
            upd = bool(trans_payload.get("update_existing", True))
            self.engine.transfer_transcription_to_timeline(
                res, mode=mode, offset_time=offset, update_existing=upd
            )
        else:
            self.engine.add_transcription_to_timeline(trans_payload)

    def receive_dubbing_takes(self, records: list, lang: str, offset_ms: float) -> None:
        """Dublaj Yönetmeni modunda onaylanan replik kayıtlarını zaman kodlarıyla timeline'a yerleştirir."""
        track_name = f"Dublaj ({lang})"
        dub_track = next((t for t in self.engine.project.tracks if t.name == track_name), None)
        if not dub_track:
            dub_track = self.engine.add_track(name=track_name, track_type=TrackType.DUBBING, color="#C0392B")

        self.engine.snapshot()
        offset_s = offset_ms / 1000.0
        for rec in records:
            if not rec.takes or rec.approved_take_idx < 0:
                continue
            wav_p = Path(rec.takes[rec.approved_take_idx])
            if not wav_p.exists():
                continue
            info = MediaExtractor.probe(wav_p)
            dur = max(0.1, info.duration_sec)
            t_in = max(0.0, rec.time_in + offset_s)
            clip = TimelineClip(
                name=f"{rec.character}: {rec.text[:18]}",
                media_path=str(wav_p),
                clip_type=ClipType.AUDIO,
                track_id=dub_track.id,
                timeline_in=t_in,
                timeline_out=t_in + dur,
                source_in=0.0,
                source_out=dur,
                color="#C0392B",
            )
            dub_track.clips.append(clip)
            self._register_media_in_bin(wav_p, info=info)
        dub_track.clips.sort(key=lambda c: c.timeline_in)
        self.engine._notify()
        self._refresh_media_bin_view()

    # ------------------------------------------------------------- Sürükle ve Bırak
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            if paths:
                self.import_files(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    # ------------------------------------------------------------- Dublaj Kaydı
    def _toggle_dubbing_record(self):
        if not self._is_dubbing_recording:
            device = self.ctx.input_device()
            if not device:
                QMessageBox.warning(
                    self,
                    "Uyarı",
                    "Ses giriş cihazı bulunamadı. Lütfen Ayarlar'dan mikrofonu seçin.",
                )
                return

            self._recording_start_t = self.engine.project.playhead_pos
            self._current_rec_path = self.ctx.new_recording_path(label="Dublaj_Take")

            cfg = self.ctx.build_recording_config(label="Dublaj_Take")
            cfg.path = self._current_rec_path

            try:
                self.ctx.recorder.start(cfg)
                self.preview_widget.player.play()
                self._is_dubbing_recording = True
                self.dub_btn.setText(" Kaydı Bitir (Stop)")
                self.dub_btn.setStyleSheet("background-color: #E74C3C; color: white;")
                logger.info(f"Dublaj kaydı başlatıldı @ {self._recording_start_t:.2f}s")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Kayıt başlatılamadı:\n{e}")
        else:
            res = self.ctx.recorder.stop()
            self.preview_widget.player.pause()
            self._is_dubbing_recording = False
            self.dub_btn.setText(" Dublaj Kaydı")
            self.dub_btn.setStyleSheet("")

            if res and res.frames > 0 and self._current_rec_path and self._current_rec_path.exists():
                dur = max(0.1, res.duration)
                dub_track = None
                for t in self.engine.project.tracks:
                    if t.track_type == TrackType.DUBBING:
                        dub_track = t
                        break
                if not dub_track:
                    dub_track = self.engine.add_track(
                        name="Dublaj / Seslendirme", track_type=TrackType.DUBBING, color="#C0392B"
                    )

                clip = TimelineClip(
                    name=self._current_rec_path.stem,
                    media_path=str(self._current_rec_path),
                    clip_type=ClipType.AUDIO,
                    track_id=dub_track.id,
                    timeline_in=self._recording_start_t,
                    timeline_out=self._recording_start_t + dur,
                    source_in=0.0,
                    source_out=dur,
                    color="#C0392B",
                )
                self.engine.add_clip(dub_track.id, clip)
                self._register_media_in_bin(self._current_rec_path)
                self._refresh_media_bin_view()
                logger.info(f"Yeni dublaj kaydı timeline'a eklendi: {clip.name} ({dur:.2f}s)")

    # ------------------------------------------------------------- AI Stem Ayrıştırma
    def _on_quick_stem_separation(self):
        clip = self.timeline_widget.canvas.get_selected_clip()
        if not clip:
            for t in self.engine.project.tracks:
                if t.track_type in (TrackType.VIDEO, TrackType.MUSIC, TrackType.DIALOGUE) and t.clips:
                    clip = t.clips[0]
                    break

        if not clip or not clip.media_path:
            QMessageBox.information(
                self, "Bilgi", "Lütfen sesi ayrıştırılacak bir video veya ses klibi ekleyin ya da seçin."
            )
            return

        self._separate_clip_audio(clip)

    def _separate_clip_audio(self, clip: TimelineClip):
        if not clip.media_path or not os.path.exists(clip.media_path):
            QMessageBox.critical(self, "Hata", "Klibin medya dosyası bulunamadı.")
            return

        model_id = (
            self.editor_ai_model_combo.currentData()
            if hasattr(self, "editor_ai_model_combo") and self.editor_ai_model_combo.currentData()
            else "vox-dubbing-dialogue"
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("AI Ses Ayrıştırma")
        dialog.setMinimumSize(440, 160)
        p_layout = QVBoxLayout(dialog)
        lbl = QLabel("Video sesi çıkarılıyor ve faz-uyumlu AI/HPSS-Wiener modeli çalıştırılıyor...")
        lbl.setWordWrap(True)
        p_layout.addWidget(lbl)
        p_bar = QProgressBar()
        p_bar.setRange(0, 100)
        p_layout.addWidget(p_bar)
        cancel_btn = QPushButton("İptal Et")
        p_layout.addWidget(cancel_btn)

        out_dir = Path(self.ctx.config.recordings_dir) / f"{Path(clip.media_path).stem}_stems"
        worker = EditorStemWorker(clip.media_path, out_dir, model_id=model_id)
        self.stem_worker = worker
        task_id = self.ctx.task_manager.start_task(
            title=f"Editör Stem Ayrıştırma: {clip.name}",
            category="stem",
            target_path=clip.media_path,
            cancel_cb=worker.cancel,
        )
        cancel_btn.clicked.connect(lambda: self.ctx.task_manager.cancel_task(task_id))

        def _on_done(res: StemSeparationResult, stem_paths: dict):
            self.ctx.task_manager.complete_task(task_id, f"{len(stem_paths)} stem timeline'a eklendi")
            dialog.accept()
            self.engine.add_separated_stems_to_timeline(
                res, stem_paths, insert_time=clip.timeline_in, source_clip=clip
            )
            if clip.clip_type == ClipType.VIDEO:
                self.preview_widget.audio_output.setMuted(True)

            for sp in stem_paths.values():
                if Path(sp).exists():
                    self._register_media_in_bin(sp)
            self._refresh_media_bin_view()
            QMessageBox.information(
                self,
                "Ayrıştırma Başarılı",
                f"Medya sesi ({res.model_name}) başarıyla ayrıştırıldı!\n\n"
                f"• Rekonstrüksiyon Tutarlılığı (SNR): {res.reconstruction_snr_db:.1f} dB\n"
                f"• Kaynak klip başlangıç konumu ({clip.timeline_in:.2f}s) ve senkronu korundu.\n"
                "• Orijinal konuşma kanalı otomatik MUTE yapıldı; 'Dublaj / Seslendirme' kanalında kayda başlayabilirsiniz.",
            )

        def _on_err(err_msg: str):
            self.ctx.task_manager.fail_task(task_id, err_msg)
            dialog.reject()
            if "iptal" not in err_msg.lower():
                QMessageBox.critical(self, "Hata", f"Stem ayrıştırma hatası:\n{err_msg}")

        worker.progressSignal.connect(
            lambda val, msg: (
                p_bar.setValue(val),
                lbl.setText(msg),
                self.ctx.task_manager.update_task(task_id, val, msg),
            )
        )
        worker.finishedSignal.connect(_on_done)
        worker.errorSignal.connect(_on_err)
        worker.start()
        dialog.exec()

    # ------------------------------------------------------------- Dışa Aktarma (Bölüm 12)
    def _open_export_dialog(self):
        if self.engine.project.duration <= 0.05:
            QMessageBox.information(
                self, "Bilgi", "Dışa aktarmak için timeline'da en az bir klip bulunmalıdır."
            )
            return

        dest_file, _ = QFileDialog.getSaveFileName(
            self,
            "Nihai Videoyu veya Sesi Kaydet",
            str(Path(self.ctx.config.recordings_dir) / f"{self.engine.project.name}.mp4"),
            "MP4 Video (*.mp4);;MOV Video (*.mov);;MKV Video (*.mkv);;WebM Video (*.webm);;WAV Ses Miksajı (*.wav)",
        )
        if not dest_file:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Medya Dışa Aktarılıyor")
        dialog.setMinimumSize(450, 165)
        d_layout = QVBoxLayout(dialog)
        status_lbl = QLabel("Tamamlanmış video ve dublaj miksajı oluşturuluyor...")
        status_lbl.setWordWrap(True)
        d_layout.addWidget(status_lbl)
        prog = QProgressBar()
        prog.setRange(0, 100)
        d_layout.addWidget(prog)
        cancel_btn = QPushButton("İptal Et")
        d_layout.addWidget(cancel_btn)

        settings = RenderSettings(
            output_path=dest_file,
            resolution=self.engine.project.resolution,
            fps=self.engine.project.fps,
            export_video=Path(dest_file).suffix.lower() not in (".wav", ".flac", ".mp3"),
        )
        renderer = VideoRenderer(self.engine.project, settings)
        task_id = self.ctx.task_manager.start_task(
            title=f"Video/Ses Dışa Aktarma: {Path(dest_file).name}",
            category="export",
            target_path=dest_file,
            cancel_cb=renderer.cancel,
        )
        cancel_btn.clicked.connect(lambda: self.ctx.task_manager.cancel_task(task_id))

        self.render_worker = RenderWorker(renderer)
        self.render_worker.progressSignal.connect(
            lambda p, msg: (
                prog.setValue(int(p * 100)),
                status_lbl.setText(msg),
                self.ctx.task_manager.update_task(task_id, int(p * 100), msg),
            )
        )

        def _on_render_done(f: str):
            self.ctx.task_manager.complete_task(task_id, f"Kaydedildi: {Path(f).name}")
            self.ctx.add_imported_media(f)
            dialog.accept()
            QMessageBox.information(self, "Tamamlandı", f"Medya başarıyla dışa aktarıldı ve doğrulandı:\n{f}")

        def _on_render_err(e: str):
            self.ctx.task_manager.fail_task(task_id, e)
            dialog.reject()
            QMessageBox.critical(self, "Hata", f"Render hatası:\n{e}")

        self.render_worker.finishedSignal.connect(_on_render_done)
        self.render_worker.errorSignal.connect(_on_render_err)

        self.render_worker.start()
        dialog.exec()
