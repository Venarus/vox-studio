"""
VOX Studio — AI Stem Separation Workspace
Referans Tasarım (Image 2):
- Üst Çubuk: İçe Aktar, Analiz Et, A/B Karşılaştırma Anahtarı, Model Seçimi (HTDemucs v4 Pro / VOX), Ayrıştırma Modu (4-Stem / 2-Stem), Dışa Aktar
- Sol Panel: Yüklenen Kaynak Medya Kartı (Artwork, Süre, Örnekleme, Format, Mini VU Metre, Play/Pause)
- Orta Alan: Zaman Cetveli, Orijinal Stereo Miks Dalga Formu ve Renkli Stem Kanalları (1. VOCALS, 2. DRUMS, 3. BASS, 4. OTHER)
- Sağ Denetçi: AI Separation Engine (Motor, Mod, Format, Transient Geliştirme, İzolasyon, De-Reverb, GPU Hızlandırma, İlerleme & Yeniden Ayrıştır)
- Alt Master Transport: Zaman Kodu, Oynatma/Durdurma, Döngü, Yakınlaştırma ve Stereo Master VU Metre
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ai.exporter import StemExporter
from ai.media_extractor import MediaExtractor
from ai.mixer import StemMixer, StemTrack
from ai.stem_separator import (
    ModelCatalog,
    ModelInfo,
    StemEnhancer,
    StemQualityMetrics,
    StemSeparationResult,
    StemSeparator,
    detect_compute_backend,
)
from core import constants as C
from core.format import format_duration
from core.logger import get_logger
from ui import icons
from ui.context import AppContext

logger = get_logger("ui.views.stems")

STEM_COLORS = {
    "vocals": "#F97316",  # Sıcak Şeftali / Turuncu
    "vocal": "#F97316",
    "dialogue": "#F97316",
    "drums": "#EF4444",   # Mercan Kırmızısı
    "drum": "#EF4444",
    "bass": "#3B82F6",    # Kobalt Mavisi
    "other": "#10B981",   # Zümrüt Yeşili
    "music": "#10B981",
    "ambient": "#8B5CF6", # Mor / Ortam
    "original": "#94A3B8",# Gümüş Gri
}


class StemWaveformCanvas(QWidget):
    """Her stem için çift kanallı (Stereo L/R) dalga formu ve senkron imleç (Playhead) çizici."""

    seekRequested = Signal(float)

    def __init__(self, audio_data: Optional[np.ndarray] = None, sample_rate: int = 48000, color: str = "#94A3B8", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.color = color
        self.playhead_pos_sec = 0.0
        self.duration_sec = 0.0
        self._peaks: Optional[np.ndarray] = None

        self.setFixedHeight(54)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._compute_peaks()

    def set_audio(self, audio: np.ndarray, sample_rate: int, color: Optional[str] = None) -> None:
        self.audio_data = audio
        self.sample_rate = sample_rate
        if color:
            self.color = color
        self._compute_peaks()
        self.update()

    def set_playhead(self, pos_sec: float) -> None:
        self.playhead_pos_sec = max(0.0, float(pos_sec))
        self.update()

    def _compute_peaks(self) -> None:
        if self.audio_data is None or len(self.audio_data) == 0:
            self._peaks = None
            self.duration_sec = 0.0
            return

        self.duration_sec = len(self.audio_data) / float(self.sample_rate)
        arr = self.audio_data
        if arr.ndim == 2:
            mono = 0.5 * (arr[:, 0] + arr[:, 1])
        else:
            mono = arr

        num_points = 600
        step = max(1, len(mono) // num_points)
        peaks = []
        for i in range(0, len(mono), step):
            chunk = mono[i : i + step]
            if len(chunk):
                peaks.append(float(np.max(np.abs(chunk))))
        self._peaks = np.array(peaks, dtype=np.float32)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Arka plan
        painter.fillRect(0, 0, w, h, QColor("#111317"))
        painter.setPen(QPen(QColor("#20232A"), 1))
        painter.drawLine(0, h - 1, w, h - 1)

        # Orta hat
        mid_y = h / 2.0
        painter.setPen(QPen(QColor("#181B22"), 1))
        painter.drawLine(0, int(mid_y), w, int(mid_y))

        # Dalga formu çubukları
        if self._peaks is not None and len(self._peaks) > 0:
            base_col = QColor(self.color)
            painter.setPen(QPen(base_col, 1.4))
            n = len(self._peaks)
            bar_w = max(1.0, w / float(n))
            max_amp = h * 0.44

            for i, amp in enumerate(self._peaks):
                bx = i * bar_w
                bh = max(1.5, amp * max_amp)
                painter.drawLine(int(bx), int(mid_y - bh), int(bx), int(mid_y + bh))

        # İmleç (Playhead)
        if self.duration_sec > 0:
            px = int((self.playhead_pos_sec / self.duration_sec) * w)
            painter.setPen(QPen(QColor("#E53945"), 1.8))
            painter.drawLine(px, 0, px, h)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.duration_sec > 0:
            frac = max(0.0, min(1.0, event.position().x() / float(self.width())))
            seek_t = frac * self.duration_sec
            self.seekRequested.emit(seek_t)


class StemTrackHeaderWidget(QFrame):
    """Stem sol kontrol bloğu: Başlık, Mute, Solo, Volume dB Slider, Pan, VU Metre ve Dışa Aktar."""

    changed = Signal()
    exportRequested = Signal()

    def __init__(self, track: StemTrack, idx: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.track = track
        self.idx = idx
        self.setFixedWidth(240)
        self.setFixedHeight(54)
        self.setStyleSheet("background: #14161B; border-right: 1px solid #23262D;")
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(2)

        # Üst Satır: İsim + Mute + Solo + Export
        top_row = QHBoxLayout()
        top_row.setSpacing(4)

        t_name = self.track.name.upper()
        self.name_lbl = QLabel(f"{self.idx}. {t_name}")
        col = STEM_COLORS.get(self.track.name.lower(), "#F8FAFC")
        self.name_lbl.setStyleSheet(f"color: {col}; font-weight: 700; font-size: 11px;")
        top_row.addWidget(self.name_lbl)
        top_row.addStretch(1)

        self.mute_btn = QPushButton("M")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setFixedSize(22, 20)
        self.mute_btn.setStyleSheet("font-size: 9px; font-weight: bold; padding: 0;")
        self.mute_btn.toggled.connect(self._on_mute)
        top_row.addWidget(self.mute_btn)

        self.solo_btn = QPushButton("S")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setFixedSize(22, 20)
        self.solo_btn.setStyleSheet("font-size: 9px; font-weight: bold; padding: 0;")
        self.solo_btn.toggled.connect(self._on_solo)
        top_row.addWidget(self.solo_btn)

        self.exp_btn = QPushButton("Export")
        self.exp_btn.setProperty("compact", "true")
        self.exp_btn.setFixedSize(44, 20)
        self.exp_btn.setStyleSheet("font-size: 9px; padding: 0;")
        self.exp_btn.clicked.connect(self.exportRequested.emit)
        top_row.addWidget(self.exp_btn)
        root.addLayout(top_row)

        # Alt Satır: Volume Slider + dB Label + Pan
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(4)

        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 150)
        self.vol_slider.setValue(100)
        self.vol_slider.setFixedWidth(80)
        self.vol_slider.valueChanged.connect(self._on_vol)
        bottom_row.addWidget(self.vol_slider)

        self.vol_lbl = QLabel("0.0 dB")
        self.vol_lbl.setFixedWidth(44)
        self.vol_lbl.setStyleSheet("color: #38BDF8; font-size: 10px; font-weight: 600;")
        bottom_row.addWidget(self.vol_lbl)

        bottom_row.addWidget(QLabel("Pan:"))
        self.pan_slider = QSlider(Qt.Orientation.Horizontal)
        self.pan_slider.setRange(-100, 100)
        self.pan_slider.setValue(0)
        self.pan_slider.setFixedWidth(45)
        self.pan_slider.valueChanged.connect(self._on_pan)
        bottom_row.addWidget(self.pan_slider)
        root.addLayout(bottom_row)

    def _on_mute(self, checked: bool) -> None:
        self.track.is_muted = checked
        self.mute_btn.setStyleSheet(
            "background: #E53945; color: white; font-weight: bold; font-size: 9px;"
            if checked
            else "font-size: 9px; font-weight: bold; padding: 0;"
        )
        self.changed.emit()

    def _on_solo(self, checked: bool) -> None:
        self.track.is_solo = checked
        self.solo_btn.setStyleSheet(
            "background: #F59E0B; color: black; font-weight: bold; font-size: 9px;"
            if checked
            else "font-size: 9px; font-weight: bold; padding: 0;"
        )
        self.changed.emit()

    def _on_vol(self, val: int) -> None:
        self.track.volume = val / 100.0
        db = 20.0 * np.log10(max(1e-4, self.track.volume)) if val > 0 else -60.0
        self.vol_lbl.setText(f"{db:+.1f} dB" if val > 0 else "-inf")
        self.changed.emit()

    def _on_pan(self, val: int) -> None:
        self.track.pan = val / 100.0
        self.changed.emit()


class StemTrackWidget(QWidget):
    """Track Header + Waveform Canvas bileşimi (Image 2 tam uyumlu)."""

    changed = Signal()
    playSingleRequested = Signal(str, object)
    downloadSingleRequested = Signal(str, object)

    def __init__(self, track: StemTrack, sample_rate: int, metrics: Optional[StemQualityMetrics] = None, idx: int = 1, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.track = track
        self.sample_rate = sample_rate
        self.idx = idx
        self.orig_audio = track.audio.copy()
        self.metrics = metrics or StemSeparator.evaluate_stem_quality(track.name, track.audio, sample_rate)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = StemTrackHeaderWidget(track, idx)
        self.header.changed.connect(self.changed.emit)
        self.header.exportRequested.connect(lambda: self.downloadSingleRequested.emit(self.track.name, self.track.audio))
        layout.addWidget(self.header)

        col = STEM_COLORS.get(track.name.lower(), "#94A3B8")
        self.canvas = StemWaveformCanvas(track.audio, sample_rate, color=col)
        layout.addWidget(self.canvas, 1)

    @property
    def is_selected(self) -> bool:
        return True

    def set_playhead(self, pos: float) -> None:
        self.canvas.set_playhead(pos)


class StemWorker(QThread):
    progressSignal = Signal(float, str)
    finishedSignal = Signal(object, object, int)
    errorSignal = Signal(str)

    def __init__(self, media_path: str, model_id: str):
        super().__init__()
        self.media_path = media_path
        self.model_id = model_id
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):
        try:
            self.progressSignal.emit(0.05, "PCM ses akışı çözülüyor...")
            info = MediaExtractor.probe(self.media_path)
            if not info.streams:
                raise RuntimeError(f"Ayrıştırılabilir ses akışı bulunamadı: {Path(self.media_path).name}")

            audio_arr, fs = MediaExtractor.extract_audio(
                self.media_path, target_sample_rate=48000, target_channels=2
            )
            if self._cancelled:
                raise RuntimeError("İptal edildi.")

            self.progressSignal.emit(0.18, "AI modeli ile stem ayrıştırma başlatılıyor...")
            sep = StemSeparator(model_id=self.model_id)

            def _on_prog(p: float, msg: str):
                self.progressSignal.emit(0.18 + (p * 0.78), msg)

            res = sep.separate(audio_arr, fs, progress_cb=_on_prog, cancel_check=lambda: self._cancelled)
            self.progressSignal.emit(1.0, "Tamamlandı.")
            self.finishedSignal.emit(res, audio_arr, fs)
        except Exception as e:
            logger.error(f"StemWorker hatası: {e}")
            self.errorSignal.emit(str(e))


class StemStudioView(QWidget):
    """VOX Studio AI Stem Separation Studio."""

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx
        self.mixer = StemMixer(sample_rate=48000)
        self.current_media_path: Optional[str] = None
        self.current_result: Optional[StemSeparationResult] = None
        self.original_audio: Optional[np.ndarray] = None
        self.original_sr: int = 48000
        self.track_widgets: List[StemTrackWidget] = []
        self.worker: Optional[StemWorker] = None
        self._active_task_id: Optional[str] = None
        self._ab_playing_original: bool = False

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # ------------------------------------------------------------- 1. ÜST ÇUBUK (Top Bar)
        top_bar = QFrame()
        top_bar.setObjectName("Card")
        top_bar.setFixedHeight(48)
        tb_lay = QHBoxLayout(top_bar)
        tb_lay.setContentsMargins(10, 4, 10, 4)
        tb_lay.setSpacing(8)

        self.choose_btn = QPushButton(" Dosya Seç (Import File)")
        self.choose_btn.setProperty("compact", "true")
        self.choose_btn.setIcon(icons.icon("upload", size=14))
        self.choose_btn.clicked.connect(self._choose_media)
        tb_lay.addWidget(self.choose_btn)

        self.analyze_btn = QPushButton(" Analiz Et")
        self.analyze_btn.setProperty("compact", "true")
        self.analyze_btn.setIcon(icons.icon("activity", size=14))
        tb_lay.addWidget(self.analyze_btn)

        # A/B Karşılaştırma Anahtarı
        self.ab_switch = QCheckBox("A/B Karşılaştırma (A: Orijinal / B: Ayrıştırılmış)")
        self.ab_switch.setStyleSheet("color: #F8FAFC; font-weight: 600; font-size: 11px;")
        self.ab_switch.toggled.connect(self._on_ab_switch_toggled)
        tb_lay.addWidget(self.ab_switch)

        tb_lay.addStretch(1)

        tb_lay.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.models_catalog = ModelCatalog.get_available_models()
        for m in self.models_catalog:
            self.model_combo.addItem(m.name, m.id)
        self.model_combo.setMinimumWidth(180)
        tb_lay.addWidget(self.model_combo)

        tb_lay.addWidget(QLabel("İşlem:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("4 Stems (Vokal, Davul, Bas, Diğer)", "4stems")
        self.mode_combo.addItem("2 Stems (Vokal & Enstrümantal)", "2stems")
        tb_lay.addWidget(self.mode_combo)

        self.export_all_btn = QPushButton(" Export All")
        self.export_all_btn.setProperty("variant", "primary")
        self.export_all_btn.setIcon(icons.icon("download", size=13, color="#FFFFFF"))
        self.export_all_btn.setStyleSheet(
            f"background-color: {C.COLOR_ACCENT}; color: #FFFFFF; font-weight: 700; "
            "border: none; border-radius: 5px; padding: 5px 14px;"
        )
        self.export_all_btn.setEnabled(False)
        self.export_all_btn.clicked.connect(self._export_all_stems)
        tb_lay.addWidget(self.export_all_btn)

        root.addWidget(top_bar)

        # ------------------------------------------------------------- 2. ORTA BÖLÜM (3-PANEL SPLITTER)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # SOL PANEL: YÜKLENEN KAYNAK MEDYA KARTI (Loaded Source)
        left_card = QFrame()
        left_card.setObjectName("Card")
        left_card.setMinimumWidth(210)
        left_card.setMaximumWidth(260)
        lc_lay = QVBoxLayout(left_card)
        lc_lay.setContentsMargins(10, 8, 10, 8)
        lc_lay.setSpacing(6)

        lc_title = QLabel("Loaded Source (Kaynak):")
        lc_title.setStyleSheet("color: #9CA6B8; font-size: 11px; font-weight: 600;")
        lc_lay.addWidget(lc_title)

        self.source_name_lbl = QLabel("Henüz dosya seçilmedi")
        self.source_name_lbl.setWordWrap(True)
        self.source_name_lbl.setStyleSheet("color: #F8FAFC; font-weight: 700; font-size: 12px;")
        lc_lay.addWidget(self.source_name_lbl)

        # Artwork / Mini Dalga Formu
        self.source_art_box = QFrame()
        self.source_art_box.setFixedHeight(85)
        self.source_art_box.setStyleSheet("background: #090A0C; border-radius: 6px;")
        sa_lay = QVBoxLayout(self.source_art_box)
        sa_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_art_lbl = QLabel()
        self.source_art_lbl.setPixmap(icons.pixmap("waveform", 38, "#6C7687"))
        sa_lay.addWidget(self.source_art_lbl)
        lc_lay.addWidget(self.source_art_box)

        self.source_meta_lbl = QLabel("Süre: 00:00:00\nFormat: 48 kHz / 24-bit PCM")
        self.source_meta_lbl.setStyleSheet("color: #9CA6B8; font-size: 11px; line-height: 1.4;")
        lc_lay.addWidget(self.source_meta_lbl)

        # Mini Oynatıcı ve Seviye Çubukları
        mini_row = QHBoxLayout()
        self.mini_play_btn = QPushButton()
        self.mini_play_btn.setIcon(icons.icon("play", size=13))
        self.mini_play_btn.setFixedSize(28, 28)
        self.mini_play_btn.clicked.connect(self._toggle_play_mix)
        mini_row.addWidget(self.mini_play_btn)

        vu_col = QVBoxLayout()
        vu_col.setSpacing(2)
        self.mini_vu_l = QProgressBar()
        self.mini_vu_l.setRange(0, 100)
        self.mini_vu_l.setValue(65)
        self.mini_vu_l.setFixedHeight(5)
        self.mini_vu_l.setTextVisible(False)
        self.mini_vu_l.setStyleSheet("QProgressBar::chunk { background: #10B981; }")
        vu_col.addWidget(self.mini_vu_l)

        self.mini_vu_r = QProgressBar()
        self.mini_vu_r.setRange(0, 100)
        self.mini_vu_r.setValue(60)
        self.mini_vu_r.setFixedHeight(5)
        self.mini_vu_r.setTextVisible(False)
        self.mini_vu_r.setStyleSheet("QProgressBar::chunk { background: #10B981; }")
        vu_col.addWidget(self.mini_vu_r)
        mini_row.addLayout(vu_col)
        lc_lay.addLayout(mini_row)

        lc_lay.addStretch(1)
        splitter.addWidget(left_card)

        # ORTA PANEL: DALGA FORMU MİKSERİ (Multitrack Waveform Area)
        center_card = QFrame()
        center_card.setObjectName("Card")
        cc_lay = QVBoxLayout(center_card)
        cc_lay.setContentsMargins(0, 0, 0, 0)
        cc_lay.setSpacing(2)

        # Zaman Cetveli Başlığı
        self.ruler_bar = QFrame()
        self.ruler_bar.setFixedHeight(22)
        self.ruler_bar.setStyleSheet("background: #0E1014; border-bottom: 1px solid #23262D;")
        rb_lay = QHBoxLayout(self.ruler_bar)
        rb_lay.setContentsMargins(245, 0, 10, 0)
        for t_mark in ("00:00:00", "00:00:30", "00:01:00", "00:01:30", "00:02:00", "00:02:30", "00:03:00"):
            lbl = QLabel(t_mark)
            lbl.setStyleSheet("color: #6C7687; font-size: 9px; font-family: monospace;")
            rb_lay.addWidget(lbl)
            rb_lay.addStretch(1)
        cc_lay.addWidget(self.ruler_bar)

        # Üst Orijinal Stereo Miks Kanalı
        self.orig_track_frame = QFrame()
        self.orig_track_frame.setFixedHeight(54)
        self.orig_track_frame.setStyleSheet("background: #14161B; border-bottom: 1px solid #23262D;")
        ot_lay = QHBoxLayout(self.orig_track_frame)
        ot_lay.setContentsMargins(0, 0, 0, 0)
        ot_lay.setSpacing(0)

        orig_hdr = QLabel(" ORIGINAL STEREO MIX ")
        orig_hdr.setFixedWidth(240)
        orig_hdr.setStyleSheet("color: #94A3B8; font-weight: 700; font-size: 11px; padding-left: 8px;")
        ot_lay.addWidget(orig_hdr)

        self.orig_canvas = StemWaveformCanvas(color="#94A3B8")
        self.orig_canvas.seekRequested.connect(self._on_seek_requested)
        ot_lay.addWidget(self.orig_canvas, 1)
        cc_lay.addWidget(self.orig_track_frame)

        # Stem Kanalları Listesi (Scroll Area)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        self.tracks_container = QWidget()
        self.tracks_layout = QVBoxLayout(self.tracks_container)
        self.tracks_layout.setContentsMargins(0, 0, 0, 0)
        self.tracks_layout.setSpacing(2)

        self.empty_label = QLabel("Ayrıştırılmış stem katmanları (Vokal, Davul, Bas, Diğer) burada görüntülenecektir.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #6C7687; padding: 48px; font-size: 12px;")
        self.tracks_layout.addWidget(self.empty_label)

        scroll.setWidget(self.tracks_container)
        cc_lay.addWidget(scroll, 1)
        splitter.addWidget(center_card)

        # SAĞ PANEL: AI SEPARATION ENGINE INSPECTOR
        right_card = QFrame()
        right_card.setObjectName("Card")
        right_card.setMinimumWidth(240)
        right_card.setMaximumWidth(280)
        rc_lay = QVBoxLayout(right_card)
        rc_lay.setContentsMargins(10, 8, 10, 8)
        rc_lay.setSpacing(6)

        rc_title = QLabel("AI Separation Engine")
        rc_title.setStyleSheet("color: #F8FAFC; font-weight: 700; font-size: 12px;")
        rc_lay.addWidget(rc_title)

        rc_lay.addWidget(QLabel("AI Engine:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("HTDemucs v4 (HQ)", "htdemucs")
        self.engine_combo.addItem("VOX Dubbing Pro", "vox-dubbing-dialogue")
        self.engine_combo.addItem("VOX HPSS Wiener", "vox-hpss-wiener")
        rc_lay.addWidget(self.engine_combo)

        rc_lay.addWidget(QLabel("Split Mode:"))
        self.split_mode_combo = QComboBox()
        self.split_mode_combo.addItem("4 Stems (Tam Ayrıştırma)", "4stems")
        self.split_mode_combo.addItem("2 Stems (Vokal & Altyapı)", "2stems")
        rc_lay.addWidget(self.split_mode_combo)

        rc_lay.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        for fid, flbl, _ in StemExporter.SUPPORTED_FORMATS:
            self.format_combo.addItem(flbl, fid)
        rc_lay.addWidget(self.format_combo)

        # Anahtarlar / Toggles
        self.transients_check = QCheckBox("Enhance Transients")
        self.transients_check.setChecked(True)
        rc_lay.addWidget(self.transients_check)

        self.isolation_check = QCheckBox("Aggressive Isolation")
        rc_lay.addWidget(self.isolation_check)

        self.dereverb_check = QCheckBox("De-Reverb Bleed")
        rc_lay.addWidget(self.dereverb_check)

        # GPU Hızlandırma Durumu (GTX 1660 Super)
        hw = detect_compute_backend()
        self.gpu_check = QCheckBox(f"GPU Acceleration ({hw.get('active_mode', 'Active')})")
        self.gpu_check.setChecked(hw.get("has_cuda", False))
        rc_lay.addWidget(self.gpu_check)

        rc_lay.addStretch(1)

        # Durum ve İlerleme Kutusu
        self.status_label = QLabel("Hazır")
        self.status_label.setStyleSheet("color: #9CA6B8; font-size: 11px;")
        rc_lay.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setVisible(False)
        rc_lay.addWidget(self.progress_bar)

        self.start_btn = QPushButton(" Ayrıştır (Separate)")
        self.start_btn.setProperty("variant", "primary")
        self.start_btn.setIcon(icons.icon("sparkles", size=14, color="#FFFFFF"))
        self.start_btn.setStyleSheet(
            f"background-color: {C.COLOR_ACCENT}; color: #FFFFFF; font-weight: 700; "
            "border: none; border-radius: 5px; padding: 7px 12px;"
        )
        self.start_btn.clicked.connect(self._start_separation)
        rc_lay.addWidget(self.start_btn)

        self.cancel_btn = QPushButton(" İptal Et")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_separation)
        rc_lay.addWidget(self.cancel_btn)

        self.send_all_tl_btn = QPushButton(" Timeline'a Aktar")
        self.send_all_tl_btn.setProperty("compact", "true")
        self.send_all_tl_btn.setIcon(icons.icon("layers", size=13))
        self.send_all_tl_btn.setEnabled(False)
        self.send_all_tl_btn.clicked.connect(lambda: self._send_to_timeline(only_selected=False))
        rc_lay.addWidget(self.send_all_tl_btn)

        splitter.addWidget(right_card)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 1)
        root.addWidget(splitter, 1)

        # ------------------------------------------------------------- 3. ALT MASTER TRANSPORT ÇUBUĞU
        transport_bar = QFrame()
        transport_bar.setObjectName("Card")
        transport_bar.setFixedHeight(50)
        tr_lay = QHBoxLayout(transport_bar)
        tr_lay.setContentsMargins(14, 4, 14, 4)
        tr_lay.setSpacing(10)

        self.timecode_lbl = QLabel("00:00:00:00 / 00:00:00:00")
        self.timecode_lbl.setStyleSheet("color: #F8FAFC; font-family: monospace; font-size: 13px; font-weight: 700;")
        tr_lay.addWidget(self.timecode_lbl)

        tr_lay.addStretch(1)

        # Oynatma Kontrolleri
        self.prev_btn = QPushButton("⏮")
        self.prev_btn.setFixedSize(30, 30)
        self.prev_btn.clicked.connect(lambda: self._on_seek_requested(0.0))
        tr_lay.addWidget(self.prev_btn)

        self.play_mix_btn = QPushButton("▶")
        self.play_mix_btn.setFixedSize(38, 38)
        self.play_mix_btn.setStyleSheet(
            f"background-color: {C.COLOR_ACCENT}; color: #FFFFFF; font-size: 15px; font-weight: bold; border-radius: 19px; border: none;"
        )
        self.play_mix_btn.clicked.connect(self._toggle_play_mix)
        tr_lay.addWidget(self.play_mix_btn)

        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setFixedSize(30, 30)
        self.stop_btn.clicked.connect(self._stop_playback)
        tr_lay.addWidget(self.stop_btn)

        self.loop_btn = QPushButton("🔁")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setFixedSize(30, 30)
        tr_lay.addWidget(self.loop_btn)

        tr_lay.addStretch(1)

        tr_lay.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(50, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(80)
        tr_lay.addWidget(self.zoom_slider)

        # Master Stereo VU Metre
        master_vu_col = QVBoxLayout()
        master_vu_col.setSpacing(2)
        self.master_vu_l = QProgressBar()
        self.master_vu_l.setRange(0, 100)
        self.master_vu_l.setValue(72)
        self.master_vu_l.setFixedSize(110, 6)
        self.master_vu_l.setTextVisible(False)
        self.master_vu_l.setStyleSheet("QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:0.8 #F59E0B, stop:1.0 #EF4444); }")
        master_vu_col.addWidget(self.master_vu_l)

        self.master_vu_r = QProgressBar()
        self.master_vu_r.setRange(0, 100)
        self.master_vu_r.setValue(68)
        self.master_vu_r.setFixedSize(110, 6)
        self.master_vu_r.setTextVisible(False)
        self.master_vu_r.setStyleSheet("QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:0.8 #F59E0B, stop:1.0 #EF4444); }")
        master_vu_col.addWidget(self.master_vu_r)
        tr_lay.addLayout(master_vu_col)

        root.addWidget(transport_bar)

        # Oynatma Zamanlayıcısı (Playhead Takibi)
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(40)
        self._playback_timer.timeout.connect(self._on_playback_tick)

    # ================================================================ Ayrıştırma ve Medya
    def load_media_file(self, path: str | Path) -> bool:
        p = Path(path)
        try:
            info = MediaExtractor.probe(p)
            self.current_media_path = str(p)
            self.source_name_lbl.setText(p.name)
            self.source_meta_lbl.setText(
                f"Süre: {format_duration(info.duration_sec)}\nFormat: 48 kHz / 24-bit PCM Stereo"
            )
            self.start_btn.setEnabled(True)

            audio_arr, fs = MediaExtractor.extract_audio(str(p), target_sample_rate=48000, target_channels=2)
            self.original_audio = audio_arr
            self.original_sr = fs
            self.orig_canvas.set_audio(audio_arr, fs, color="#94A3B8")
            self.timecode_lbl.setText(f"00:00:00:00 / {format_duration(info.duration_sec)}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Medya yüklenemedi: {e}")
            return False

    def _choose_media(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Ayrıştırılacak Medyayı Seç",
            str(self.ctx.recordings_dir()),
            "Tüm Medyalar (*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.flac *.m4a *.aac);;Video (*.mp4 *.mov);;Ses (*.wav *.mp3 *.flac)",
        )
        if path:
            self.load_media_file(path)

    def _start_separation(self) -> None:
        if not self.current_media_path:
            QMessageBox.information(self, "Bilgi", "Lütfen önce bir dosya seçin.")
            return

        model_id = self.model_combo.currentData()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)

        self.worker = StemWorker(self.current_media_path, model_id)
        self._active_task_id = self.ctx.task_manager.start_task(
            title=f"AI Stem Ayrıştırma: {Path(self.current_media_path).name}",
            category="stem",
            target_path=self.current_media_path,
            cancel_cb=self.worker.cancel,
        )
        self.worker.progressSignal.connect(self._on_progress)
        self.worker.finishedSignal.connect(self._on_finished)
        self.worker.errorSignal.connect(self._on_error)
        self.worker.start()

    def _cancel_separation(self) -> None:
        if self.worker:
            self.worker.cancel()
        if self._active_task_id:
            self.ctx.task_manager.cancel_task(self._active_task_id)

    def _on_progress(self, p: float, msg: str) -> None:
        pct = int(p * 100)
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"{pct}% - {msg}")

    def _on_finished(self, res: StemSeparationResult, orig_audio: np.ndarray, sr: int) -> None:
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.start_btn.setEnabled(True)
        self.current_result = res
        self.original_audio = orig_audio
        self.original_sr = sr
        self.status_label.setText(f"Ayrıştırma Başarılı (%100) · SNR: {res.reconstruction_snr_db:.1f} dB")

        # Kanalları oluştur
        self.mixer = StemMixer(sample_rate=res.sample_rate)
        while self.tracks_layout.count():
            item = self.tracks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.track_widgets.clear()

        idx = 1
        for stem_name, audio_data in res.stems.items():
            track = self.mixer.add_stem(stem_name, audio_data)
            metrics = res.quality_metrics.get(stem_name)
            tw = StemTrackWidget(track, sample_rate=res.sample_rate, metrics=metrics, idx=idx)
            tw.canvas.seekRequested.connect(self._on_seek_requested)
            tw.downloadSingleRequested.connect(self._download_single_stem)
            self.tracks_layout.addWidget(tw)
            self.track_widgets.append(tw)
            idx += 1

        self.tracks_layout.addStretch(1)
        self.export_all_btn.setEnabled(True)
        self.send_all_tl_btn.setEnabled(True)

    def _on_error(self, err: str) -> None:
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.start_btn.setEnabled(True)
        self.status_label.setText(f"Hata: {err}")

    # ================================================================ Oynatma ve Senkronizasyon
    def _on_seek_requested(self, seek_t: float) -> None:
        self.orig_canvas.set_playhead(seek_t)
        for tw in self.track_widgets:
            tw.set_playhead(seek_t)
        total_d = self.orig_canvas.duration_sec
        self.timecode_lbl.setText(f"{format_duration(seek_t)} / {format_duration(total_d)}")

    def _toggle_play_mix(self) -> None:
        if self._playback_timer.isActive():
            self._stop_playback()
        else:
            self._playback_timer.start()
            self.play_mix_btn.setText("⏸")

    def _stop_playback(self) -> None:
        self._playback_timer.stop()
        self.play_mix_btn.setText("▶")
        self._on_seek_requested(0.0)

    def _on_playback_tick(self) -> None:
        pos = self.orig_canvas.playhead_pos_sec + 0.04
        dur = self.orig_canvas.duration_sec
        if dur > 0 and pos >= dur:
            if self.loop_btn.isChecked():
                pos = 0.0
            else:
                self._stop_playback()
                return
        self._on_seek_requested(pos)

    def _on_ab_switch_toggled(self, checked: bool) -> None:
        self._ab_playing_original = checked
        self.status_label.setText(
            "A/B: Orijinal Stereo Kaynak Dinleniyor" if checked else "A/B: Ayrıştırılmış Stem Miksi Dinleniyor"
        )

    # ================================================================ Dışa Aktarma
    def _export_all_stems(self) -> None:
        if not self.current_result:
            return
        dest_dir = QFileDialog.getExistingDirectory(self, "Stem Klasörünü Seç", str(self.ctx.recordings_dir()))
        if dest_dir:
            proj_name = Path(self.current_media_path).stem if self.current_media_path else "Stems"
            report = StemExporter.export_all_stems_detailed(
                self.current_result.stems,
                self.current_result.sample_rate,
                Path(dest_dir),
                project_name=proj_name,
                format_type=self.format_combo.currentData() or "wav",
            )
            QMessageBox.information(
                self, "Tamamlandı", f"{len(report.succeeded)} stem başarıyla kaydedildi:\n{dest_dir}"
            )

    def _download_single_stem(self, stem_name: str, audio_arr: np.ndarray) -> None:
        if not self.current_result:
            return
        fmt = self.format_combo.currentData() or "wav"
        dest, _ = QFileDialog.getSaveFileName(
            self, f"'{stem_name}' Stemini Kaydet", f"{stem_name}.{fmt}", f"{fmt.upper()} (*.{fmt})"
        )
        if dest:
            sf.write(dest, audio_arr, self.current_result.sample_rate, subtype="PCM_24")
            QMessageBox.information(self, "Kaydedildi", f"'{stem_name}' kaydedildi:\n{dest}")

    def _send_to_timeline(self, only_selected: bool = False) -> None:
        if not self.current_result or not self.current_media_path:
            return
        out_dir = Path(self.ctx.config.recordings_dir) / f"{Path(self.current_media_path).stem}_stems"
        out_dir.mkdir(parents=True, exist_ok=True)
        report = StemExporter.export_all_stems_detailed(
            self.current_result.stems,
            self.current_result.sample_rate,
            out_dir,
            project_name=Path(self.current_media_path).stem,
            format_type="wav",
        )
        stem_paths = {k: str(v) for k, v in report.succeeded.items()}
        self.ctx.stemsToTimelineRequested.emit(self.current_result, stem_paths, self.current_media_path)
        QMessageBox.information(self, "Timeline'a Aktarıldı", "Stem katmanları video editörüne aktarıldı.")
