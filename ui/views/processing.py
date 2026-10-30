"""
VOX Studio — Ultimate AI Speech Restoration & Audio Cleaning Laboratory
(Profesyonel Yapay Zekâ Destekli Ses Temizleme ve Konuşma Restorasyon Motoru)

Özellikler:
* Basit Mod: Tek tıkla "Maksimum Kaliteyle Temizle" (Adaptive Restoration) ve 3 Hazır Profil
  (Natural, Balanced, Maximum Restoration).
* Profesyonel Mod: GTCRN ONNX Neural Denoise, De-Hum, De-Clip, De-Click, De-Reverb,
  Speech Clarity Engine, De-Esser, Dinamik Kompresör, LUFS Normalizasyonu.
* Gerçek Zamanlı A/B Karşılaştırma: Orijinal (A) vs Temizlenmiş (B) ses arasında
  sıfır gecikmeli, anlık ve isteğe bağlı Loudness-Matched (LUFS eşitlemeli) geçiş.
* Seçili Kesit Önizleme: 2-5 saniyelik bölgeyi saliseler içinde test edip dinleme.
* Öncesi / Sonrası Kalite Kontrol Tablosu: Peak, RMS, LUFS, Gürültü Tabanı, Kırpılma, SNR.
* Dışa Aktarma: 24-bit PCM WAV, 128-320 kbps MP3 (PyAV libmp3lame), FLAC.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from audio.ab_player import ABChannel, ABPlayer
from audio.exporter import export_flac, export_mp3_pyav, export_wav, validate_exported_file
from core import constants as C
from core.format import format_duration, format_file_size
from core.logger import get_logger
from dsp.analysis import AudioAnalysis, analyze_array, analyze_file
from dsp.chain import (
    DEFAULT_STAGE_ORDER,
    STAGE_DISPLAY_NAMES,
    AIDenoiseSettings,
    DSPProcessor,
    ProcessingChainConfig,
    compute_detailed_metrics,
    create_adaptive_profile,
    create_balanced_profile,
    create_maximum_restoration_profile,
    create_natural_profile,
)
from ui import icons
from ui.context import AppContext
from ui.widgets.level_meter import LevelMeter
from ui.widgets.waveform import WaveformView, peaks_from_samples

logger = get_logger("ui.views.processing")


class DSPProcessWorker(QThread):
    """Arka planda tam dosya veya seçili kesit (snippet) DSP/AI zincirini çalıştırır."""

    finishedSignal = Signal(object, float, dict, bool)  # audio_arr, fs, stats, is_snippet
    progressSignal = Signal(float)
    errorSignal = Signal(str)

    def __init__(
        self,
        audio: np.ndarray,
        fs: float,
        config: ProcessingChainConfig,
        is_snippet: bool = False,
    ):
        super().__init__()
        self.audio = audio
        self.fs = fs
        self.config = config.clone()
        self.is_snippet = is_snippet

    def run(self) -> None:
        try:
            processed, stats = DSPProcessor.process(
                self.audio,
                self.fs,
                self.config,
                progress_cb=self.progressSignal.emit,
            )
            self.finishedSignal.emit(processed, self.fs, stats, self.is_snippet)
        except Exception as exc:
            logger.error(f"DSPProcessWorker hatası: {exc}")
            self.errorSignal.emit(str(exc))


class ProcessingView(QWidget):
    """VOX Studio AI Konuşma Restorasyonu ve Ses Temizleme Laboratuvarı."""

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx

        # Ses Durumu
        self._raw_audio: Optional[np.ndarray] = None
        self._processed_audio: Optional[np.ndarray] = None
        self._fs: float = 48000.0
        self._current_path: Optional[Path] = None
        self._analysis: Optional[AudioAnalysis] = None

        # Aktif Yapılandırma ve Oynatıcı
        self.config = create_balanced_profile()
        self.ab_player = ABPlayer(int(self._fs))
        self.ab_player.set_position_callback(self._on_ab_position_update)
        self.ab_player.set_finished_callback(self._on_ab_playback_finished)

        # UI Durumu
        self._is_simple_mode = True
        self._worker: Optional[DSPProcessWorker] = None
        self._undo_stack: List[np.ndarray] = []

        self._build_ui()
        self._wire_signals()
        self._sync_ui_to_config()

    # ==========================================================================
    # UI YERLEŞİMİ (AUDACITY MİMARİSİNDEN İLHAM ALAN MODERN STÜDYO)
    # ==========================================================================
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        main_layout.addWidget(self._build_header_card())

        # Ana Bölücü: Sol/Orta (Waveform + Teşhis + A/B), Sağ (Basit / Pro Ayarlar)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        left_layout.addWidget(self._build_waveform_section(), 3)
        left_layout.addWidget(self._build_transport_section(), 0)
        left_layout.addWidget(self._build_diagnosis_section(), 2)

        self.main_splitter.addWidget(left_container)
        self.main_splitter.addWidget(self._build_settings_panel())
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)

        main_layout.addWidget(self.main_splitter, 1)

    # --------------------------------------------------------------------------
    # 1. ÜST BAŞLIK VE DOSYA BİLGİSİ
    # --------------------------------------------------------------------------
    def _build_header_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("TopHeaderCard")
        card.setStyleSheet(
            f"#TopHeaderCard {{ background: {C.COLOR_BG_CARD}; border: 1px solid {C.COLOR_BORDER}; "
            "border-radius: 8px; padding: 6px; }"
        )
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(12)

        # Sol: Başlık ve Rozet
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        lbl_title = QLabel("AI SPEECH RESTORATION ENGINE")
        lbl_title.setStyleSheet(f"color: {C.COLOR_ACCENT_HOVER}; font-size: 11px; font-weight: 800; letter-spacing: 1px;")
        self.lbl_file_name = QLabel("Ses dosyası seçilmedi (Sürükleyip bırakabilirsiniz)")
        self.lbl_file_name.setStyleSheet(f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(self.lbl_file_name)
        lay.addLayout(title_box)

        self.lbl_file_meta = QLabel("—")
        self.lbl_file_meta.setStyleSheet(f"color: {C.COLOR_TEXT_MUTED}; font-size: 11px;")
        lay.addWidget(self.lbl_file_meta)

        lay.addStretch(1)

        # Mod Değiştirici: Basit Mod / Profesyonel Mod
        mode_box = QHBoxLayout()
        mode_box.setSpacing(4)
        lbl_mode = QLabel("Mod:")
        lbl_mode.setStyleSheet(f"color: {C.COLOR_TEXT_SECONDARY}; font-size: 11px; font-weight: 600;")
        mode_box.addWidget(lbl_mode)

        self.btn_mode_simple = QPushButton("Basit Mod")
        self.btn_mode_simple.setCheckable(True)
        self.btn_mode_simple.setChecked(True)
        self.btn_mode_simple.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_simple.setStyleSheet(
            f"QPushButton {{ background: {C.COLOR_BG_SURFACE}; color: {C.COLOR_TEXT_PRIMARY}; border: 1px solid {C.COLOR_BORDER}; border-radius: 4px; padding: 4px 10px; font-size: 11px; }}"
            f"QPushButton:checked {{ background: {C.COLOR_ACCENT}; color: #FFFFFF; font-weight: 700; border-color: {C.COLOR_ACCENT}; }}"
        )
        self.btn_mode_simple.clicked.connect(lambda: self._set_mode(simple=True))
        mode_box.addWidget(self.btn_mode_simple)

        self.btn_mode_pro = QPushButton("Profesyonel Mod")
        self.btn_mode_pro.setCheckable(True)
        self.btn_mode_pro.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_pro.setStyleSheet(
            f"QPushButton {{ background: {C.COLOR_BG_SURFACE}; color: {C.COLOR_TEXT_PRIMARY}; border: 1px solid {C.COLOR_BORDER}; border-radius: 4px; padding: 4px 10px; font-size: 11px; }}"
            f"QPushButton:checked {{ background: {C.COLOR_ACCENT}; color: #FFFFFF; font-weight: 700; border-color: {C.COLOR_ACCENT}; }}"
        )
        self.btn_mode_pro.clicked.connect(lambda: self._set_mode(simple=False))
        mode_box.addWidget(self.btn_mode_pro)

        lay.addLayout(mode_box)

        # Dosya Aç ve Son Kaydı Yükle Butonları
        self.btn_load_last = QPushButton("Son Kaydı Yükle")
        self.btn_load_last.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_last.setIcon(icons.icon("mic", C.COLOR_TEXT_PRIMARY, 14))
        self.btn_load_last.clicked.connect(self._load_last_recording)
        lay.addWidget(self.btn_load_last)

        self.btn_open_file = QPushButton("Dosya Seç")
        self.btn_open_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_file.setIcon(icons.icon("folder", C.COLOR_TEXT_PRIMARY, 14))
        self.btn_open_file.clicked.connect(self._choose_file)
        lay.addWidget(self.btn_open_file)

        return card

    # --------------------------------------------------------------------------
    # 2. WAVEFORM ALANI (GERÇEK VERİ, SEÇİM VE A/B OVERLAY)
    # --------------------------------------------------------------------------
    def _build_waveform_section(self) -> QWidget:
        card = QFrame()
        card.setObjectName("WaveformCard")
        card.setStyleSheet(
            f"#WaveformCard {{ background: {C.COLOR_BG_SURFACE}; border: 1px solid {C.COLOR_BORDER}; border-radius: 8px; }}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        top_info = QHBoxLayout()
        self.lbl_wf_title = QLabel("Dalga Formu (Orijinal vs Temizlenmiş Ses)")
        self.lbl_wf_title.setStyleSheet(f"color: {C.COLOR_TEXT_SECONDARY}; font-size: 12px; font-weight: 600;")
        top_info.addWidget(self.lbl_wf_title)
        top_info.addStretch(1)

        self.lbl_sel_info = QLabel("Seçim: Yok (Tıklayıp sürükleyerek kesit seçebilirsiniz)")
        self.lbl_sel_info.setStyleSheet(f"color: {C.COLOR_TEXT_MUTED}; font-size: 11px;")
        top_info.addWidget(self.lbl_sel_info)
        lay.addLayout(top_info)

        self.waveform_view = WaveformView()
        self.waveform_view.setMinimumHeight(150)
        self.waveform_view.seekRequested.connect(self._on_waveform_seek)
        self.waveform_view.selectionChanged.connect(self._on_waveform_selection_changed)
        lay.addWidget(self.waveform_view, 1)

        return card

    # --------------------------------------------------------------------------
    # 3. TAŞIMA VE A/B KARŞILAŞTIRMA KONTROLLERİ
    # --------------------------------------------------------------------------
    def _build_transport_section(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TransportCard")
        bar.setStyleSheet(
            f"#TransportCard {{ background: {C.COLOR_BG_CARD}; border: 1px solid {C.COLOR_BORDER}; border-radius: 8px; }}"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(10)

        # Oynat / Durdur
        self.btn_play = QPushButton()
        self.btn_play.setIcon(icons.icon("play", C.COLOR_TEXT_PRIMARY, 16))
        self.btn_play.setFixedSize(36, 36)
        self.btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_play.setToolTip("Oynat / Duraklat (Boşluk Tuşu)")
        self.btn_play.clicked.connect(self._toggle_playback)
        lay.addWidget(self.btn_play)

        self.btn_stop = QPushButton()
        self.btn_stop.setIcon(icons.icon("square", C.COLOR_TEXT_PRIMARY, 14))
        self.btn_stop.setFixedSize(36, 36)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.setToolTip("Durdur ve başa sar")
        self.btn_stop.clicked.connect(self._stop_playback)
        lay.addWidget(self.btn_stop)

        # Süre Göstergesi
        self.lbl_time = QLabel("00:00.00 / 00:00.00")
        self.lbl_time.setStyleSheet(f"color: {C.COLOR_TEXT_PRIMARY}; font-family: monospace; font-size: 12px; font-weight: 700;")
        lay.addWidget(self.lbl_time)

        lay.addSpacing(14)

        # A/B Karşılaştırma Butonu (Büyük ve Vurgulu)
        self.btn_ab = QPushButton("A: Orijinal Sesi Dinle")
        self.btn_ab.setCheckable(True)
        self.btn_ab.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_ab.setMinimumSize(190, 36)
        self.btn_ab.setStyleSheet(
            f"QPushButton {{ background: {C.COLOR_BG_SURFACE}; color: {C.COLOR_TEXT_PRIMARY}; "
            f"border: 2px solid {C.COLOR_BORDER}; border-radius: 6px; font-weight: 700; font-size: 12px; }}"
            f"QPushButton:checked {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:1 #059669); "
            "color: #FFFFFF; border-color: #10B981; }"
        )
        self.btn_ab.setToolTip("A (Orijinal) ve B (Restore Edilmiş) ses arasında anında geçiş yap (Tab / Ctrl+B)")
        self.btn_ab.toggled.connect(self._on_ab_toggled)
        lay.addWidget(self.btn_ab)

        # Loudness Eşitleme Kutucuğu
        self.chk_match_loudness = QCheckBox("Loudness Eşitle (Match LUFS)")
        self.chk_match_loudness.setChecked(True)
        self.chk_match_loudness.setToolTip("İki kaydın ses yüksekliğini eşitleyerek saf temizleme kalitesini tarafsız dinletir")
        self.chk_match_loudness.toggled.connect(lambda c: self.ab_player.set_loudness_matching(c))
        lay.addWidget(self.chk_match_loudness)

        lay.addStretch(1)

        # Kesit Önizle Butonu
        self.btn_preview_snippet = QPushButton("Kesit Önizle (Seçim)")
        self.btn_preview_snippet.setIcon(icons.icon("sliders", C.COLOR_TEXT_PRIMARY, 14))
        self.btn_preview_snippet.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_preview_snippet.setToolTip("Tüm dosyayı beklemeden yalnızca seçilen bölgeye ayarları uygula ve dinle")
        self.btn_preview_snippet.clicked.connect(self._run_snippet_preview)
        lay.addWidget(self.btn_preview_snippet)

        return bar

    # --------------------------------------------------------------------------
    # 4. AKILLI SES ANALİZİ VE TEŞHİS PANELİ
    # --------------------------------------------------------------------------
    def _build_diagnosis_section(self) -> QWidget:
        card = QFrame()
        card.setObjectName("DiagCard")
        card.setStyleSheet(
            f"#DiagCard {{ background: {C.COLOR_BG_CARD}; border: 1px solid {C.COLOR_BORDER}; border-radius: 8px; }}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        hdr = QHBoxLayout()
        hdr_icon = QLabel()
        hdr_icon.setPixmap(icons.pixmap("activity", 16, C.COLOR_ACCENT_HOVER))
        hdr.addWidget(hdr_icon)

        lbl = QLabel("VOX Ses Teşhis ve Öneri Raporu")
        lbl.setStyleSheet(f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 12px; font-weight: 700;")
        hdr.addWidget(lbl)
        hdr.addStretch(1)

        self.btn_reanalyze = QPushButton("Yeniden Analiz Et")
        self.btn_reanalyze.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reanalyze.clicked.connect(self._run_full_analysis)
        hdr.addWidget(self.btn_reanalyze)
        lay.addLayout(hdr)

        self.lbl_diag_text = QLabel("Ses dosyası yüklendiğinde otomatik gürültü, şebeke uğultusu, kırpılma ve yankı teşhisi burada sunulur.")
        self.lbl_diag_text.setWordWrap(True)
        self.lbl_diag_text.setStyleSheet(f"color: {C.COLOR_TEXT_SECONDARY}; font-size: 11px; line-height: 1.4;")
        lay.addWidget(self.lbl_diag_text)

        return card

    # --------------------------------------------------------------------------
    # 5. AYARLAR VE RESTORASYON KONTROL PANELİ (BASİT / PRO)
    # --------------------------------------------------------------------------
    def _build_settings_panel(self) -> QWidget:
        self.panel_container = QFrame()
        self.panel_container.setObjectName("SettingsPanel")
        self.panel_container.setMinimumWidth(360)
        self.panel_container.setStyleSheet(
            f"#SettingsPanel {{ background: {C.COLOR_BG_SURFACE}; border: 1px solid {C.COLOR_BORDER}; border-radius: 8px; }}"
        )
        self.panel_layout = QVBoxLayout(self.panel_container)
        self.panel_layout.setContentsMargins(12, 12, 12, 12)
        self.panel_layout.setSpacing(10)

        # 1. Basit Mod Görünümü
        self.simple_widget = self._build_simple_mode_view()
        self.panel_layout.addWidget(self.simple_widget)

        # 2. Profesyonel Mod Görünümü (Sekmeli)
        self.pro_tabs = self._build_pro_mode_tabs()
        self.pro_tabs.setVisible(False)
        self.panel_layout.addWidget(self.pro_tabs, 1)

        # İlerleme Çubuğu ve Durum
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setFixedHeight(6)
        self.prog_bar.setTextVisible(False)
        self.prog_bar.setVisible(False)
        self.panel_layout.addWidget(self.prog_bar)

        self.lbl_status = QLabel("Hazır")
        self.lbl_status.setStyleSheet(f"color: {C.COLOR_TEXT_MUTED}; font-size: 11px;")
        self.panel_layout.addWidget(self.lbl_status)

        # Dışa Aktarma Butonları
        export_box = QHBoxLayout()
        export_box.setSpacing(8)

        self.btn_export_wav = QPushButton("WAV Olarak Kaydet")
        self.btn_export_wav.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_wav.setProperty("variant", "primary")
        self.btn_export_wav.clicked.connect(lambda: self._export_audio(fmt="wav"))
        export_box.addWidget(self.btn_export_wav)

        self.btn_export_mp3 = QPushButton("MP3 Olarak Kaydet")
        self.btn_export_mp3.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_mp3.clicked.connect(lambda: self._export_audio(fmt="mp3"))
        export_box.addWidget(self.btn_export_mp3)

        self.panel_layout.addLayout(export_box)

        return self.panel_container

    # --------------------------------------------------------------------------
    # 5.A. BASİT MOD ARAYÜZÜ (TEK TIKLA TEMİZLEME + 3 PROFİL)
    # --------------------------------------------------------------------------
    def _build_simple_mode_view(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        # Büyük Vurgulu "Maksimum Kaliteyle Temizle" Butonu
        self.btn_clean_auto = QPushButton("  Maksimum Kaliteyle Temizle")
        self.btn_clean_auto.setIcon(icons.icon("sparkles", "#FFFFFF", 20))
        self.btn_clean_auto.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clean_auto.setMinimumHeight(48)
        self.btn_clean_auto.setStyleSheet(
            f"QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {C.COLOR_ACCENT}, stop:1 #B71C1C); "
            "color: #FFFFFF; font-weight: 800; font-size: 13px; border-radius: 8px; border: none; letter-spacing: 0.5px; }"
            f"QPushButton:hover {{ background: {C.COLOR_ACCENT_HOVER}; }}"
            f"QPushButton:pressed {{ background: #8E0000; }}"
        )
        self.btn_clean_auto.clicked.connect(self._run_adaptive_restoration)
        lay.addWidget(self.btn_clean_auto)

        lbl_profiles = QLabel("Kalite ve Restorasyon Profili:")
        lbl_profiles.setStyleSheet(f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 11px; font-weight: 700;")
        lay.addWidget(lbl_profiles)

        # 3 Temel Profil Seçim Grubu
        self.profile_group = QButtonGroup(self)

        self.radio_natural = QRadioButton("Natural — Doğal Konuşma")
        self.radio_natural.setToolTip("Hafif temizlik; nefesleri ve doğal dinamikleri korur (-19 LUFS)")
        self.radio_natural.toggled.connect(lambda c: c and self._apply_preset_profile("natural"))
        self.profile_group.addButton(self.radio_natural)
        lay.addWidget(self.radio_natural)

        self.radio_balanced = QRadioButton("Balanced — Dengeli Restorasyon (Önerilen)")
        self.radio_balanced.setChecked(True)
        self.radio_balanced.setToolTip("Fan, klima ve dip seslerini doğal ses dengesiyle temizler (-16 LUFS)")
        self.radio_balanced.toggled.connect(lambda c: c and self._apply_preset_profile("balanced"))
        self.profile_group.addButton(self.radio_balanced)
        lay.addWidget(self.radio_balanced)

        self.radio_max = QRadioButton("Maximum Restoration — Ağır Gürültü Kurtarma")
        self.radio_max.setToolTip("Yoğun uğultu, sokak gürültüsü ve sert ortamlar için derin temizlik")
        self.radio_max.toggled.connect(lambda c: c and self._apply_preset_profile("maximum"))
        self.profile_group.addButton(self.radio_max)
        lay.addWidget(self.radio_max)

        # Temizleme Gücü Slider
        strength_box = QVBoxLayout()
        strength_box.setSpacing(4)
        s_hdr = QHBoxLayout()
        lbl_s = QLabel("AI Gürültü Azaltma Gücü:")
        lbl_s.setStyleSheet(f"color: {C.COLOR_TEXT_SECONDARY}; font-size: 11px;")
        self.lbl_simple_strength_val = QLabel("80%")
        self.lbl_simple_strength_val.setStyleSheet(f"color: {C.COLOR_ACCENT_HOVER}; font-weight: 700; font-size: 11px;")
        s_hdr.addWidget(lbl_s)
        s_hdr.addStretch(1)
        s_hdr.addWidget(self.lbl_simple_strength_val)
        strength_box.addLayout(s_hdr)

        self.slider_simple_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_simple_strength.setRange(10, 100)
        self.slider_simple_strength.setValue(80)
        self.slider_simple_strength.valueChanged.connect(self._on_simple_strength_changed)
        strength_box.addWidget(self.slider_simple_strength)
        lay.addLayout(strength_box)

        # Seçili profili hemen uygula butonu
        self.btn_apply_profile = QPushButton("Seçili Profili Uygula")
        self.btn_apply_profile.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply_profile.clicked.connect(self._apply_current_chain)
        lay.addWidget(self.btn_apply_profile)

        lay.addStretch(1)
        return w

    # --------------------------------------------------------------------------
    # 5.B. PROFESYONEL MOD ARAYÜZÜ (TÜM MODÜLLERİ AYRI AYRI KONTROL ETME)
    # --------------------------------------------------------------------------
    def _build_pro_mode_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid {C.COLOR_BORDER}; background: {C.COLOR_BG_CARD}; border-radius: 6px; }}"
            f"QTabBar::tab {{ background: {C.COLOR_BG_SURFACE}; color: {C.COLOR_TEXT_MUTED}; padding: 6px 10px; font-size: 11px; }}"
            f"QTabBar::tab:selected {{ background: {C.COLOR_BG_CARD}; color: {C.COLOR_TEXT_PRIMARY}; font-weight: 700; border-bottom: 2px solid {C.COLOR_ACCENT}; }}"
        )

        tabs.addTab(self._build_pro_ai_tab(), "1. AI Denoise")
        tabs.addTab(self._build_pro_dehum_tab(), "2. De-Hum")
        tabs.addTab(self._build_pro_declip_tab(), "3. De-Clip/Click")
        tabs.addTab(self._build_pro_dereverb_tab(), "4. De-Reverb")
        tabs.addTab(self._build_pro_clarity_tab(), "5. Clarity/EQ")
        tabs.addTab(self._build_pro_dynamics_tab(), "6. Dinamikler")
        tabs.addTab(self._build_pro_normalize_tab(), "7. Normalizasyon")
        tabs.addTab(self._build_pro_metrics_tab(), "8. Kalite Raporu")

        return tabs

    def _build_pro_ai_tab(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.chk_ai_enabled = QCheckBox("Yapay Zeka Konuşma Temizleme Etkin")
        self.chk_ai_enabled.setChecked(True)
        lay.addRow(self.chk_ai_enabled)

        self.combo_ai_model = QComboBox()
        self.combo_ai_model.addItems(["GTCRN (Glance & Gaze Targeted CRN)", "RNNoise (22-Bark Recurrent)"])
        lay.addRow("Model:", self.combo_ai_model)

        self.spin_ai_strength = QDoubleSpinBox()
        self.spin_ai_strength.setRange(0.0, 1.0)
        self.spin_ai_strength.setSingleStep(0.05)
        self.spin_ai_strength.setValue(0.80)
        lay.addRow("Güç (Strength):", self.spin_ai_strength)

        self.chk_ai_preserve_hi = QCheckBox("8 kHz Üstü Tiz Frekansları Koru (Air & Crossover)")
        self.chk_ai_preserve_hi.setChecked(True)
        lay.addRow(self.chk_ai_preserve_hi)

        self.spin_ai_blend = QDoubleSpinBox()
        self.spin_ai_blend.setRange(0.0, 1.0)
        self.spin_ai_blend.setSingleStep(0.05)
        self.spin_ai_blend.setValue(1.0)
        lay.addRow("Miks (Blend):", self.spin_ai_blend)

        return w

    def _build_pro_dehum_tab(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.chk_hum_enabled = QCheckBox("De-Hum Şebeke Filtresi Etkin")
        lay.addRow(self.chk_hum_enabled)

        self.combo_hum_freq = QComboBox()
        self.combo_hum_freq.addItems(["50 Hz (Türkiye / Avrupa)", "60 Hz (ABD / Diğer)"])
        lay.addRow("Temel Frekans:", self.combo_hum_freq)

        self.spin_hum_harmonics = QSpinBox()
        self.spin_hum_harmonics.setRange(1, 5)
        self.spin_hum_harmonics.setValue(3)
        lay.addRow("Harmonik Sayısı:", self.spin_hum_harmonics)

        self.spin_hum_q = QDoubleSpinBox()
        self.spin_hum_q.setRange(5.0, 60.0)
        self.spin_hum_q.setValue(30.0)
        lay.addRow("Çentik Keskinliği (Q):", self.spin_hum_q)

        return w

    def _build_pro_declip_tab(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.chk_declip_enabled = QCheckBox("De-Clip Kırpılma Onarımı Etkin")
        self.chk_declip_enabled.setChecked(True)
        lay.addRow(self.chk_declip_enabled)

        self.spin_declip_thresh = QDoubleSpinBox()
        self.spin_declip_thresh.setRange(-3.0, 0.0)
        self.spin_declip_thresh.setValue(-0.1)
        lay.addRow("Kırpılma Eşiği (dBFS):", self.spin_declip_thresh)

        self.chk_declick_enabled = QCheckBox("De-Click / De-Crackle Etkin")
        self.chk_declick_enabled.setChecked(True)
        lay.addRow(self.chk_declick_enabled)

        self.spin_declick_sens = QDoubleSpinBox()
        self.spin_declick_sens.setRange(0.1, 1.0)
        self.spin_declick_sens.setValue(0.6)
        lay.addRow("Tıkırtı Hassasiyeti:", self.spin_declick_sens)

        self.chk_plosive_enabled = QCheckBox("Plosive / De-Pop (Nefes Vuruşu) Filtresi")
        self.chk_plosive_enabled.setChecked(True)
        lay.addRow(self.chk_plosive_enabled)

        return w

    def _build_pro_dereverb_tab(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.chk_dereverb_enabled = QCheckBox("De-Reverb (Oda Yankısı Azaltma) Etkin")
        lay.addRow(self.chk_dereverb_enabled)

        self.spin_rt60 = QDoubleSpinBox()
        self.spin_rt60.setRange(0.1, 1.5)
        self.spin_rt60.setValue(0.40)
        self.spin_rt60.setSuffix(" s")
        lay.addRow("Tahmini RT60 Süresi:", self.spin_rt60)

        self.spin_dereverb_amount = QDoubleSpinBox()
        self.spin_dereverb_amount.setRange(0.0, 1.0)
        self.spin_dereverb_amount.setValue(0.50)
        lay.addRow("Bastırma Gücü (Amount):", self.spin_dereverb_amount)

        self.spin_dereverb_floor = QDoubleSpinBox()
        self.spin_dereverb_floor.setRange(0.05, 0.5)
        self.spin_dereverb_floor.setValue(0.20)
        lay.addRow("Spektral Taban (Floor):", self.spin_dereverb_floor)

        return w

    def _build_pro_clarity_tab(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.chk_clarity_enabled = QCheckBox("Speech Clarity Engine Etkin")
        self.chk_clarity_enabled.setChecked(True)
        lay.addRow(self.chk_clarity_enabled)

        self.spin_presence = QDoubleSpinBox()
        self.spin_presence.setRange(-2.0, 6.0)
        self.spin_presence.setValue(2.5)
        self.spin_presence.setSuffix(" dB")
        lay.addRow("3.5 kHz Konuşma Varlığı:", self.spin_presence)

        self.spin_boxiness = QDoubleSpinBox()
        self.spin_boxiness.setRange(-6.0, 0.0)
        self.spin_boxiness.setValue(-2.0)
        self.spin_boxiness.setSuffix(" dB")
        lay.addRow("350 Hz Kutu Hissi Kesimi:", self.spin_boxiness)

        self.spin_air = QDoubleSpinBox()
        self.spin_air.setRange(0.0, 4.0)
        self.spin_air.setValue(1.5)
        self.spin_air.setSuffix(" dB")
        lay.addRow("12 kHz Parlaklık / Hava Rafı:", self.spin_air)

        return w

    def _build_pro_dynamics_tab(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.chk_deesser_enabled = QCheckBox("De-Esser (Sibilans Sertliği Azaltma)")
        self.chk_deesser_enabled.setChecked(True)
        lay.addRow(self.chk_deesser_enabled)

        self.chk_comp_enabled = QCheckBox("Dinamik Kompresör Etkin")
        self.chk_comp_enabled.setChecked(True)
        lay.addRow(self.chk_comp_enabled)

        self.spin_comp_thresh = QDoubleSpinBox()
        self.spin_comp_thresh.setRange(-40.0, 0.0)
        self.spin_comp_thresh.setValue(-16.0)
        self.spin_comp_thresh.setSuffix(" dBFS")
        lay.addRow("Kompresör Eşiği:", self.spin_comp_thresh)

        self.spin_comp_ratio = QDoubleSpinBox()
        self.spin_comp_ratio.setRange(1.5, 8.0)
        self.spin_comp_ratio.setValue(3.0)
        lay.addRow("Kompresör Oranı (Ratio):", self.spin_comp_ratio)

        self.chk_limiter_enabled = QCheckBox("Brickwall Tepe Limiter (-1.0 dBFS Tavan)")
        self.chk_limiter_enabled.setChecked(True)
        lay.addRow(self.chk_limiter_enabled)

        return w

    def _build_pro_normalize_tab(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.chk_norm_lufs = QCheckBox("ITU-R BS.1770-4 Entegre LUFS Normalizasyonu")
        self.chk_norm_lufs.setChecked(True)
        lay.addRow(self.chk_norm_lufs)

        self.combo_target_lufs = QComboBox()
        self.combo_target_lufs.addItem("Podcast / Mono Konuşma: -19.0 LUFS", -19.0)
        self.combo_target_lufs.addItem("Video / Dublaj / Yayın: -16.0 LUFS", -16.0)
        self.combo_target_lufs.addItem("Müzik / Streaming: -14.0 LUFS", -14.0)
        self.combo_target_lufs.setCurrentIndex(1)
        lay.addRow("Hedef Seviye:", self.combo_target_lufs)

        return w

    def _build_pro_metrics_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        self.table_metrics = QTableWidget(6, 3)
        self.table_metrics.setHorizontalHeaderLabels(["Metrik", "Orijinal (A)", "Temizlenmiş (B)"])
        self.table_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_metrics.verticalHeader().setVisible(False)
        self.table_metrics.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        metrics = [
            "Tepe Seviyesi (Peak dBFS)",
            "Ortalama Seviye (RMS dBFS)",
            "Entegre Seslilik (LUFS)",
            "Gürültü Tabanı (Noise Floor)",
            "Sinyal/Gürültü Oranı (SNR)",
            "Kırpılma (Clipping Sayısı)",
        ]
        for row, name in enumerate(metrics):
            self.table_metrics.setItem(row, 0, QTableWidgetItem(name))
            self.table_metrics.setItem(row, 1, QTableWidgetItem("—"))
            self.table_metrics.setItem(row, 2, QTableWidgetItem("—"))

        lay.addWidget(self.table_metrics)
        return w

    # ==========================================================================
    # İŞLEVSELLİK VE OLAY YÖNETİMİ
    # ==========================================================================
    def _wire_signals(self) -> None:
        pass

    def _set_mode(self, simple: bool) -> None:
        self._is_simple_mode = simple
        self.btn_mode_simple.setChecked(simple)
        self.btn_mode_pro.setChecked(not simple)
        self.simple_widget.setVisible(simple)
        self.pro_tabs.setVisible(not simple)

    def _apply_preset_profile(self, name: str) -> None:
        if name == "natural":
            self.config = create_natural_profile()
        elif name == "maximum":
            self.config = create_maximum_restoration_profile()
        else:
            self.config = create_balanced_profile()

        self._sync_ui_to_config()
        self.lbl_status.setText(f"'{name.capitalize()}' profili seçildi.")

    def _on_simple_strength_changed(self, val: int) -> None:
        strength = val / 100.0
        self.lbl_simple_strength_val.setText(f"{val}%")
        self.config.ai_denoise.strength = strength

    def _sync_ui_to_config(self) -> None:
        # Basit slider
        val = int(self.config.ai_denoise.strength * 100)
        self.slider_simple_strength.setValue(val)
        self.lbl_simple_strength_val.setText(f"{val}%")

        # Pro kontroller
        self.chk_ai_enabled.setChecked(self.config.ai_denoise.enabled)
        self.spin_ai_strength.setValue(self.config.ai_denoise.strength)
        self.chk_ai_preserve_hi.setChecked(self.config.ai_denoise.preserve_high_frequencies)
        self.spin_ai_blend.setValue(self.config.ai_denoise.blend)

        self.chk_hum_enabled.setChecked(self.config.hum_filter.enabled)
        self.spin_hum_harmonics.setValue(self.config.hum_filter.harmonics)

        self.chk_declip_enabled.setChecked(self.config.declip.enabled)
        self.chk_declick_enabled.setChecked(self.config.declick.enabled)

        self.chk_dereverb_enabled.setChecked(self.config.dereverb.enabled)
        self.spin_rt60.setValue(self.config.dereverb.rt60)
        self.spin_dereverb_amount.setValue(self.config.dereverb.amount)

        self.chk_clarity_enabled.setChecked(self.config.clarity.enabled)
        self.spin_presence.setValue(self.config.clarity.presence_gain_db)
        self.spin_boxiness.setValue(self.config.clarity.boxiness_cut_db)
        self.spin_air.setValue(self.config.clarity.air_shelf_db)

        self.chk_comp_enabled.setChecked(self.config.compressor.enabled)
        self.spin_comp_thresh.setValue(self.config.compressor.threshold_dbfs)

    def _collect_config_from_pro_ui(self) -> None:
        self.config.ai_denoise.enabled = self.chk_ai_enabled.isChecked()
        self.config.ai_denoise.strength = self.spin_ai_strength.value()
        self.config.ai_denoise.preserve_high_frequencies = self.chk_ai_preserve_hi.isChecked()
        self.config.ai_denoise.blend = self.spin_ai_blend.value()

        self.config.hum_filter.enabled = self.chk_hum_enabled.isChecked()
        self.config.hum_filter.base_freq_hz = 50.0 if self.combo_hum_freq.currentIndex() == 0 else 60.0
        self.config.hum_filter.harmonics = self.spin_hum_harmonics.value()
        self.config.hum_filter.q = self.spin_hum_q.value()

        self.config.declip.enabled = self.chk_declip_enabled.isChecked()
        self.config.declick.enabled = self.chk_declick_enabled.isChecked()
        self.config.plosive.enabled = self.chk_plosive_enabled.isChecked()

        self.config.dereverb.enabled = self.chk_dereverb_enabled.isChecked()
        self.config.dereverb.rt60 = self.spin_rt60.value()
        self.config.dereverb.amount = self.spin_dereverb_amount.value()
        self.config.dereverb.spectral_floor = self.spin_dereverb_floor.value()

        self.config.clarity.enabled = self.chk_clarity_enabled.isChecked()
        self.config.clarity.presence_gain_db = self.spin_presence.value()
        self.config.clarity.boxiness_cut_db = self.spin_boxiness.value()
        self.config.clarity.air_shelf_db = self.spin_air.value()

        self.config.compressor.enabled = self.chk_comp_enabled.isChecked()
        self.config.compressor.threshold_dbfs = self.spin_comp_thresh.value()
        self.config.compressor.ratio = self.spin_comp_ratio.value()

        self.config.normalize_lufs_target = (
            float(self.combo_target_lufs.currentData()) if self.chk_norm_lufs.isChecked() else None
        )

    # --------------------------------------------------------------------------
    # DOSYA YÜKLEME VE SÜRÜKLE BIRAK
    # --------------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            self.load_file(path)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Ses Dosyası Seç",
            "",
            "Ses Dosyaları (*.wav *.flac *.mp3 *.ogg *.m4a *.aac);;Tüm Dosyalar (*.*)",
        )
        if path:
            self.load_file(Path(path))

    def _load_last_recording(self) -> None:
        from core import paths
        takes_dir = paths.takes_dir()
        if not takes_dir.exists():
            QMessageBox.information(self, "Bilgi", "Kayıt klasörü henüz bulunmuyor.")
            return

        wavs = sorted(takes_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not wavs:
            QMessageBox.information(self, "Bilgi", "Henüz alınmış bir kayıt bulunamadı.")
            return

        self.load_file(wavs[0])

    def load_file(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False

        try:
            audio, fs = sf.read(str(p), dtype="float32")
        except Exception:
            try:
                from ai.media_extractor import MediaExtractor
                audio, fs = MediaExtractor.read_audio_any(p, always_2d=False)
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Dosya açılamadı: {e}")
                return False

        self._raw_audio = audio
        self._processed_audio = None
        self._fs = float(fs)
        self._current_path = p

        # Header Güncelle
        self.lbl_file_name.setText(p.name)
        ch_str = "Stereo" if (audio.ndim == 2 and audio.shape[1] >= 2) else "Mono"
        dur_str = format_duration(len(audio) / fs)
        size_str = format_file_size(p.stat().st_size)
        self.lbl_file_meta.setText(f"{int(fs)} Hz • {ch_str} • {dur_str} • {size_str}")

        # Waveform Güncelle
        self._refresh_waveform(self._raw_audio)

        # A/B Player'a Yükle (İlk başta Orijinal = A, B henüz yok)
        self.ab_player.load(self._raw_audio, self._raw_audio, int(fs))
        self.btn_ab.setChecked(False)
        self.btn_ab.setText("A: Orijinal Sesi Dinle")

        # Otomatik Teşhis Çalıştır
        self._run_full_analysis()
        return True

    def _refresh_waveform(self, audio: np.ndarray) -> None:
        mins, maxs = peaks_from_samples(audio, points=1600)
        dur = len(audio) / self._fs
        self.waveform_view.set_peaks(mins, maxs, dur)

    # --------------------------------------------------------------------------
    # AKILLI ANALİZ VE RESTORASYON ÇALIŞTIRMA
    # --------------------------------------------------------------------------
    def _run_full_analysis(self) -> None:
        if self._raw_audio is None:
            return

        self._analysis = analyze_array(self._raw_audio, int(self._fs))
        self.lbl_diag_text.setText(self._analysis.recommendations_summary)
        self._update_metrics_table()

    def _run_adaptive_restoration(self) -> None:
        """Kullanıcının tek tıkla 'Maksimum Kaliteyle Temizle' aksiyonu."""
        if self._raw_audio is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir ses kaydı veya dosyası seçin.")
            return

        if self._analysis is None:
            self._analysis = analyze_array(self._raw_audio, int(self._fs))

        # Teşhise göre en iyi parametreleri dinamik oluştur
        self.config = create_adaptive_profile(self._analysis)
        self._sync_ui_to_config()
        self._start_worker(is_snippet=False)

    def _apply_current_chain(self) -> None:
        if self._raw_audio is None:
            return
        if not self._is_simple_mode:
            self._collect_config_from_pro_ui()
        self._start_worker(is_snippet=False)

    def _run_snippet_preview(self) -> None:
        """Yalnızca seçilen bölgeyi saliseler içinde restore eder ve çalar."""
        if self._raw_audio is None:
            return

        in_sec = self.waveform_view.selection_in()
        out_sec = self.waveform_view.selection_out()
        if in_sec is None or out_sec is None or (out_sec - in_sec) < 0.2:
            # Seçim yoksa ilk 4 saniyeyi kesit yap
            in_sec = 0.0
            out_sec = min(4.0, len(self._raw_audio) / self._fs)

        start_idx = int(in_sec * self._fs)
        end_idx = int(out_sec * self._fs)
        snippet_data = self._raw_audio[start_idx:end_idx]

        if not self._is_simple_mode:
            self._collect_config_from_pro_ui()

        self._start_worker(is_snippet=True, audio_override=snippet_data)

    def _start_worker(self, is_snippet: bool = False, audio_override: Optional[np.ndarray] = None) -> None:
        audio_to_process = audio_override if audio_override is not None else self._raw_audio
        if audio_to_process is None:
            return

        self.btn_clean_auto.setEnabled(False)
        self.prog_bar.setVisible(True)
        self.prog_bar.setValue(0)
        self.lbl_status.setText("Yapay Zeka ve DSP Restorasyon Motoru devrede...")

        self._worker = DSPProcessWorker(audio_to_process, self._fs, self.config, is_snippet=is_snippet)
        self._worker.progressSignal.connect(lambda p: self.prog_bar.setValue(int(p * 100)))
        self._worker.finishedSignal.connect(self._on_worker_finished)
        self._worker.errorSignal.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_finished(self, processed: np.ndarray, fs: float, stats: dict, is_snippet: bool) -> None:
        self.btn_clean_auto.setEnabled(True)
        self.prog_bar.setVisible(False)

        stages_run = " -> ".join(stats.get("stages_executed", []))
        self.lbl_status.setText(f"Restorasyon tamamlandı: {stages_run}")

        if is_snippet:
            # Sadece önizleme: A/B oynatıcıya bağla ve hemen dinlet
            self.ab_player.load(self._raw_audio, processed, int(fs))
            self.ab_player.switch_channel(ABChannel.B)
            self.btn_ab.setChecked(True)
            self.btn_ab.setText("B: Temizlenmiş Sesi Dinle")
            self.ab_player.play()
            return

        # Tam dosya restorasyonu
        self._processed_audio = processed
        self._refresh_waveform(processed)

        # A/B oynatıcıyı güncelle
        self.ab_player.load(self._raw_audio, self._processed_audio, int(fs))
        self.ab_player.switch_channel(ABChannel.B)
        self.btn_ab.setChecked(True)
        self.btn_ab.setText("B: Temizlenmiş Sesi Dinle")

        # Metrik tablosunu güncelle
        self._update_metrics_table()

    def _on_worker_error(self, err_msg: str) -> None:
        self.btn_clean_auto.setEnabled(True)
        self.prog_bar.setVisible(False)
        self.lbl_status.setText(f"İşlem hatası: {err_msg}")
        QMessageBox.critical(self, "Hata", f"Restorasyon sırasında hata oluştu:\n{err_msg}")

    # --------------------------------------------------------------------------
    # OYNATMA VE A/B GEÇİŞİ
    # --------------------------------------------------------------------------
    def _toggle_playback(self) -> None:
        if self.ab_player.is_playing:
            self.ab_player.pause()
            self.btn_play.setIcon(icons.icon("play", C.COLOR_TEXT_PRIMARY, 16))
        else:
            self.ab_player.play()
            self.btn_play.setIcon(icons.icon("pause", C.COLOR_TEXT_PRIMARY, 16))

    def _stop_playback(self) -> None:
        self.ab_player.stop()
        self.btn_play.setIcon(icons.icon("play", C.COLOR_TEXT_PRIMARY, 16))
        self.waveform_view.set_playhead(0.0)
        self.lbl_time.setText(f"00:00.00 / {format_duration(self.ab_player.duration)}")

    def _on_ab_toggled(self, checked: bool) -> None:
        if checked:
            self.ab_player.switch_channel(ABChannel.B)
            self.btn_ab.setText("B: Temizlenmiş Sesi Dinle")
            if self._processed_audio is not None:
                self._refresh_waveform(self._processed_audio)
        else:
            self.ab_player.switch_channel(ABChannel.A)
            self.btn_ab.setText("A: Orijinal Sesi Dinle")
            if self._raw_audio is not None:
                self._refresh_waveform(self._raw_audio)

    def _on_ab_position_update(self, cur_sec: float, total_sec: float) -> None:
        self.waveform_view.set_playhead(cur_sec)
        self.lbl_time.setText(f"{format_duration(cur_sec)} / {format_duration(total_sec)}")

    def _on_ab_playback_finished(self) -> None:
        self.btn_play.setIcon(icons.icon("play", C.COLOR_TEXT_PRIMARY, 16))
        self.waveform_view.set_playhead(0.0)

    def _on_waveform_seek(self, ratio: float) -> None:
        if self.ab_player.duration > 0:
            target_sec = ratio * self.ab_player.duration
            self.ab_player.seek(target_sec)
            self.waveform_view.set_playhead(target_sec)

    def _on_waveform_selection_changed(self, in_sec: Optional[float], out_sec: Optional[float]) -> None:
        if in_sec is not None and out_sec is not None:
            dur = max(0.0, out_sec - in_sec)
            self.lbl_sel_info.setText(f"Seçim: {format_duration(in_sec)} — {format_duration(out_sec)} ({dur:.2f}s)")
        else:
            self.lbl_sel_info.setText("Seçim: Yok")

    # --------------------------------------------------------------------------
    # METRİK TABLOSU GÜNCELLEME
    # --------------------------------------------------------------------------
    def _update_metrics_table(self) -> None:
        if self._raw_audio is None:
            return

        m_raw = compute_detailed_metrics(self._raw_audio, self._fs)
        self.table_metrics.setItem(0, 1, QTableWidgetItem(f"{m_raw['peak_dbfs']} dBFS"))
        self.table_metrics.setItem(1, 1, QTableWidgetItem(f"{m_raw['rms_dbfs']} dBFS"))
        self.table_metrics.setItem(2, 1, QTableWidgetItem(f"{m_raw['lufs']} LUFS"))
        self.table_metrics.setItem(3, 1, QTableWidgetItem(f"{m_raw['noise_floor_dbfs']} dBFS"))
        snr_raw = f"{m_raw['snr_db']} dB" if m_raw['snr_db'] is not None else "—"
        self.table_metrics.setItem(4, 1, QTableWidgetItem(snr_raw))
        self.table_metrics.setItem(5, 1, QTableWidgetItem(str(m_raw['clipping_samples'])))

        if self._processed_audio is not None:
            m_proc = compute_detailed_metrics(self._processed_audio, self._fs)
            self.table_metrics.setItem(0, 2, QTableWidgetItem(f"{m_proc['peak_dbfs']} dBFS"))
            self.table_metrics.setItem(1, 2, QTableWidgetItem(f"{m_proc['rms_dbfs']} dBFS"))
            self.table_metrics.setItem(2, 2, QTableWidgetItem(f"{m_proc['lufs']} LUFS"))
            self.table_metrics.setItem(3, 2, QTableWidgetItem(f"{m_proc['noise_floor_dbfs']} dBFS"))
            snr_proc = f"{m_proc['snr_db']} dB" if m_proc['snr_db'] is not None else "—"
            self.table_metrics.setItem(4, 2, QTableWidgetItem(snr_proc))
            self.table_metrics.setItem(5, 2, QTableWidgetItem(str(m_proc['clipping_samples'])))

    # --------------------------------------------------------------------------
    # DIŞA AKTARMA (WAV, MP3, FLAC)
    # --------------------------------------------------------------------------
    def _export_audio(self, fmt: str = "wav") -> None:
        audio_to_export = self._processed_audio if self._processed_audio is not None else self._raw_audio
        if audio_to_export is None:
            QMessageBox.warning(self, "Uyarı", "Dışa aktarılacak ses verisi bulunmuyor.")
            return

        def_name = f"{self._current_path.stem}_temiz" if self._current_path else "vox_restored"
        if fmt == "mp3":
            filter_str = "MP3 Dosyası (*.mp3)"
            out_path_str, _ = QFileDialog.getSaveFileName(self, "MP3 Olarak Kaydet", f"{def_name}.mp3", filter_str)
            if not out_path_str:
                return
            export_mp3_pyav(audio_to_export, int(self._fs), out_path_str, bitrate_kbps=320)
        else:
            filter_str = "WAV Dosyası (*.wav);;FLAC Dosyası (*.flac)"
            out_path_str, _ = QFileDialog.getSaveFileName(self, "WAV Olarak Kaydet", f"{def_name}.wav", filter_str)
            if not out_path_str:
                return
            if out_path_str.endswith(".flac"):
                export_flac(audio_to_export, int(self._fs), out_path_str, bit_depth=24)
            else:
                export_wav(audio_to_export, int(self._fs), out_path_str, bit_depth=24)

        info = validate_exported_file(out_path_str)
        QMessageBox.information(
            self,
            "Başarılı",
            f"Dosya başarıyla kaydedildi:\n{out_path_str}\n\nFormat: {info['format']} • {info['size_mb']} MB • {format_duration(info['duration'])}",
        )
