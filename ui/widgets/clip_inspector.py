"""
VOX Studio — CapCut Tarzı Bağlama Duyarlı Sağ Özellikler (Inspector) ve Proje Detayları Paneli
Seçili öğe olmadığında Proje Detaylarını, bir klip seçildiğinde ise sekmeli modern
CapCut arayüzüyle Video (Dönüşüm, Konum, Ölçek, Opaklık, Renk), Ses (Seviye dB, Fade, Pan, AI Ayrıştırma),
Hız (0.25x-4.0x) ve Altyazı/Metin ayarlarını gösterir.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import soundfile as sf
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai.media_extractor import MediaExtractor
from ai.stem_separator import StemEnhancer
from core import constants as C
from core.logger import get_logger
from ui import icons
from video_editor.models import ClipType, TimelineClip, TimelineProject

logger = get_logger("ui.widgets.inspector")


class ClipInspectorWidget(QWidget):
    """Seçili klibin video, ses, altyazı veya proje detaylarını düzenleyen profesyonel sağ panel."""

    clipChanged = Signal(object)
    separateAudioRequested = Signal(object)  # clip

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_clip: Optional[TimelineClip] = None
        self.current_project: Optional[TimelineProject] = None
        self._updating_ui = False
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ------------------------------------------------------------------ Üst Başlık
        title_box = QHBoxLayout()
        self.header_title = QLabel("Proje Detayları")
        self.header_title.setObjectName("SectionTitle")
        self.header_title.setStyleSheet("font-weight: 700; font-size: 13px; color: #F8FAFC;")
        title_box.addWidget(self.header_title)
        title_box.addStretch(1)
        root.addLayout(title_box)

        # ------------------------------------------------------------------ 0. Proje Detayları Kartı
        self.project_details_card = QFrame()
        self.project_details_card.setObjectName("Card")
        pd_layout = QVBoxLayout(self.project_details_card)
        pd_layout.setContentsMargins(12, 10, 12, 10)
        pd_layout.setSpacing(6)

        self.pd_name_lbl = QLabel("Ad: Yeni Dublaj Projesi")
        self.pd_name_lbl.setWordWrap(True)
        self.pd_name_lbl.setStyleSheet("color: #F2F4F8; font-weight: 600; font-size: 12px;")
        pd_layout.addWidget(self.pd_name_lbl)

        self.pd_specs_lbl = QLabel(
            "Oran: 16:9 Geniş\n"
            "Çözünürlük: 1920×1080 (Full HD)\n"
            "Kare Hızı: 30.00 fps\n"
            "Renk Uzayı: Rec.709 SDR\n"
            "Otomatik Kayıt: Aktif (Yerel)"
        )
        self.pd_specs_lbl.setWordWrap(True)
        self.pd_specs_lbl.setStyleSheet("color: #9CA6B8; font-size: 11px; line-height: 1.4;")
        pd_layout.addWidget(self.pd_specs_lbl)

        self.pd_hint_lbl = QLabel(
            "Düzenlemek için zaman çizelgesinden bir video, ses veya altyazı klibine tıklayın."
        )
        self.pd_hint_lbl.setWordWrap(True)
        self.pd_hint_lbl.setStyleSheet("color: #6C7687; font-size: 11px; padding-top: 4px;")
        pd_layout.addWidget(self.pd_hint_lbl)
        root.addWidget(self.project_details_card)

        # ------------------------------------------------------------------ Klip Düzenleme Konteyneri
        self.clip_container = QWidget()
        c_layout = QVBoxLayout(self.clip_container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(8)

        # Klip Başlık ve Metadata Kartı
        self.info_card = QFrame()
        self.info_card.setObjectName("Card")
        i_layout = QVBoxLayout(self.info_card)
        i_layout.setContentsMargins(10, 8, 10, 8)
        i_layout.setSpacing(5)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Klip Adı...")
        self.name_edit.textChanged.connect(self._on_name_changed)
        i_layout.addWidget(self.name_edit)

        self.dur_label = QLabel("Süre: 0.00s | Giriş: 0.00s | Çıkış: 0.00s")
        self.dur_label.setWordWrap(True)
        self.dur_label.setStyleSheet("color: #38BDF8; font-size: 11px; font-weight: 600;")
        i_layout.addWidget(self.dur_label)

        self.source_meta_label = QLabel("")
        self.source_meta_label.setWordWrap(True)
        self.source_meta_label.setStyleSheet("color: #6C7687; font-size: 10px;")
        i_layout.addWidget(self.source_meta_label)
        c_layout.addWidget(self.info_card)

        # ------------------------------------------------------------------ CapCut Sekmeleri
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid {C.COLOR_BORDER}; background: #131417; border-radius: 6px; }} "
            f"QTabBar::tab {{ background: #101114; color: #9CA6B8; padding: 6px 12px; font-size: 11px; font-weight: 600; border: none; border-bottom: 2px solid transparent; }} "
            f"QTabBar::tab:selected {{ color: #F8FAFC; border-bottom: 2px solid {C.COLOR_ACCENT}; background: #18191E; }} "
            f"QTabBar::tab:hover {{ color: #FFFFFF; }}"
        )

        # ================================================================== 1. VIDEO SEKMESİ
        self.tab_video = QWidget()
        tv_layout = QVBoxLayout(self.tab_video)
        tv_layout.setContentsMargins(8, 8, 8, 8)
        tv_layout.setSpacing(8)

        # Dönüşüm (Transform) Grubu
        trans_box = QGroupBox("Dönüşüm (Transform)")
        t_lay = QVBoxLayout(trans_box)
        t_lay.setSpacing(6)

        # Ölçek (Scale)
        sc_row = QHBoxLayout()
        sc_lbl = QLabel("Ölçek:")
        sc_lbl.setFixedWidth(50)
        sc_row.addWidget(sc_lbl)

        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(10, 300)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self._on_scale_slider_changed)
        sc_row.addWidget(self.scale_slider, 1)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(10.0, 500.0)
        self.scale_spin.setSingleStep(5.0)
        self.scale_spin.setValue(100.0)
        self.scale_spin.setSuffix("%")
        self.scale_spin.setFixedWidth(70)
        self.scale_spin.valueChanged.connect(self._on_scale_spin_changed)
        sc_row.addWidget(self.scale_spin)

        self.scale_reset_btn = QPushButton("↺")
        self.scale_reset_btn.setFixedSize(22, 22)
        self.scale_reset_btn.setToolTip("Ölçeği %100'e Sıfırla")
        self.scale_reset_btn.clicked.connect(lambda: self.scale_spin.setValue(100.0))
        sc_row.addWidget(self.scale_reset_btn)
        t_lay.addLayout(sc_row)

        self.uniform_scale_check = QCheckBox("Orantılı Ölçekle (Uniform)")
        self.uniform_scale_check.setChecked(True)
        self.uniform_scale_check.setStyleSheet("color: #9CA6B8; font-size: 11px;")
        t_lay.addWidget(self.uniform_scale_check)

        # Konum (Position X & Y)
        pos_row = QHBoxLayout()
        pos_lbl = QLabel("Konum:")
        pos_lbl.setFixedWidth(50)
        pos_row.addWidget(pos_lbl)

        pos_row.addWidget(QLabel("X:"))
        self.pos_x_spin = QSpinBox()
        self.pos_x_spin.setRange(-3840, 3840)
        self.pos_x_spin.setValue(0)
        self.pos_x_spin.setSuffix(" px")
        self.pos_x_spin.valueChanged.connect(self._on_transform_changed)
        pos_row.addWidget(self.pos_x_spin, 1)

        pos_row.addWidget(QLabel("Y:"))
        self.pos_y_spin = QSpinBox()
        self.pos_y_spin.setRange(-2160, 2160)
        self.pos_y_spin.setValue(0)
        self.pos_y_spin.setSuffix(" px")
        self.pos_y_spin.valueChanged.connect(self._on_transform_changed)
        pos_row.addWidget(self.pos_y_spin, 1)

        self.pos_reset_btn = QPushButton("↺")
        self.pos_reset_btn.setFixedSize(22, 22)
        self.pos_reset_btn.setToolTip("Konumu Merkeze Sıfırla (0, 0)")
        self.pos_reset_btn.clicked.connect(self._reset_position)
        pos_row.addWidget(self.pos_reset_btn)
        t_lay.addLayout(pos_row)

        # Döndürme (Rotation)
        rot_row = QHBoxLayout()
        rot_lbl = QLabel("Döndürme:")
        rot_lbl.setFixedWidth(65)
        rot_row.addWidget(rot_lbl)

        self.rot_spin = QDoubleSpinBox()
        self.rot_spin.setRange(-360.0, 360.0)
        self.rot_spin.setSingleStep(5.0)
        self.rot_spin.setValue(0.0)
        self.rot_spin.setSuffix("°")
        self.rot_spin.valueChanged.connect(self._on_transform_changed)
        rot_row.addWidget(self.rot_spin, 1)

        for deg in (0, 90, 180, 270):
            r_btn = QPushButton(f"{deg}°")
            r_btn.setProperty("compact", "true")
            r_btn.setFixedWidth(36)
            r_btn.clicked.connect(lambda _, d=deg: self.rot_spin.setValue(d))
            rot_row.addWidget(r_btn)
        t_lay.addLayout(rot_row)

        # Hızlı Hizalama Düğmeleri (Sol, Orta, Sağ, Üst, Dikey Orta, Alt)
        align_row = QHBoxLayout()
        align_row.setSpacing(4)
        align_btns = [
            ("⇤", "Sola Hizala", lambda: self.pos_x_spin.setValue(-300)),
            ("⇥", "Sağa Hizala", lambda: self.pos_x_spin.setValue(300)),
            ("⇡", "Üste Hizala", lambda: self.pos_y_spin.setValue(-200)),
            ("⇣", "Alta Hizala", lambda: self.pos_y_spin.setValue(200)),
            ("⊕", "Merkeze Hizala", self._reset_position),
        ]
        for icon_txt, tip, func in align_btns:
            ab = QPushButton(icon_txt)
            ab.setProperty("compact", "true")
            ab.setToolTip(tip)
            ab.clicked.connect(func)
            align_row.addWidget(ab)
        t_lay.addLayout(align_row)

        # Karışım & Opaklık (Opacity)
        op_row = QHBoxLayout()
        op_lbl = QLabel("Opaklık:")
        op_lbl.setFixedWidth(50)
        op_row.addWidget(op_lbl)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        op_row.addWidget(self.opacity_slider, 1)

        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 100.0)
        self.opacity_spin.setValue(100.0)
        self.opacity_spin.setSuffix("%")
        self.opacity_spin.setFixedWidth(65)
        self.opacity_spin.valueChanged.connect(self._on_opacity_spin_changed)
        op_row.addWidget(self.opacity_spin)
        t_lay.addLayout(op_row)

        tv_layout.addWidget(trans_box)

        # Renk ve Görünüm Ayarı (Color Adjustments)
        color_box = QGroupBox("Görüntü & Renk (Adjustment)")
        cl_lay = QVBoxLayout(color_box)
        cl_lay.setSpacing(5)

        # Parlaklık
        br_row = QHBoxLayout()
        br_row.addWidget(QLabel("Parlaklık:"))
        self.bright_spin = QDoubleSpinBox()
        self.bright_spin.setRange(-1.0, 1.0)
        self.bright_spin.setSingleStep(0.05)
        self.bright_spin.setValue(0.0)
        self.bright_spin.valueChanged.connect(self._on_color_prop_changed)
        br_row.addWidget(self.bright_spin)
        cl_lay.addLayout(br_row)

        # Kontrast
        ct_row = QHBoxLayout()
        ct_row.addWidget(QLabel("Kontrast:"))
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.2, 2.5)
        self.contrast_spin.setSingleStep(0.1)
        self.contrast_spin.setValue(1.0)
        self.contrast_spin.valueChanged.connect(self._on_color_prop_changed)
        ct_row.addWidget(self.contrast_spin)
        cl_lay.addLayout(ct_row)

        # Doygunluk
        sat_row = QHBoxLayout()
        sat_row.addWidget(QLabel("Doygunluk:"))
        self.sat_spin = QDoubleSpinBox()
        self.sat_spin.setRange(0.0, 2.0)
        self.sat_spin.setSingleStep(0.1)
        self.sat_spin.setValue(1.0)
        self.sat_spin.valueChanged.connect(self._on_color_prop_changed)
        sat_row.addWidget(self.sat_spin)
        cl_lay.addLayout(sat_row)

        clr_reset_btn = QPushButton("Renkleri Sıfırla")
        clr_reset_btn.setProperty("compact", "true")
        clr_reset_btn.clicked.connect(self._reset_colors)
        cl_lay.addWidget(clr_reset_btn)

        tv_layout.addWidget(color_box)
        tv_layout.addStretch(1)
        self.tabs.addTab(self.tab_video, "Video")

        # ================================================================== 2. SES SEKMESİ
        self.tab_audio = QWidget()
        ta_layout = QVBoxLayout(self.tab_audio)
        ta_layout.setContentsMargins(8, 8, 8, 8)
        ta_layout.setSpacing(8)

        # Temel Ses Ayarları (Basic Audio)
        basic_audio_box = QGroupBox("Temel Ses Ayarları (Basic)")
        ba_lay = QVBoxLayout(basic_audio_box)
        ba_lay.setSpacing(6)

        # Ses Seviyesi (Volume Slider & dB SpinBox & Reset)
        v_top = QHBoxLayout()
        v_lbl = QLabel("Ses Seviyesi:")
        v_lbl.setStyleSheet("font-weight: 600; color: #F8FAFC;")
        v_top.addWidget(v_lbl)
        v_top.addStretch(1)

        self.vol_db_spin = QDoubleSpinBox()
        self.vol_db_spin.setRange(-60.0, 12.0)
        self.vol_db_spin.setSingleStep(0.5)
        self.vol_db_spin.setValue(0.0)
        self.vol_db_spin.setSuffix(" dB")
        self.vol_db_spin.setFixedWidth(78)
        self.vol_db_spin.valueChanged.connect(self._on_vol_db_spin_changed)
        v_top.addWidget(self.vol_db_spin)

        self.vol_reset_btn = QPushButton("↺ 0 dB")
        self.vol_reset_btn.setProperty("compact", "true")
        self.vol_reset_btn.setToolTip("Ses Seviyesini 0.0 dB'ye Sıfırla")
        self.vol_reset_btn.clicked.connect(lambda: self.vol_db_spin.setValue(0.0))
        v_top.addWidget(self.vol_reset_btn)
        ba_lay.addLayout(v_top)

        # Slider (-60 dB ile +12 dB arası)
        v_slider_row = QHBoxLayout()
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(-60, 12)
        self.vol_slider.setValue(0)
        self.vol_slider.valueChanged.connect(self._on_vol_slider_changed)
        v_slider_row.addWidget(self.vol_slider, 1)

        self.vol_db_label = QLabel("0.0 dB")
        self.vol_db_label.setFixedWidth(52)
        self.vol_db_label.setStyleSheet("color: #38BDF8; font-weight: 700; font-size: 11px;")
        v_slider_row.addWidget(self.vol_db_label)
        ba_lay.addLayout(v_slider_row)

        # Fade In & Fade Out
        fade_row = QHBoxLayout()
        fade_row.addWidget(QLabel("Açılma (Fade In):"))
        self.fade_in_spin = QDoubleSpinBox()
        self.fade_in_spin.setRange(0.0, 10.0)
        self.fade_in_spin.setSingleStep(0.2)
        self.fade_in_spin.setValue(0.0)
        self.fade_in_spin.setSuffix("s")
        self.fade_in_spin.valueChanged.connect(self._on_audio_prop_changed)
        fade_row.addWidget(self.fade_in_spin)

        fade_row.addWidget(QLabel("Kapanma (Out):"))
        self.fade_out_spin = QDoubleSpinBox()
        self.fade_out_spin.setRange(0.0, 10.0)
        self.fade_out_spin.setSingleStep(0.2)
        self.fade_out_spin.setValue(0.0)
        self.fade_out_spin.setSuffix("s")
        self.fade_out_spin.valueChanged.connect(self._on_audio_prop_changed)
        fade_row.addWidget(self.fade_out_spin)
        ba_lay.addLayout(fade_row)

        # Pan (Sol / Sağ Denge)
        pan_row = QHBoxLayout()
        pan_row.addWidget(QLabel("Pan (L/R):"))
        self.pan_slider = QSlider(Qt.Orientation.Horizontal)
        self.pan_slider.setRange(-100, 100)
        self.pan_slider.setValue(0)
        self.pan_slider.valueChanged.connect(self._on_pan_slider_changed)
        pan_row.addWidget(self.pan_slider, 1)

        self.pan_val_label = QLabel("C")
        self.pan_val_label.setFixedWidth(36)
        self.pan_val_label.setStyleSheet("font-weight: 600; color: #9CA6B8;")
        pan_row.addWidget(self.pan_val_label)

        pan_reset_btn = QPushButton("C")
        pan_reset_btn.setProperty("compact", "true")
        pan_reset_btn.setFixedSize(22, 22)
        pan_reset_btn.setToolTip("Panı Merkeze Sıfırla")
        pan_reset_btn.clicked.connect(lambda: self.pan_slider.setValue(0))
        pan_row.addWidget(pan_reset_btn)
        ba_lay.addLayout(pan_row)

        self.mute_check = QCheckBox("Bu Klibin Sesini Sustur (Mute)")
        self.mute_check.setStyleSheet("color: #E53945; font-weight: 600;")
        self.mute_check.toggled.connect(self._on_audio_prop_changed)
        ba_lay.addWidget(self.mute_check)

        ta_layout.addWidget(basic_audio_box)

        # Akıllı İyileştirme ve AI Ayrıştırma Grubu
        ai_audio_box = QGroupBox("Akıllı Ses İyileştirme & AI")
        aa_lay = QVBoxLayout(ai_audio_box)
        aa_lay.setSpacing(6)

        self.ai_sep_btn = QPushButton(" AI ile Vokal & Müzik Ayrıştır")
        self.ai_sep_btn.setProperty("variant", "primary")
        self.ai_sep_btn.setIcon(icons.icon("sparkles", size=14, color="#FFFFFF"))
        self.ai_sep_btn.setToolTip("Bu klibin sesini Diyalog, Ortam ve Müzik katmanlarına ayırıp timeline'a yerleştirir")
        self.ai_sep_btn.clicked.connect(self._on_ai_sep_clicked)
        aa_lay.addWidget(self.ai_sep_btn)

        enh_lbl = QLabel("Kontrollü Ses İyileştirme (Kayıpsız Yeni WAV):")
        enh_lbl.setStyleSheet("color: #9CA6B8; font-size: 11px;")
        aa_lay.addWidget(enh_lbl)

        self.enhance_combo = QComboBox()
        for pid, pname in StemEnhancer.PRESETS:
            self.enhance_combo.addItem(pname, pid)
        aa_lay.addWidget(self.enhance_combo)

        self.apply_enhance_btn = QPushButton(" İyileştirmeyi Klibe Uygula")
        self.apply_enhance_btn.setProperty("compact", "true")
        self.apply_enhance_btn.setIcon(icons.icon("sliders", size=13))
        self.apply_enhance_btn.clicked.connect(self._apply_clip_enhancement)
        aa_lay.addWidget(self.apply_enhance_btn)

        ta_layout.addWidget(ai_audio_box)
        ta_layout.addStretch(1)
        self.tabs.addTab(self.tab_audio, "Ses")

        # ================================================================== 3. HIZ SEKMESİ
        self.tab_speed = QWidget()
        ts_layout = QVBoxLayout(self.tab_speed)
        ts_layout.setContentsMargins(8, 8, 8, 8)
        ts_layout.setSpacing(8)

        spd_box = QGroupBox("Oynatma Hızı (Speed)")
        sp_lay = QVBoxLayout(spd_box)
        sp_lay.setSpacing(6)

        spd_row = QHBoxLayout()
        spd_row.addWidget(QLabel("Hız:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(25, 400)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self._on_speed_slider_changed)
        spd_row.addWidget(self.speed_slider, 1)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.25, 4.0)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix("x")
        self.speed_spin.setFixedWidth(68)
        self.speed_spin.valueChanged.connect(self._on_speed_spin_changed)
        spd_row.addWidget(self.speed_spin)
        sp_lay.addLayout(spd_row)

        # Önayar Hız Düğmeleri
        presets_row = QHBoxLayout()
        presets_row.setSpacing(4)
        for spd_val in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            p_btn = QPushButton(f"{spd_val}x")
            p_btn.setProperty("compact", "true")
            p_btn.clicked.connect(lambda _, s=spd_val: self.speed_spin.setValue(s))
            presets_row.addWidget(p_btn)
        sp_lay.addLayout(presets_row)

        self.speed_info_lbl = QLabel("Süre: 1.0x (Değişiklik yok)")
        self.speed_info_lbl.setStyleSheet("color: #38BDF8; font-size: 11px;")
        sp_lay.addWidget(self.speed_info_lbl)

        ts_layout.addWidget(spd_box)
        ts_layout.addStretch(1)
        self.tabs.addTab(self.tab_speed, "Hız")

        # ================================================================== 4. ALTYAZI / METİN SEKMESİ
        self.tab_subtitle = QWidget()
        tsub_layout = QVBoxLayout(self.tab_subtitle)
        tsub_layout.setContentsMargins(8, 8, 8, 8)
        tsub_layout.setSpacing(8)

        sub_box = QGroupBox("Altyazı ve Diyalog Metni")
        sub_lay = QVBoxLayout(sub_box)
        sub_lay.setSpacing(6)

        self.sub_text_edit = QTextEdit()
        self.sub_text_edit.setPlaceholderText("Altyazı veya diyalog metni...")
        self.sub_text_edit.setFixedHeight(75)
        self.sub_text_edit.textChanged.connect(self._on_sub_text_changed)
        sub_lay.addWidget(self.sub_text_edit)

        sz_row = QHBoxLayout()
        sz_row.addWidget(QLabel("Yazı Tipi Boyutu:"))
        self.sub_size_spin = QSpinBox()
        self.sub_size_spin.setRange(12, 96)
        self.sub_size_spin.setValue(24)
        self.sub_size_spin.setSuffix(" pt")
        self.sub_size_spin.valueChanged.connect(self._on_sub_prop_changed)
        sz_row.addWidget(self.sub_size_spin)
        sub_lay.addLayout(sz_row)

        tsub_layout.addWidget(sub_box)

        prompter_box = QGroupBox("Seslendirme Prompter")
        p_layout = QVBoxLayout(prompter_box)
        self.prompter_label = QLabel("Mikrofon kaydına başladığınızda seçili replik burada büyük puntolarla akar.")
        self.prompter_label.setWordWrap(True)
        self.prompter_label.setStyleSheet(
            "color: #F2F4F8; font-size: 12px; font-weight: 600; background: #13151A; padding: 9px; border-radius: 5px;"
        )
        p_layout.addWidget(self.prompter_label)
        tsub_layout.addWidget(prompter_box)

        tsub_layout.addStretch(1)
        self.tabs.addTab(self.tab_subtitle, "Metin")

        c_layout.addWidget(self.tabs)
        root.addWidget(self.clip_container)
        root.addStretch(1)

        scroll.setWidget(container)
        outer.addWidget(scroll)

        self.set_clip(None)

    # ------------------------------------------------------------------ Proje & Klip Senkronizasyonu
    def update_project_summary(self, project: TimelineProject, extra_note: str = "") -> None:
        self.current_project = project
        clip_cnt = sum(len(t.clips) for t in project.tracks)
        self.pd_name_lbl.setText(f"Proje: {project.name}")
        lines = [
            f"Çözünürlük: {project.resolution[0]}×{project.resolution[1]}",
            f"Kare Hızı: {project.fps:.2f} fps",
            f"Örnekleme Hızı: {project.sample_rate} Hz",
            f"Kanal Sayısı: {len(project.tracks)} Katman · {clip_cnt} Klip",
            f"Toplam Süre: {project.duration:.2f} sn",
        ]
        if extra_note:
            lines.append(extra_note)
        self.pd_specs_lbl.setText("\n".join(lines))

    def set_clip(self, clip: Optional[TimelineClip]):
        """Bir klip seçildiğinde veya seçim kaldırıldığında arayüzü CapCut tarzı günceller."""
        self._updating_ui = True
        self.current_clip = clip
        try:
            if not clip:
                self.header_title.setText("Proje Detayları")
                self.project_details_card.setVisible(True)
                self.clip_container.setVisible(False)
                return

            self.header_title.setText("Klip Özellikleri")
            self.project_details_card.setVisible(False)
            self.clip_container.setVisible(True)

            self.name_edit.setText(clip.name)
            self.dur_label.setText(
                f"Süre: {clip.duration:.2f}s | Giriş: {clip.timeline_in:.2f}s | Çıkış: {clip.timeline_out:.2f}s"
            )

            if clip.media_path and Path(clip.media_path).exists():
                p = Path(clip.media_path)
                sz_kb = p.stat().st_size / 1024.0
                self.source_meta_label.setText(f"Kaynak: {p.name} ({sz_kb:.0f} KB)")
            elif clip.media_path:
                self.source_meta_label.setText(f"Eksik Kaynak: {Path(clip.media_path).name}")
            else:
                self.source_meta_label.setText("Katman: Metin / Altyazı")

            is_video = clip.clip_type in (ClipType.VIDEO, ClipType.IMAGE)
            is_audio = clip.clip_type == ClipType.AUDIO
            is_sub = clip.clip_type == ClipType.SUBTITLE

            # Sekmeleri türe göre etkinleştir/devre dışı bırak
            self.tabs.setTabVisible(self.tabs.indexOf(self.tab_video), is_video)
            self.tabs.setTabVisible(self.tabs.indexOf(self.tab_audio), is_video or is_audio)
            self.tabs.setTabVisible(self.tabs.indexOf(self.tab_speed), True)
            self.tabs.setTabVisible(self.tabs.indexOf(self.tab_subtitle), is_sub)
            self.ai_sep_btn.setVisible(clip.clip_type == ClipType.VIDEO)

            if is_video:
                self.tabs.setCurrentWidget(self.tab_video)
            elif is_audio:
                self.tabs.setCurrentWidget(self.tab_audio)
            elif is_sub:
                self.tabs.setCurrentWidget(self.tab_subtitle)

            # Değerleri doldur
            self.sync_clip_values(clip)

            if is_sub:
                self.sub_text_edit.setPlainText(clip.text_content)
                self.sub_size_spin.setValue(clip.font_size)
                self.prompter_label.setText(clip.text_content or "Replik metni boş.")
        finally:
            self._updating_ui = False

    def sync_clip_values(self, clip: TimelineClip) -> None:
        """Timeline üzerindeki etkileşimlerde (örn. ses seviyesi çizgisinin sürüklenmesi)
        inspector denetimlerini anında ve döngüsüz (cycle-free) günceller."""
        old_updating = self._updating_ui
        self._updating_ui = True
        try:
            # Video dönüşüm & renk
            scale_pct = clip.scale * 100.0
            self.scale_spin.setValue(scale_pct)
            self.scale_slider.setValue(int(round(max(10, min(300, scale_pct)))))
            self.pos_x_spin.setValue(int(clip.pos_x))
            self.pos_y_spin.setValue(int(clip.pos_y))
            self.rot_spin.setValue(float(clip.rotation))
            op_pct = clip.opacity * 100.0
            self.opacity_spin.setValue(op_pct)
            self.opacity_slider.setValue(int(round(op_pct)))

            self.bright_spin.setValue(clip.brightness)
            self.contrast_spin.setValue(clip.contrast)
            self.sat_spin.setValue(clip.saturation)

            # Ses (dB ve slider)
            vol_linear = max(0.0, float(clip.volume))
            if vol_linear <= 0.001:
                db_val = -60.0
            else:
                db_val = max(-60.0, min(12.0, 20.0 * math.log10(vol_linear)))
            self.vol_db_spin.setValue(db_val)
            self.vol_slider.setValue(int(round(db_val)))
            self._update_vol_label(vol_linear)

            self.fade_in_spin.setValue(clip.fade_in_sec)
            self.fade_out_spin.setValue(clip.fade_out_sec)

            pan_pct = int(round(clip.pan * 100.0))
            self.pan_slider.setValue(pan_pct)
            self._update_pan_label(clip.pan)
            self.mute_check.setChecked(clip.is_muted)

            # Hız
            self.speed_spin.setValue(clip.speed)
            self.speed_slider.setValue(int(round(clip.speed * 100.0)))
            self._update_speed_info(clip.speed)

            # Süre bilgisi
            self.dur_label.setText(
                f"Süre: {clip.duration:.2f}s | Giriş: {clip.timeline_in:.2f}s | Çıkış: {clip.timeline_out:.2f}s"
            )
        finally:
            self._updating_ui = old_updating

    # ------------------------------------------------------------------ Olay Yöneticileri
    def _on_name_changed(self, txt: str):
        if self._updating_ui or not self.current_clip:
            return
        self.current_clip.name = txt
        self.clipChanged.emit(self.current_clip)

    def _on_scale_spin_changed(self, val: float):
        if self._updating_ui or not self.current_clip:
            return
        self._updating_ui = True
        self.scale_slider.setValue(int(round(max(10, min(300, val)))))
        self._updating_ui = False
        self.current_clip.scale = val / 100.0
        self.clipChanged.emit(self.current_clip)

    def _on_scale_slider_changed(self, val: int):
        if self._updating_ui or not self.current_clip:
            return
        self._updating_ui = True
        self.scale_spin.setValue(float(val))
        self._updating_ui = False
        self.current_clip.scale = val / 100.0
        self.clipChanged.emit(self.current_clip)

    def _on_transform_changed(self):
        if self._updating_ui or not self.current_clip:
            return
        self.current_clip.pos_x = float(self.pos_x_spin.value())
        self.current_clip.pos_y = float(self.pos_y_spin.value())
        self.current_clip.rotation = float(self.rot_spin.value())
        self.clipChanged.emit(self.current_clip)

    def _reset_position(self):
        self.pos_x_spin.setValue(0)
        self.pos_y_spin.setValue(0)

    def _on_opacity_spin_changed(self, val: float):
        if self._updating_ui or not self.current_clip:
            return
        self._updating_ui = True
        self.opacity_slider.setValue(int(round(val)))
        self._updating_ui = False
        self.current_clip.opacity = val / 100.0
        self.clipChanged.emit(self.current_clip)

    def _on_opacity_slider_changed(self, val: int):
        if self._updating_ui or not self.current_clip:
            return
        self._updating_ui = True
        self.opacity_spin.setValue(float(val))
        self._updating_ui = False
        self.current_clip.opacity = val / 100.0
        self.clipChanged.emit(self.current_clip)

    def _on_color_prop_changed(self):
        if self._updating_ui or not self.current_clip:
            return
        self.current_clip.brightness = self.bright_spin.value()
        self.current_clip.contrast = self.contrast_spin.value()
        self.current_clip.saturation = self.sat_spin.value()
        self.clipChanged.emit(self.current_clip)

    def _reset_colors(self):
        self.bright_spin.setValue(0.0)
        self.contrast_spin.setValue(1.0)
        self.sat_spin.setValue(1.0)

    def _on_vol_db_spin_changed(self, db_val: float):
        if self._updating_ui or not self.current_clip:
            return
        self._updating_ui = True
        self.vol_slider.setValue(int(round(db_val)))
        self._updating_ui = False
        self._apply_volume_from_db(db_val)

    def _on_vol_slider_changed(self, val: int):
        if self._updating_ui or not self.current_clip:
            return
        self._updating_ui = True
        self.vol_db_spin.setValue(float(val))
        self._updating_ui = False
        self._apply_volume_from_db(float(val))

    def _apply_volume_from_db(self, db_val: float):
        if db_val <= -59.5:
            vol_linear = 0.0
        else:
            vol_linear = 10.0 ** (db_val / 20.0)
        self.current_clip.volume = vol_linear
        self._update_vol_label(vol_linear)
        self.clipChanged.emit(self.current_clip)

    def _update_vol_label(self, vol: float) -> None:
        if vol <= 0.001:
            self.vol_db_label.setText("-inf dB")
        else:
            db = 20.0 * math.log10(vol)
            self.vol_db_label.setText(f"{db:+.1f} dB")

    def _on_pan_slider_changed(self, val: int):
        if self._updating_ui or not self.current_clip:
            return
        pan_val = val / 100.0
        self.current_clip.pan = pan_val
        self._update_pan_label(pan_val)
        self.clipChanged.emit(self.current_clip)

    def _update_pan_label(self, pan: float) -> None:
        if abs(pan) < 0.03:
            self.pan_val_label.setText("C")
        elif pan < 0:
            self.pan_val_label.setText(f"L{int(abs(pan) * 100)}")
        else:
            self.pan_val_label.setText(f"R{int(pan * 100)}")

    def _on_audio_prop_changed(self):
        if self._updating_ui or not self.current_clip:
            return
        self.current_clip.fade_in_sec = self.fade_in_spin.value()
        self.current_clip.fade_out_sec = self.fade_out_spin.value()
        self.current_clip.is_muted = self.mute_check.isChecked()
        self.clipChanged.emit(self.current_clip)

    def _on_speed_spin_changed(self, val: float):
        if self._updating_ui or not self.current_clip:
            return
        self._updating_ui = True
        self.speed_slider.setValue(int(round(val * 100.0)))
        self._updating_ui = False
        self.current_clip.speed = val
        self._update_speed_info(val)
        self.clipChanged.emit(self.current_clip)

    def _on_speed_slider_changed(self, val: int):
        if self._updating_ui or not self.current_clip:
            return
        spd_val = val / 100.0
        self._updating_ui = True
        self.speed_spin.setValue(spd_val)
        self._updating_ui = False
        self.current_clip.speed = spd_val
        self._update_speed_info(spd_val)
        self.clipChanged.emit(self.current_clip)

    def _update_speed_info(self, spd: float):
        if not self.current_clip:
            return
        if abs(spd - 1.0) < 0.01:
            self.speed_info_lbl.setText("Hız: 1.0x (Orijinal Süre)")
        elif spd > 1.0:
            self.speed_info_lbl.setText(f"Hız: {spd:.2f}x ({int((spd - 1.0) * 100)}% Daha Hızlı)")
        else:
            self.speed_info_lbl.setText(f"Hız: {spd:.2f}x ({int((1.0 - spd) * 100)}% Daha Yavaş / Ağır Çekim)")

    def _on_sub_text_changed(self):
        if self._updating_ui or not self.current_clip:
            return
        txt = self.sub_text_edit.toPlainText()
        self.current_clip.text_content = txt
        self.prompter_label.setText(txt or "Replik metni boş.")
        self.clipChanged.emit(self.current_clip)

    def _on_sub_prop_changed(self):
        if self._updating_ui or not self.current_clip:
            return
        self.current_clip.font_size = self.sub_size_spin.value()
        self.clipChanged.emit(self.current_clip)

    def _on_ai_sep_clicked(self):
        if self.current_clip:
            self.separateAudioRequested.emit(self.current_clip)

    def _apply_clip_enhancement(self):
        """Seçili ses klibine orijinal dosyayı bozmadan kontrollü iyileştirme uygular."""
        if not self.current_clip or not self.current_clip.media_path:
            return
        preset_id = self.enhance_combo.currentData()
        if not preset_id or preset_id == "none":
            return
        src = Path(self.current_clip.media_path)
        if not src.exists():
            return
        try:
            audio, sr = MediaExtractor.read_audio_any(src)
            enhanced = StemEnhancer.apply_enhancement(audio, sr, preset_id=preset_id)
            out_path = src.parent / f"{src.stem}_{preset_id}.wav"
            sf.write(str(out_path), enhanced, sr, subtype="PCM_24")
            self.current_clip.media_path = str(out_path)
            if not self.current_clip.name.endswith(f" [{preset_id}]"):
                self.current_clip.name = f"{self.current_clip.name} [{preset_id}]"
            self.set_clip(self.current_clip)
            self.clipChanged.emit(self.current_clip)
        except Exception as exc:
            logger.error(f"Klip ses iyileştirme hatası: {exc}")
