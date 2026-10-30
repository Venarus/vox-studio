"""
Kayıt Stüdyosu görünümü — Profesyonel Mikrofon Algılama, Kontrollü Temiz Kayıt Zinciri
ve Mikrofon Kalibrasyon Sihirbazı.

Gerçek cihaz durumu, canlı seviye, giriş kontrolü, kayıt waveform'u ve
ölçülen analiz değerlerini gösterir. Hiçbir değer örnek/sahte değildir;
veri yoksa "—" gösterilir. Ham mikrofon kaydı ASLA ezilmez; temiz kayıt zinciri
etkinleştirildiğinde ham kayıt (.wav) korunur ve yanına (_temiz.wav) üretilir.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from audio.level import InputCheck, LevelSnapshot
from audio.player import PlaybackInfo
from audio.recorder import RecordingResult
from core import constants as C
from core.format import format_duration, format_file_size
from core.logger import get_logger

logger = get_logger("ui.views.recording_studio")
from dsp.analysis import AudioAnalysis, analyze_file
from dsp.chain import DSPProcessor, ProcessingChainConfig
from dsp.denoise import DenoiseSettings
from dsp.dynamics import CompressorSettings, LimiterSettings
from dsp.filters import EQBand
from ui import icons
from ui.context import AppContext
from ui.widgets.level_meter import LevelMeter
from ui.widgets.waveform import WaveformWidget


def _db_with_unit(value: float, unit: str) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.1f} {unit}"


class MicCalibrationWizardDialog(QDialog):
    """
    Bölüm 3.6 — Mikrofon Kalibrasyon Sihirbazı (8 Adımlı Gerçek Ölçümlü Sihirbaz):
    1. Mikrofonu seç
    2. Canlı sinyali kontrol et
    3. 3 saniye konuşma testi kaydet
    4. Kırpılma (clipping) var mı kontrol et
    5. Arka plan gürültü tabanını ve SNR değerini ölç
    6. Uygun giriş kazancını (dB) ve konuşma mesafesini öner
    7. Test kaydını kullanıcıya dinlet
    8. Kullanıcı onaylarsa ayarları Kayıt Stüdyosuna uygula
    """

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Mikrofon Kalibrasyon Sihirbazı")
        self.setMinimumSize(580, 500)
        self.setModal(True)

        self._test_wav_path: Optional[Path] = None
        self._recommended_gain_db: float = 0.0
        self._recommended_clean_chain: bool = True
        self._recommended_intensity: str = "balanced"
        self._countdown_sec: int = 0

        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._on_test_tick)

        self._build_ui()
        self._populate_devices()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        title = QLabel("Mikrofon Kalibrasyon ve Ortam Analizi Sihirbazı")
        title.setObjectName("SectionTitle")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        root.addWidget(title)

        sub = QLabel(
            "1) Mikrofonunuzu seçin  →  2) 3 saniyelik gerçek test kaydı alın  →  "
            "3) Kırpılma, gürültü tabanı ve SNR raporunu inceleyip test kaydını dinleyin  →  "
            "4) Önerilen kazanç ve temiz kayıt zinciri ayarlarını onaylayın."
        )
        sub.setObjectName("MutedSmall")
        sub.setWordWrap(True)
        root.addWidget(sub)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Mikrofon Cihazı:"))
        self.dev_combo = QComboBox()
        dev_row.addWidget(self.dev_combo, 1)
        root.addLayout(dev_row)

        # Canlı seviye çubuğu
        meter_row = QHBoxLayout()
        meter_row.addWidget(QLabel("Anlık Giriş Seviyesi:"))
        self.live_meter = LevelMeter(self, show_scale=False, show_clip=True)
        self.live_meter.setFixedHeight(16)
        meter_row.addWidget(self.live_meter, 1)
        root.addLayout(meter_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 3000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        root.addWidget(self.progress_bar)

        ctrl_row = QHBoxLayout()
        self.start_test_btn = QPushButton("3 Saniyelik Konuşma Testini Başlat")
        self.start_test_btn.setProperty("variant", "primary")
        self.start_test_btn.setIcon(icons.icon("mic", size=14, color="#FFFFFF"))
        self.start_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_test_btn.clicked.connect(self.start_calibration_test)
        ctrl_row.addWidget(self.start_test_btn)

        self.listen_test_btn = QPushButton("Test Kaydını Dinle")
        self.listen_test_btn.setProperty("variant", "ghost")
        self.listen_test_btn.setIcon(icons.icon("play", size=14))
        self.listen_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.listen_test_btn.setEnabled(False)
        self.listen_test_btn.clicked.connect(self._listen_test_recording)
        ctrl_row.addWidget(self.listen_test_btn)
        root.addLayout(ctrl_row)

        self.report_box = QPlainTextEdit()
        self.report_box.setReadOnly(True)
        self.report_box.setPlaceholderText(
            "Test başlatıldığında mikrofonunuzdan 3 saniyelik gerçek ses kaydedilir ve:\n"
            "• Tepe Seviyesi (Peak dBFS) ve Kırpılma (Clipping)\n"
            "• Ortam Gürültü Tabanı (Noise Floor dBFS) ve SNR (dB)\n"
            "• Önerilen Ön Kazanç (dB) ve Mikrofon Mesafesi\n"
            "burada ayrıntılı olarak raporlanır."
        )
        root.addWidget(self.report_box, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("İptal")
        cancel_btn.setProperty("variant", "ghost")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self.apply_btn = QPushButton("Önerilen Ayarları Uygula")
        self.apply_btn.setProperty("variant", "primary")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.apply_btn)
        root.addLayout(btn_row)

    def _populate_devices(self) -> None:
        self.dev_combo.clear()
        self.ctx.refresh_devices()
        selected = self.ctx.input_device()
        for dev in self.ctx.devices.input_devices():
            marker = " (Varsayılan)" if dev.is_default_input else ""
            self.dev_combo.addItem(f"{dev.name} · {dev.hostapi_name}{marker}", dev.index)
        if selected is not None:
            idx = self.dev_combo.findData(selected.index)
            if idx >= 0:
                self.dev_combo.setCurrentIndex(idx)

    def start_calibration_test(self) -> None:
        dev_idx = self.dev_combo.currentData()
        dev = self.ctx.devices.get(int(dev_idx)) if dev_idx is not None else self.ctx.input_device()
        if dev is not None:
            self.ctx.remember_device(dev, "input")

        self.ctx.monitor.stop()
        self.ctx.player.stop()

        tmp_dir = Path(tempfile.gettempdir()) / "vox_studio_calib"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self._test_wav_path = tmp_dir / "mic_calibration_test.wav"

        sr = self.ctx.config.get_int("sample_rate", 48000)
        ch = self.ctx.config.get_int("channels", 1)
        try:
            self.ctx.recorder.start(
                output_path=self._test_wav_path,
                samplerate=sr,
                channels=ch,
                subtype="PCM_24",
                device_index=dev.index if dev else None,
            )
            self._elapsed_ms = 0
            self.progress_bar.setValue(0)
            self.start_test_btn.setEnabled(False)
            self.report_box.setPlainText(
                "Kayıt alınıyor (3 saniye)... Lütfen normal seslendirme tonunuzla mikrofona konuşun."
            )
            self._timer.start()
        except Exception as exc:
            self.report_box.setPlainText(f"Test kaydı başlatılamadı: {exc}")

    def _on_test_tick(self) -> None:
        self._elapsed_ms += 250
        self.progress_bar.setValue(min(3000, self._elapsed_ms))
        if self._elapsed_ms >= 3000:
            self._timer.stop()
            res = self.ctx.recorder.stop()
            self.start_test_btn.setEnabled(True)
            if res and res.path and Path(res.path).exists():
                self._test_wav_path = Path(res.path)
                self.evaluate_test_file(self._test_wav_path)
            else:
                self.report_box.setPlainText("Test kaydı tamamlanamadı.")

    def evaluate_test_file(self, wav_path: Path) -> AudioAnalysis:
        """Gerçek kaydedilmiş WAV dosyasını analiz eder ve kalibrasyon önerilerini üretir."""
        analysis = analyze_file(wav_path)
        self.live_meter.set_level(
            analysis.peak_dbfs,
            analysis.peak_dbfs,
            analysis.clipped_samples > 0,
        )

        # Uygun kazanç ve mesafe önerisi
        target_peak = -6.0
        if not math.isfinite(analysis.peak_dbfs) or analysis.peak_dbfs < -55.0:
            self._recommended_gain_db = 12.0
            dist_advice = (
                "Sinyal çok zayıf veya sessiz! Mikrofonun sessize (mute) alınmadığından emin olun "
                "ve mikrofona 10–15 cm mesafeden konuşun."
            )
        elif analysis.clipped_samples > 0 or analysis.peak_dbfs > -1.0:
            self._recommended_gain_db = -6.0
            dist_advice = (
                "Kırpılma (Clipping) veya aşırı yüksek seviye tespit edildi! Mikrofondan 20–25 cm "
                "uzaklaşın veya ses kartı/Windows mikrofon kazancını kısın."
            )
        else:
            diff = target_peak - analysis.peak_dbfs
            self._recommended_gain_db = round(float(np.clip(diff, -12.0, 12.0)), 1)
            dist_advice = "Seviye sağlıklı aralıkta. Mikrofona 15–20 cm mesafeyi koruyun."

        if analysis.noise_floor_dbfs > -42.0:
            self._recommended_clean_chain = True
            self._recommended_intensity = "strong"
            noise_advice = "Ortam gürültüsü yüksek (> -42 dBFS). 'Güçlü' Temiz Kayıt Zinciri önerildi."
        elif analysis.noise_floor_dbfs > -55.0:
            self._recommended_clean_chain = True
            self._recommended_intensity = "balanced"
            noise_advice = "Orta düzey ortam gürültüsü algılandı. 'Dengeli' Temiz Kayıt Zinciri önerildi."
        else:
            self._recommended_clean_chain = True
            self._recommended_intensity = "light"
            noise_advice = "Ortam oldukça sessiz. 'Hafif (Doğal)' Temiz Kayıt Zinciri önerildi."

        lines = [
            "=== MİKROFON KALİBRASYON ÖLÇÜM RAPORU ===",
            f"• Test Süresi: {analysis.duration:.2f} sn ({analysis.samplerate} Hz, {analysis.channels} kanal)",
            f"• Tepe Seviyesi (Peak): {_db_with_unit(analysis.peak_dbfs, 'dBFS')}",
            f"• Ortalama Seviye (RMS): {_db_with_unit(analysis.rms_dbfs, 'dBFS')}",
            f"• Algılanan Ses Yüksekliği: {_db_with_unit(analysis.loudness_lufs, 'LUFS')}",
            f"• Ortam Gürültü Tabanı: {_db_with_unit(analysis.noise_floor_dbfs, 'dBFS')}",
            f"• Sinyal/Gürültü Oranı (SNR): {_db_with_unit(analysis.snr_estimate_db, 'dB')}",
            f"• Kırpılan Örnek (Clipping): {analysis.clipped_samples} örnek",
            "",
            "=== ÖNERİLEN AYARLAR ===",
            f"• Önerilen Ön Kazanç (Gain): {self._recommended_gain_db:+.1f} dB",
            f"• Mesafe ve Konum Önerisi: {dist_advice}",
            f"• Gürültü & Zincir Önerisi: {noise_advice}",
        ]
        self.report_box.setPlainText("\n".join(lines))
        self.listen_test_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        return analysis

    def _listen_test_recording(self) -> None:
        if self._test_wav_path and self._test_wav_path.exists():
            out_dev = self.ctx.output_device()
            self.ctx.player.load(self._test_wav_path)
            self.ctx.player.play(device_index=out_dev.index if out_dev else None)

    @property
    def recommended_gain_db(self) -> float:
        return self._recommended_gain_db

    @property
    def recommended_clean_chain(self) -> bool:
        return self._recommended_clean_chain

    @property
    def recommended_intensity(self) -> str:
        return self._recommended_intensity


class RecordingStudioView(QWidget):
    openFileRequested = Signal(str)
    libraryRequested = Signal()
    analyzeRequested = Signal(str)
    seekRequested = Signal(float)
    settingsChanged = Signal()
    monitorStateChanged = Signal(bool)
    sendToTimelineRequested = Signal(str)

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx
        self._input_check = InputCheck()
        self._current_path: Optional[str] = None
        self._raw_take_path: Optional[str] = None
        self._clean_take_path: Optional[str] = None
        self._showing_clean_take: bool = False
        self._file_text = "Kayıt bekleniyor"
        self._build_ui()
        self.ctx.bridge.monitorError.connect(self._on_monitor_error)
        self.refresh_devices()
        self.refresh_recent()
        self._set_analysis_placeholder()
        self.show_level(None)

    # ================================================================ arayüz
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(14)

        # ---------------------------------------------------------- başlık
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Kayıt Stüdyosu")
        title.setObjectName("ViewTitle")
        subtitle = QLabel(
            "Mikrofonunuzu seçin, kalibrasyon sihirbazıyla girişi doğrulayın ve kontrollü temiz kayıt alın."
        )
        subtitle.setObjectName("ViewSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)

        self.calib_wizard_btn = QPushButton("Mikrofon Kalibrasyon Sihirbazı")
        self.calib_wizard_btn.setProperty("variant", "secondary")
        self.calib_wizard_btn.setIcon(icons.icon("sparkle", size=15, color=C.COLOR_ACCENT_PRIMARY))
        self.calib_wizard_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.calib_wizard_btn.setToolTip(
            "3 saniyelik gerçek test kaydıyla mikrofon kazancı, gürültü tabanı ve kırpılma kontrolü yapar"
        )
        self.calib_wizard_btn.clicked.connect(self.open_calibration_wizard)
        header.addWidget(self.calib_wizard_btn)

        self.library_btn = QPushButton("Kütüphane")
        self.library_btn.setProperty("variant", "ghost")
        self.library_btn.setIcon(icons.icon("library"))
        self.library_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.library_btn.clicked.connect(self.libraryRequested.emit)
        header.addWidget(self.library_btn)
        root.addLayout(header)

        # ------------------------------------------------------------ gövde
        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(14)
        body.addLayout(left, 1)

        left.addWidget(self._build_config_card())
        left.addWidget(self._build_waveform_card(), 1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setFixedWidth(348)

        right_inner = QWidget()
        right = QVBoxLayout(right_inner)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(14)
        right.addWidget(self._build_analysis_card(), 1)
        right.addWidget(self._build_recent_card())
        right_scroll.setWidget(right_inner)
        body.addWidget(right_scroll)

    # ------------------------------------------------------------ cihaz kartı
    def _build_config_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        head = QHBoxLayout()
        section = QLabel("GİRİŞ, KALİBRASYON VE TEMİZ KAYIT ZİNCİRİ")
        section.setObjectName("SectionTitle")
        head.addWidget(section)
        head.addStretch(1)
        self.refresh_btn = QPushButton("Cihazları Yenile")
        self.refresh_btn.setProperty("variant", "link")
        self.refresh_btn.setIcon(icons.icon("refresh", size=14))
        self.refresh_btn.setToolTip("Ses cihazlarını yeniden tara (cihaz takıldı/çıkarıldı)")
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.refresh_devices)
        head.addWidget(self.refresh_btn)
        layout.addLayout(head)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)

        self.input_combo = QComboBox()
        self.input_combo.setMinimumWidth(170)
        self.input_combo.setToolTip("Kayıt yapılacak mikrofon / ses arayüzü girişi")
        self.output_combo = QComboBox()
        self.output_combo.setMinimumWidth(170)
        self.output_combo.setToolTip("Oynatma için kullanılacak çıkış cihazı")

        self.rate_combo = QComboBox()
        self.rate_combo.setToolTip(
            "Kayıt örnekleme hızı. Cihaz bu hızı desteklemiyorsa kayıt başlatılmaz; "
            "desteklenen hızlar hata mesajında listelenir."
        )
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("Mono (1 kanal)", 1)
        self.channel_combo.addItem("Stereo (2 kanal)", 2)
        self.channel_combo.setToolTip("Seslendirme için mono önerilir")

        self.format_combo = QComboBox()
        for label, subtype in C.RECORDING_SUBTYPES.items():
            self.format_combo.addItem(label, subtype)
        self.format_combo.setToolTip("WAV bit derinliği. 24-bit, seslendirme için güvenli varsayılandır.")

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("İsteğe bağlı — örn. Sahne 4 / replik 12")
        self.label_edit.setMaxLength(60)
        self.label_edit.setToolTip("Dosya adına eklenir: tarih_saat_etiket.wav")

        labels = [
            ("Giriş cihazı", self.input_combo, 0, 0),
            ("Çıkış cihazı", self.output_combo, 0, 2),
            ("Örnekleme hızı", self.rate_combo, 1, 0),
            ("Kayıt kanalı", self.channel_combo, 1, 2),
            ("Kayıt biçimi", self.format_combo, 2, 0),
            ("Kayıt etiketi", self.label_edit, 2, 2),
        ]
        for text, widget, row, col in labels:
            cell = QVBoxLayout()
            cell.setSpacing(3)
            caption = QLabel(text)
            caption.setObjectName("MutedSmall")
            cell.addWidget(caption)
            cell.addWidget(widget)
            grid.addLayout(cell, row, col)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addLayout(grid)

        # Kontrollü Temiz Kayıt Zinciri (Bölüm 3.4)
        chain_row = QHBoxLayout()
        chain_row.setSpacing(10)
        self.clean_chain_check = QCheckBox("Temiz Kayıt Zinciri (Ham kaydı korur + _temiz.wav üretir)")
        self.clean_chain_check.setChecked(False)
        self.clean_chain_check.setToolTip(
            "Kayıt bittiğinde ham mikrofon kaydını ASLA ezmez; DC offset + 80Hz HPF + "
            "hafif gürültü azaltma + vokal netlik + yumuşak kompresör + -1 dBFS limiter ile "
            "ayrı bir '_temiz.wav' kopyası hazırlar."
        )
        chain_row.addWidget(self.clean_chain_check)

        chain_row.addWidget(QLabel("Ön Kazanç:"))
        self.input_gain_spin = QDoubleSpinBox()
        self.input_gain_spin.setRange(-18.0, 18.0)
        self.input_gain_spin.setSingleStep(1.0)
        self.input_gain_spin.setValue(0.0)
        self.input_gain_spin.setSuffix(" dB")
        chain_row.addWidget(self.input_gain_spin)

        chain_row.addWidget(QLabel("Yoğunluk:"))
        self.clean_intensity_combo = QComboBox()
        self.clean_intensity_combo.addItem("Hafif (Doğal)", "light")
        self.clean_intensity_combo.addItem("Dengeli (Önerilen)", "balanced")
        self.clean_intensity_combo.addItem("Güçlü (Gürültülü Ortam)", "strong")
        self.clean_intensity_combo.setCurrentIndex(1)
        chain_row.addWidget(self.clean_intensity_combo)
        chain_row.addStretch(1)
        layout.addLayout(chain_row)

        # monitor + seviye
        monitor_row = QHBoxLayout()
        monitor_row.setSpacing(10)
        self.monitor_check = QCheckBox("Girişi dinle (kayıt yapılmaz)")
        self.monitor_check.setToolTip(
            "Mikrofon sinyalini kaydetmeden ölçer. Kayıt başlarken otomatik kapanır."
        )
        self.monitor_check.toggled.connect(self._on_monitor_toggled)
        monitor_row.addWidget(self.monitor_check)

        self.level_meter = LevelMeter(self, show_scale=False, show_clip=True)
        self.level_meter.setFixedWidth(220)
        self.level_meter.setFixedHeight(14)
        monitor_row.addWidget(self.level_meter)
        monitor_row.addStretch(1)
        layout.addLayout(monitor_row)

        self.check_label = QLabel("")
        self.check_label.setObjectName("MutedSmall")
        self.check_label.setWordWrap(True)
        layout.addWidget(self.check_label)

        self.record_hint = QLabel(
            "Kayıt başlatmak için alt taşıma çubuğundaki kırmızı kayıt düğmesini kullanın (Ctrl+R)."
        )
        self.record_hint.setObjectName("MutedSmall")
        self.record_hint.setWordWrap(True)
        layout.addWidget(self.record_hint)

        # değişiklik sinyalleri
        self.input_combo.currentIndexChanged.connect(self._on_input_changed)
        self.output_combo.currentIndexChanged.connect(self._on_output_changed)
        self.rate_combo.currentIndexChanged.connect(self._on_rate_changed)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        return card

    # --------------------------------------------------------- waveform kartı
    def _build_waveform_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        head = QHBoxLayout()
        self.file_label = QLabel("Kayıt bekleniyor")
        self.file_label.setObjectName("SectionTitle")
        head.addWidget(self.file_label)
        head.addStretch(1)

        self.ab_take_btn = QPushButton("Mod: Ham Kayıt (A)")
        self.ab_take_btn.setProperty("variant", "ghost")
        self.ab_take_btn.setCheckable(True)
        self.ab_take_btn.setEnabled(False)
        self.ab_take_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ab_take_btn.setToolTip("Ham mikrofon kaydı (A) ile Temiz Kayıt Zinciri çıktısı (B) arasında geçiş yapın")
        self.ab_take_btn.toggled.connect(self._on_ab_take_toggled)
        head.addWidget(self.ab_take_btn)

        self.send_take_tl_btn = QPushButton("AI Restorasyon Lab'ında Aç")
        self.send_take_tl_btn.setProperty("variant", "primary")
        self.send_take_tl_btn.setIcon(icons.icon("sparkles", size=14, color="#FFFFFF"))
        self.send_take_tl_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_take_tl_btn.setToolTip("Kaydı AI Konuşma Restorasyonu ve Ses Temizleme Laboratuvarına aktarır")
        self.send_take_tl_btn.clicked.connect(self._on_send_take_to_timeline)
        head.addWidget(self.send_take_tl_btn)

        self.duration_label = QLabel("—")
        self.duration_label.setObjectName("TimeSub")
        head.addWidget(self.duration_label)
        layout.addLayout(head)

        self.waveform = WaveformWidget()
        self.waveform.seekRequested.connect(self.seekRequested.emit)
        layout.addWidget(self.waveform, 1)

        hint = QLabel(
            "Waveform kayıt sırasında canlı oluşur. Konum seçmek için dalga biçimine tıklayın."
        )
        hint.setObjectName("MutedSmall")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return card

    # --------------------------------------------------------- analiz kartı
    def _build_analysis_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        head = QHBoxLayout()
        section = QLabel("SES ANALİZİ")
        section.setObjectName("SectionTitle")
        head.addWidget(section)
        head.addStretch(1)
        self.analyze_btn = QPushButton("Analiz et")
        self.analyze_btn.setProperty("variant", "link")
        self.analyze_btn.setIcon(icons.icon("analyze", size=14))
        self.analyze_btn.setToolTip("Seçili dosyayı gerçek DSP ölçümleriyle analiz eder")
        self.analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)
        head.addWidget(self.analyze_btn)
        layout.addLayout(head)

        score_row = QHBoxLayout()
        self.score_value = QLabel("—")
        self.score_value.setObjectName("ScoreValue")
        self.score_value.setStyleSheet(f"color: {C.COLOR_TEXT_PRIMARY};")
        self.score_unit = QLabel("/ 100")
        self.score_unit.setObjectName("MutedSmall")
        score_row.addWidget(self.score_value)
        score_row.addWidget(self.score_unit, 0, Qt.AlignmentFlag.AlignBottom)
        score_row.addStretch(1)
        self.analysis_state = QLabel("Dosya seçilmedi")
        self.analysis_state.setObjectName("MutedSmall")
        score_row.addWidget(self.analysis_state, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(score_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self._metrics: dict[str, QLabel] = {}
        specs = [
            ("peak", "TEPE"),
            ("rms", "RMS"),
            ("loudness", "LOUDNESS"),
            ("crest", "CREST"),
            ("noise", "GÜRÜLTÜ TABANI"),
            ("snr", "SNR (TAHMİN)"),
            ("silence", "SESSİZLİK"),
            ("clipping", "KIRPILMA"),
        ]
        for i, (key, label) in enumerate(specs):
            tile = QFrame()
            tile.setObjectName("MetricBox")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(9, 6, 9, 6)
            tile_layout.setSpacing(1)
            value = QLabel("—")
            value.setObjectName("MetricValueSmall")
            caption = QLabel(label)
            caption.setObjectName("MetricLabel")
            tile_layout.addWidget(value)
            tile_layout.addWidget(caption)
            self._metrics[key] = value
            grid.addWidget(tile, i // 2, i % 2)
        layout.addLayout(grid)

        self.format_label = QLabel("Biçim: —")
        self.format_label.setObjectName("MutedSmall")
        self.format_label.setWordWrap(True)
        layout.addWidget(self.format_label)

        self.method_label = QLabel(
            "Loudness yöntemi: ITU-R BS.1770-4 (K-ağırlıklı, kapılı entegre)."
        )
        self.method_label.setObjectName("MutedSmall")
        self.method_label.setWordWrap(True)
        layout.addWidget(self.method_label)

        self.warnings_box = QPlainTextEdit()
        self.warnings_box.setReadOnly(True)
        self.warnings_box.setMaximumHeight(70)
        self.warnings_box.setPlaceholderText("Analiz uyarıları burada görünür.")
        self.warnings_box.setObjectName("WarningsBox")
        layout.addWidget(self.warnings_box)

        self.disclaimer = QLabel(
            "Puan mutlak bir ses kalitesi ölçütü değildir; yalnızca kırpılma, seviye, "
            "gürültü tabanı ve DC ofset kriterlerinin özetidir."
        )
        self.disclaimer.setObjectName("MutedSmall")
        self.disclaimer.setWordWrap(True)
        layout.addWidget(self.disclaimer)
        return card

    # ---------------------------------------------------------- son kayıtlar
    def _build_recent_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        card.setMaximumHeight(210)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        head = QHBoxLayout()
        section = QLabel("SON KAYITLAR")
        section.setObjectName("SectionTitle")
        head.addWidget(section)
        head.addStretch(1)
        all_btn = QPushButton("Tümü")
        all_btn.setProperty("variant", "link")
        all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        all_btn.clicked.connect(self.libraryRequested.emit)
        head.addWidget(all_btn)
        layout.addLayout(head)

        self.recent_list = QListWidget()
        self.recent_list.setObjectName("RecentList")
        self.recent_list.itemDoubleClicked.connect(self._on_recent_activated)
        self.recent_list.setToolTip("Çift tıklayın: dosyayı stüdyoda aç")
        layout.addWidget(self.recent_list, 1)
        return card

    # ============================================================ Kalibrasyon & Temiz Kayıt Zinciri
    def open_calibration_wizard(self) -> None:
        self.stop_monitor()
        dlg = MicCalibrationWizardDialog(self.ctx, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.input_gain_spin.setValue(dlg.recommended_gain_db)
            self.clean_chain_check.setChecked(dlg.recommended_clean_chain)
            idx = self.clean_intensity_combo.findData(dlg.recommended_intensity)
            if idx >= 0:
                self.clean_intensity_combo.setCurrentIndex(idx)
            self.refresh_devices()
            self.record_hint.setText(
                f"Kalibrasyon uygulandı: Ön Kazanç {dlg.recommended_gain_db:+.1f} dB, "
                f"Temiz Kayıt Zinciri ({self.clean_intensity_combo.currentText()})."
            )

    def apply_post_recording_clean_chain(self, raw_path: str) -> Optional[str]:
        """
        Bölüm 3.4 — Kontrollü Temiz Kayıt Zinciri:
        Ham mikrofon kaydını (`raw_path`) ASLA değiştirmez veya üzerine yazmaz.
        Aynı klasörde `<stem>_temiz.wav` olarak işlenmiş kopyayı oluşturur:
        1) DC offset temizleme
        2) Ön kazanç (dB)
        3) 80 Hz High-Pass filtre (oda uğultusu / rumble temizliği)
        4) Kontrollü spektral gürültü azaltma
        5) 3.2 kHz vokal netlik (presence) EQ
        6) Yumuşak kompresör ve -1.0 dBFS tavan koruyucu limiter
        """
        src = Path(raw_path)
        if not src.exists():
            return None
        try:
            data, sr = sf.read(str(src), dtype="float32", always_2d=False)
            if data.size == 0:
                return None
            # 1) DC Offset temizleme
            if data.ndim == 1:
                data = data - float(np.mean(data))
            else:
                data = data - np.mean(data, axis=0, keepdims=True)

            intensity = str(self.clean_intensity_combo.currentData() or "balanced")
            reduction_db = 8.0 if intensity == "light" else (12.0 if intensity == "balanced" else 16.0)
            presence_db = 1.2 if intensity == "light" else (2.0 if intensity == "balanced" else 2.5)

            cfg = ProcessingChainConfig(
                eq_enabled=True,
                eq_bands=[
                    EQBand(band_type="highpass", frequency=80.0, q=0.707, enabled=True),
                    EQBand(band_type="peaking", frequency=220.0, gain_db=-1.2, q=1.1, enabled=True),
                    EQBand(band_type="peaking", frequency=3200.0, gain_db=presence_db, q=1.0, enabled=True),
                    EQBand(band_type="highshelf", frequency=9000.0, gain_db=1.0, q=0.707, enabled=True),
                ],
                denoise=DenoiseSettings(enabled=True, reduction_db=reduction_db),
                compressor=CompressorSettings(
                    enabled=True,
                    threshold_db=-20.0,
                    ratio=2.5,
                    attack_ms=12.0,
                    release_ms=110.0,
                    makeup_gain_db=1.5,
                ),
                limiter=LimiterSettings(enabled=True, ceiling_dbfs=-1.0, release_ms=50.0),
                gain_db=float(self.input_gain_spin.value()),
            )
            processed_samples, _stats = DSPProcessor.process(data, float(sr), cfg)
            clean_out = src.with_name(f"{src.stem}_temiz{src.suffix}")
            subtype = self.ctx.config.get_str("recording_subtype", "PCM_24")
            sf.write(str(clean_out), processed_samples, int(sr), subtype=subtype)
            return str(clean_out)
        except Exception as exc:
            logger.error(f"apply_post_recording_clean_chain hatası: {exc}", exc_info=True)
            return None

    def _on_ab_take_toggled(self, checked: bool) -> None:
        if checked and self._clean_take_path and Path(self._clean_take_path).exists():
            self._showing_clean_take = True
            self.ab_take_btn.setText("Mod: Temiz Kayıt (B)")
            self.openFileRequested.emit(self._clean_take_path)
        elif self._raw_take_path and Path(self._raw_take_path).exists():
            self._showing_clean_take = False
            self.ab_take_btn.setText("Mod: Ham Kayıt (A)")
            self.openFileRequested.emit(self._raw_take_path)

    def _on_send_take_to_timeline(self) -> None:
        if self._current_path and Path(self._current_path).exists():
            self.sendToTimelineRequested.emit(self._current_path)

    # ============================================================ cihazlar
    def refresh_devices(self) -> None:
        self.ctx.refresh_devices()
        previous_input = self.ctx.input_device()
        previous_output = self.ctx.output_device()

        for combo, devices, kind, selected in (
            (self.input_combo, self.ctx.devices.input_devices(), "in", previous_input),
            (self.output_combo, self.ctx.devices.output_devices(), "out", previous_output),
        ):
            combo.blockSignals(True)
            combo.clear()
            for dev in devices:
                marker = "  · varsayılan" if (
                    dev.is_default_input if kind == "in" else dev.is_default_output
                ) else ""
                combo.addItem(f"{dev.name}  ·  {dev.hostapi_name}{marker}", dev.index)
            if selected is not None:
                idx = combo.findData(selected.index)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.setEnabled(combo.count() > 0)
            if combo.count() == 0:
                combo.addItem("Cihaz bulunamadı", None)
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(20)
            combo.blockSignals(False)

        self._update_device_tooltips()
        self._load_rates()
        self._select_combo_data(self.channel_combo, self.ctx.config.get_int("channels", 1))
        self._select_combo_data(
            self.format_combo,
            self.ctx.config.get_str("recording_subtype", "PCM_24"),
        )

        has_input = bool(self.ctx.devices.input_devices())
        self.record_hint.setText(
            "Kayıt başlatmak için alt taşıma çubuğundaki kırmızı kayıt düğmesini kullanın (Ctrl+R)."
            if has_input
            else "Ses giriş cihazı bulunamadı. Mikrofonun bağlı olduğundan emin olup 'Cihazları Yenile'ye basın."
        )
        self._update_monitor_availability()

    def _load_rates(self) -> None:
        rates = list(C.SUPPORTED_SAMPLE_RATES)
        device = self.ctx.input_device()
        if device is not None and int(device.default_samplerate) not in rates:
            rates.append(int(device.default_samplerate))
        current = self.ctx.config.get_int("sample_rate", C.DEFAULT_SAMPLE_RATE)
        self.rate_combo.blockSignals(True)
        self.rate_combo.clear()
        for rate in rates:
            self.rate_combo.addItem(f"{rate} Hz", rate)
        self._select_combo_data(self.rate_combo, current)
        self.rate_combo.blockSignals(False)

    @staticmethod
    def _select_combo_data(combo: QComboBox, data) -> None:
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _update_device_tooltips(self) -> None:
        for combo, purpose in (
            (self.input_combo, "Kayıt giriş cihazı"),
            (self.output_combo, "Oynatma çıkış cihazı"),
        ):
            text = combo.currentText().replace("  ·  ", " · ")
            combo.setToolTip(
                f"{purpose}: {text}" if combo.currentData() is not None else "Cihaz bulunamadı"
            )

    def _update_monitor_availability(self) -> None:
        has_input = bool(self.ctx.devices.input_devices())
        self.monitor_check.setEnabled(has_input and not self.ctx.recorder.is_active)

    # ------------------------------------------------------------ değişiklik
    def _remember_selected(self, kind: str) -> None:
        combo = self.input_combo if kind == "in" else self.output_combo
        index = combo.currentData()
        if index is None:
            return
        dev = self.ctx.devices.get(int(index))
        if dev is not None:
            self.ctx.remember_device(dev, "input" if kind == "in" else "output")
            self.settingsChanged.emit()

    def _on_input_changed(self) -> None:
        self._remember_selected("in")
        self._update_device_tooltips()
        self._load_rates()

    def _on_output_changed(self) -> None:
        self._remember_selected("out")
        self._update_device_tooltips()

    def _on_rate_changed(self) -> None:
        self.ctx.config.set("sample_rate", int(self.rate_combo.currentData()))
        self.settingsChanged.emit()

    def _on_channel_changed(self) -> None:
        self.ctx.config.set("channels", int(self.channel_combo.currentData()))
        self.settingsChanged.emit()

    def _on_format_changed(self) -> None:
        self.ctx.config.set("recording_subtype", str(self.format_combo.currentData()))
        self.settingsChanged.emit()

    def _on_monitor_toggled(self, checked: bool) -> None:
        if checked:
            try:
                device = self.ctx.input_device()
                self.ctx.monitor.start(
                    device_index=device.index if device else None,
                    samplerate=self.ctx.config.get_int("sample_rate", 48000),
                    channels=self.ctx.config.get_int("channels", 1),
                )
                self.check_label.setText(
                    "Dinleniyor… Sinyal algılanana kadar konuşun. Kayıt yapılmıyor."
                )
                self.check_label.setStyleSheet(f"color: {C.COLOR_TEXT_SECONDARY};")
                self.monitorStateChanged.emit(True)
            except Exception:
                self.monitor_check.blockSignals(True)
                self.monitor_check.setChecked(False)
                self.monitor_check.blockSignals(False)
                self.monitorStateChanged.emit(False)
        else:
            self.ctx.monitor.stop()
            self.level_meter.reset()
            self.show_level(None)
            self.monitorStateChanged.emit(False)

    def _on_monitor_error(self, message: str) -> None:
        self.check_label.setText(message)
        self.check_label.setStyleSheet(f"color: {C.COLOR_VU_RED};")

    def stop_monitor(self) -> None:
        was_active = self.monitor_check.isChecked() or self.ctx.monitor.active
        if self.monitor_check.isChecked():
            self.monitor_check.blockSignals(True)
            self.monitor_check.setChecked(False)
            self.monitor_check.blockSignals(False)
        self.ctx.monitor.stop()
        self.level_meter.reset()
        if was_active:
            self.monitorStateChanged.emit(False)

    # ============================================================ durum
    def set_recording_state(self, state: str) -> None:
        locked = state in {"starting", "recording", "paused", "stopping"}
        for widget in (
            self.input_combo,
            self.output_combo,
            self.rate_combo,
            self.channel_combo,
            self.format_combo,
            self.label_edit,
            self.refresh_btn,
            self.calib_wizard_btn,
        ):
            widget.setEnabled(not locked)
        self.monitor_check.setEnabled(not locked and bool(self.ctx.devices.input_devices()))

        if state == "recording":
            self.record_hint.setText(
                "Kayıt sürüyor. Dosya güvenli biçimde .part olarak yazılıyor; "
                "bitince kalıcı WAV dosyasına taşınır."
            )
            self.record_hint.setStyleSheet(f"color: {C.COLOR_ACCENT_GLOW};")
        elif state == "paused":
            self.record_hint.setText("Kayıt duraklatıldı. Devam etmek için oynat/duraklat düğmesini kullanın.")
            self.record_hint.setStyleSheet(f"color: {C.COLOR_VU_YELLOW};")
        elif state == "stopping":
            self.record_hint.setText("Kayıt sonlandırılıyor, dosya kapatılıyor…")
            self.record_hint.setStyleSheet(f"color: {C.COLOR_TEXT_SECONDARY};")
        elif state in {"idle", "finished", "cancelled", "error"}:
            self.record_hint.setText(
                "Kayıt başlatmak için alt taşıma çubuğundaki kırmızı kayıt düğmesini kullanın (Ctrl+R)."
            )
            self.record_hint.setStyleSheet("")

    def show_level(self, snap: Optional[LevelSnapshot]) -> None:
        if snap is None:
            self.level_meter.reset()
            self.check_label.setText(
                "Giriş kontrolü kapalı — mikrofonu test etmek için 'Girişi dinle' seçeneğini açın."
            )
            self.check_label.setStyleSheet(f"color: {C.COLOR_TEXT_MUTED};")
            return
        self.level_meter.set_level(snap.peak_db, snap.hold_db, snap.clip_count > 0 or snap.clip_active)
        status, message = self._input_check.evaluate(snap)
        colors = {
            "good": C.COLOR_VU_GREEN,
            "low": C.COLOR_VU_YELLOW,
            "hot": C.COLOR_VU_YELLOW,
            "clipping": C.COLOR_VU_RED,
            "silent": C.COLOR_TEXT_MUTED,
        }
        self.check_label.setText(message)
        self.check_label.setStyleSheet(f"color: {colors.get(status, C.COLOR_TEXT_SECONDARY)};")

    # ============================================================ waveform
    def _set_file_label(self, text: str, tooltip: str = "") -> None:
        self._file_text = text
        width = max(160, self.width() - 520)
        metrics = QFontMetrics(self.file_label.font())
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideMiddle, width)
        self.file_label.setText(elided.upper())
        self.file_label.setToolTip(tooltip or text)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._file_text:
            self._set_file_label(self._file_text, self.file_label.toolTip())

    def show_recording_waveform(self, mins, maxs, duration: float) -> None:
        self.waveform.set_peaks(mins, maxs, duration)
        self.duration_label.setText(format_duration(duration))

    def show_recording_result(
        self,
        result: RecordingResult,
        mins=None,
        maxs=None,
    ) -> None:
        if not result.path:
            self._set_file_label("Kayıt tamamlanamadı")
            return
        path = Path(result.path)
        self._raw_take_path = str(path)
        self._clean_take_path = None
        self._current_path = str(path)
        self._set_file_label(path.name, str(path))
        self.duration_label.setText(format_duration(result.duration))
        if mins is not None and maxs is not None:
            self.show_recording_waveform(mins, maxs, result.duration)

        if self.clean_chain_check.isChecked() or abs(self.input_gain_spin.value()) > 1e-3:
            clean_p = self.apply_post_recording_clean_chain(str(path))
            if clean_p and Path(clean_p).exists():
                self._clean_take_path = clean_p
                self.ab_take_btn.setEnabled(True)
                self.ab_take_btn.blockSignals(True)
                self.ab_take_btn.setChecked(False)
                self.ab_take_btn.setText("Mod: Ham Kayıt (A) ↔ Temiz (B)")
                self.ab_take_btn.blockSignals(False)
                # Ayrıca son kayıtlara temiz kopyayı da ekle
                recent = list(self.ctx.config.get("recent_recordings", []) or [])
                if clean_p not in recent:
                    recent.insert(0, clean_p)
                    self.ctx.config.set("recent_recordings", recent[:20])
        else:
            self.ab_take_btn.setEnabled(False)
            self.ab_take_btn.setText("Mod: Ham Kayıt (A)")

        self.refresh_recent()

    def show_file(self, info: PlaybackInfo, mins=None, maxs=None) -> None:
        path = Path(info.path)
        self._current_path = str(path)
        self._set_file_label(path.name, str(path))
        self.duration_label.setText(format_duration(info.duration))
        if mins is not None and maxs is not None:
            self.waveform.set_peaks(mins, maxs, info.duration)
        else:
            self.waveform.clear("Waveform hazırlanıyor…")

    def set_waveform_peaks(self, mins, maxs, duration: float) -> None:
        self.waveform.set_peaks(mins, maxs, duration)

    def set_playback_position(self, seconds: float) -> None:
        self.waveform.set_position(seconds)

    @property
    def current_path(self) -> Optional[str]:
        return self._current_path

    @property
    def raw_take_path(self) -> Optional[str]:
        return self._raw_take_path

    @property
    def clean_take_path(self) -> Optional[str]:
        return self._clean_take_path

    # ============================================================ analiz
    def _set_analysis_placeholder(self) -> None:
        self.score_value.setText("—")
        self.score_value.setStyleSheet(f"color: {C.COLOR_TEXT_PRIMARY};")
        self.score_value.setToolTip("")
        for label in self._metrics.values():
            label.setText("—")
        self.format_label.setText("Biçim: —")
        self.warnings_box.setPlainText("")
        self.analysis_state.setText("Dosya seçilmedi")
        self.analysis_state.setStyleSheet(f"color: {C.COLOR_TEXT_MUTED};")
        self.method_label.setText(
            "Loudness yöntemi: ITU-R BS.1770-4 (K-ağırlıklı, kapılı entegre)."
        )

    def clear_analysis(self) -> None:
        """Yeni dosya/kayıt seçildiğinde eski ölçümleri temizler."""
        self._set_analysis_placeholder()

    def show_analysis(self, analysis: AudioAnalysis) -> None:
        self.analysis_state.setText("Ölçüm tamamlandı")
        if analysis.score is not None:
            self.score_value.setText(str(analysis.score))
            color = (
                C.COLOR_VU_GREEN
                if analysis.score >= 80
                else C.COLOR_VU_YELLOW
                if analysis.score >= 60
                else C.COLOR_VU_RED
            )
            self.score_value.setStyleSheet(f"color: {color};")
            breakdown = ", ".join(f"{k}: {v:+d}" for k, v in analysis.score_breakdown.items())
            self.score_value.setToolTip(f"Puan kırılımı — {breakdown}")
        else:
            self.score_value.setText("—")

        self._metrics["peak"].setText(_db_with_unit(analysis.peak_dbfs, "dBFS"))
        self._metrics["rms"].setText(_db_with_unit(analysis.rms_dbfs, "dBFS"))
        self._metrics["loudness"].setText(_db_with_unit(analysis.loudness_lufs, "LUFS"))
        self._metrics["crest"].setText(_db_with_unit(analysis.crest_db, "dB"))
        self._metrics["noise"].setText(_db_with_unit(analysis.noise_floor_dbfs, "dBFS"))
        self._metrics["snr"].setText(_db_with_unit(analysis.snr_estimate_db, "dB"))
        self._metrics["silence"].setText(f"{analysis.silence_ratio * 100:.0f} %")
        self._metrics["clipping"].setText(
            "Yok" if analysis.clipped_samples == 0 else str(analysis.clipped_samples)
        )
        self.format_label.setText(
            f"Biçim: {analysis.file_format or '—'} {analysis.subtype or ''} · "
            f"{analysis.samplerate} Hz · {analysis.channels} kanal · "
            f"{format_duration(analysis.duration)}"
        )
        if analysis.loudness_partial:
            self.method_label.setText(
                "Loudness yöntemi: ITU-R BS.1770-4. Dosya 10 dakikadan uzun olduğu için "
                "yalnızca ilk 10 dakika ölçüldü."
            )
        else:
            self.method_label.setText(
                "Loudness yöntemi: ITU-R BS.1770-4 (K-ağırlıklı, kapılı entegre)."
            )
        warnings = [f"• {w}" for w in analysis.warnings]
        self.warnings_box.setPlainText("\n".join(warnings) if warnings else "Uyarı yok.")
        self.analyze_btn.setEnabled(True)
        if not self.ctx.monitor.active:
            self.show_level(None)

    def show_analysis_error(self, message: str) -> None:
        self.analysis_state.setText("Analiz başarısız")
        self.analysis_state.setStyleSheet(f"color: {C.COLOR_VU_RED};")
        self.warnings_box.setPlainText(f"• {message}")
        self.analyze_btn.setEnabled(True)

    def set_analysis_busy(self, busy: bool, path: Optional[str] = None) -> None:
        self.analyze_btn.setEnabled(not busy)
        if busy:
            self.analysis_state.setText("Analiz ediliyor…")
            self.analysis_state.setStyleSheet(f"color: {C.COLOR_TEXT_SECONDARY};")
            self.score_value.setText("…")
            self.score_value.setStyleSheet(f"color: {C.COLOR_TEXT_MUTED};")
            if path:
                self._set_file_label(Path(path).name, path)

    def _on_analyze_clicked(self) -> None:
        path = self._current_path
        if not path:
            self.analysis_state.setText("Önce bir kayıt seçin")
            return
        self.analyzeRequested.emit(path)

    # ============================================================ son kayıtlar
    def refresh_recent(self) -> None:
        self.recent_list.clear()
        paths = list(self.ctx.config.get("recent_recordings", []) or [])
        existing = [p for p in paths if Path(p).exists()][:12]
        if existing != paths:
            self.ctx.config.set("recent_recordings", existing)
        for raw in existing:
            path = Path(raw)
            try:
                size = format_file_size(path.stat().st_size)
            except OSError:
                size = "—"
            item = QListWidgetItem(f"{path.name}\n{size}")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.recent_list.addItem(item)
        if not existing:
            item = QListWidgetItem("Henüz kayıt yok.\nİlk kaydınızı almak için kayıt düğmesine basın.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent_list.addItem(item)

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.openFileRequested.emit(str(path))

    # ============================================================ dışarıdan
    def set_label_text(self, text: str) -> None:
        self.label_edit.setText(text)

    def set_settings_from_config(self) -> None:
        """Ayarlar ekranından döndükten sonra seçimleri tazele."""
        self._select_combo_data(
            self.rate_combo, self.ctx.config.get_int("sample_rate", C.DEFAULT_SAMPLE_RATE)
        )
        self._select_combo_data(self.channel_combo, self.ctx.config.get_int("channels", 1))
        self._select_combo_data(
            self.format_combo, self.ctx.config.get_str("recording_subtype", "PCM_24")
        )

    def playback_state_changed(self, state: str) -> None:
        if state == "idle" and self.ctx.current_info is not None:
            self.waveform.set_position(self.ctx.current_info.duration)
