"""
Ayarlar görünümü.

Tüm değişiklikler anında `config.json` dosyasına atomik olarak yazılır ve
uygulama yeniden başlatıldığında korunur.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ai.media_extractor import MediaExtractor
from audio.recorder import recover_pending_recordings
from core import constants as C
from core import paths
from core.i18n import SUPPORTED_LANGUAGES, get_language_spec, search_languages
from core.logger import get_logger
from ui import icons
from ui.context import AppContext

logger = get_logger("ui.settings")

LICENSES = [
    ("PySide6 / Qt", "6.11.1", "LGPL-3.0", "Dinamik bağlama. LGPL yükümlülükleri (bildirim, kütüphaneyi değiştirme hakkı) dağıtımda korunmalı."),
    ("PyAV / FFmpeg", "18.0.0 (libavcodec 62.28)", "BSD-3-Clause / LGPL", "Gömülü FFmpeg kitaplıklarıyla video/ses kod çözme, kodlama ve NVENC desteği."),
    ("Lucide Icons (SVG)", "1.0", "ISC License", "Gömülü ölçeklenebilir vektör simge çizimleri; ticari ve açık kaynak kullanıma uygun."),
    ("NumPy", "2.5.1", "BSD-3-Clause", "İzin verici lisans."),
    ("SciPy", "1.18.1", "BSD-3-Clause", "İzin verici lisans."),
    ("Pillow (PIL)", "11.x", "MIT-CMU", "Görsel işleme, altyazı ve kare kompozisyonu."),
    ("sounddevice", "0.5.6", "MIT", "PortAudio Python bağlayıcısı."),
    ("PortAudio", "19.7.0", "MIT", "Gerçek zamanlı ses G/Ç."),
    ("soundfile", "0.14.0", "BSD-3-Clause", "libsndfile Python bağlayıcısı."),
    ("libsndfile", "1.2.2", "LGPL-2.1", "Dinamik bağlantı. WAV/FLAC/OGG/MP3 okuma-yazma."),
    ("faster-whisper / CTranslate2", "1.1+", "MIT", "Yerel çevrimdışı konuşma tanıma (ASR) motoru."),
    ("onnxruntime", "1.28.0", "MIT", "Yerel yapay zekâ model çıkarım motoru."),
    ("pytest", "9.1.1", "MIT", "Yalnızca geliştirme/test bağımlılığı."),
]


class SettingsView(QWidget):
    settingsChanged = Signal()
    recordingsChanged = Signal()

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx
        self._build_ui()
        self.refresh()

    # ================================================================ arayüz
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 14)
        root.setSpacing(12)

        header = QVBoxLayout()
        header.setSpacing(2)
        title = QLabel("Ayarlar")
        title.setObjectName("ViewTitle")
        subtitle = QLabel(
            "Değişiklikler anında kaydedilir. Ses biçimi ayarları bir sonraki kayıtta geçerli olur."
        )
        subtitle.setObjectName("ViewSubtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll, 1)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 4, 8, 4)
        layout.setSpacing(12)

        layout.addWidget(self._build_audio_card())
        layout.addWidget(self._build_recording_card())
        layout.addWidget(self._build_codec_card())
        layout.addWidget(self._build_interface_card())
        layout.addWidget(self._build_data_card())
        layout.addWidget(self._build_about_card())
        layout.addStretch(1)
        scroll.setWidget(content)

    # ------------------------------------------------------------ yardımcılar
    @staticmethod
    def _card(title: str, description: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(10)
        section = QLabel(title)
        section.setObjectName("SectionTitle")
        layout.addWidget(section)
        if description:
            desc = QLabel(description)
            desc.setObjectName("MutedSmall")
            desc.setWordWrap(True)
            layout.addWidget(desc)
        return card, layout

    @staticmethod
    def _row(label: str, widget: QWidget, hint: str = "") -> QWidget:
        wrapper = QWidget()
        box = QHBoxLayout(wrapper)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(12)
        caption = QLabel(label)
        caption.setFixedWidth(170)
        caption.setObjectName("Muted")
        box.addWidget(caption)
        box.addWidget(widget, 1)
        if hint:
            hint_label = QLabel(hint)
            hint_label.setObjectName("MutedSmall")
            hint_label.setWordWrap(True)
            hint_label.setFixedWidth(240)
            box.addWidget(hint_label)
        return wrapper

    # ------------------------------------------------------------- ses kartı
    def _build_audio_card(self) -> QWidget:
        card, layout = self._card(
            "SES CİHAZLARI",
            "Kayıt ve oynatma cihazları. Desteklenmeyen hız/kanal kombinasyonları kayıt "
            "başlatılırken doğrulanır ve açık hata mesajı gösterilir.",
        )
        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        self.rate_combo = QComboBox()
        for rate in C.SUPPORTED_SAMPLE_RATES:
            self.rate_combo.addItem(f"{rate} Hz", rate)
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("Mono (1 kanal)", 1)
        self.channel_combo.addItem("Stereo (2 kanal)", 2)
        self.format_combo = QComboBox()
        for label, subtype in C.RECORDING_SUBTYPES.items():
            self.format_combo.addItem(label, subtype)

        layout.addWidget(self._row("Giriş cihazı", self.input_combo))
        layout.addWidget(self._row("Çıkış cihazı", self.output_combo))
        layout.addWidget(self._row("Örnekleme hızı", self.rate_combo))
        layout.addWidget(self._row("Kayıt kanalı", self.channel_combo))
        layout.addWidget(self._row("Kayıt biçimi", self.format_combo))
        self.device_note = QLabel("")
        self.device_note.setObjectName("MutedSmall")
        self.device_note.setWordWrap(True)
        layout.addWidget(self.device_note)

        refresh = QPushButton("Cihazları yeniden tara")
        refresh.setProperty("variant", "ghost")
        refresh.setIcon(icons.icon("refresh", size=14))
        refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh.clicked.connect(self._refresh_devices)
        row = QHBoxLayout()
        row.addWidget(refresh)
        row.addStretch(1)
        layout.addLayout(row)

        self.input_combo.currentIndexChanged.connect(lambda: self._on_device_changed("input"))
        self.output_combo.currentIndexChanged.connect(lambda: self._on_device_changed("output"))
        self.rate_combo.currentIndexChanged.connect(
            lambda: self._set_config("sample_rate", int(self.rate_combo.currentData()))
        )
        self.channel_combo.currentIndexChanged.connect(
            lambda: self._set_config("channels", int(self.channel_combo.currentData()))
        )
        self.format_combo.currentIndexChanged.connect(
            lambda: self._set_config("recording_subtype", str(self.format_combo.currentData()))
        )
        return card

    # ----------------------------------------------------------- kayıt kartı
    def _build_recording_card(self) -> QWidget:
        card, layout = self._card("KAYIT", "Dosyaların yazılacağı klasör ve kayıt davranışı.")
        self.folder_label = QLabel("—")
        self.folder_label.setObjectName("Mono")
        self.folder_label.setWordWrap(True)
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.folder_label)

        buttons = QHBoxLayout()
        change = QPushButton("Klasör değiştir")
        change.setIcon(icons.icon("folder", size=14))
        change.clicked.connect(self._choose_recordings_dir)
        buttons.addWidget(change)
        open_btn = QPushButton("Aç")
        open_btn.setProperty("variant", "ghost")
        open_btn.clicked.connect(lambda: self._open_path(self.ctx.recordings_dir()))
        buttons.addWidget(open_btn)
        reset = QPushButton("Varsayılana dön")
        reset.setProperty("variant", "ghost")
        reset.clicked.connect(self._reset_recordings_dir)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.countdown_spin = QSpinBox()
        self.countdown_spin.setRange(0, 10)
        self.countdown_spin.setSuffix(" sn")
        self.countdown_spin.setToolTip("Kayıt başlamadan önce geri sayım")
        self.countdown_spin.valueChanged.connect(
            lambda value: self._set_config("countdown_sec", int(value))
        )
        layout.addWidget(
            self._row("Geri sayım", self.countdown_spin, "0 = geri sayım yok")
        )

        note = QLabel(
            "Dosya adı: tarih_saat_etiket.wav. Var olan dosyanın üzerine asla yazılmaz; "
            "çakışmada otomatik olarak yeni ad üretilir."
        )
        note.setObjectName("MutedSmall")
        note.setWordWrap(True)
        layout.addWidget(note)
        return card

    # -------------------------------------------------------- kodek kartı
    def _build_codec_card(self) -> QWidget:
        card, layout = self._card(
            "MEDYA VE KODEK UYUMLULUĞU",
            "Gömülü FFmpeg (PyAV) ve libsndfile motorlarının bu bilgisayarda doğrulanan "
            "gerçek video, ses, görsel ve altyazı kodek desteği.",
        )
        self.codec_summary_label = QLabel("")
        self.codec_summary_label.setObjectName("MutedSmall")
        self.codec_summary_label.setWordWrap(True)
        layout.addWidget(self.codec_summary_label)

        row = QHBoxLayout()
        self.codec_btn = QPushButton("  Medya ve Kodek Uyumluluk Tablosunu Görüntüle")
        self.codec_btn.setIcon(icons.icon("shield-check", size=14))
        self.codec_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.codec_btn.clicked.connect(self._show_codec_dialog)
        row.addWidget(self.codec_btn)
        row.addStretch(1)
        layout.addLayout(row)
        return card

    # -------------------------------------------------------- arayüz kartı
    def _build_interface_card(self) -> QWidget:
        card, layout = self._card(
            "DİL, ARAYÜZ VE ERİŞİLEBİLİRLİK (30 DİL DESTEĞİ)",
            "Arayüz dili ve varsayılan transkripsiyon dili birbirinden bağımsızdır. "
            "Seçilen dil anında kaydedilir ve yeniden başlatma gerektirmeden arayüze uygulanır.",
        )

        self.lang_search_edit = QLineEdit()
        self.lang_search_edit.setPlaceholderText("30 dil içinde ara (ör. Türkçe, English, العربية, 日本語, Deutsch)...")
        self.lang_search_edit.setClearButtonEnabled(True)
        self.lang_search_edit.textChanged.connect(self._filter_language_combos)
        layout.addWidget(self._row("Dil Arama", self.lang_search_edit, "Yerel ad, Türkçe ad veya ISO kodu ile filtrele"))

        self.ui_lang_combo = QComboBox()
        for spec in SUPPORTED_LANGUAGES:
            self.ui_lang_combo.addItem(spec.display_label, spec.code)
        self.ui_lang_combo.currentIndexChanged.connect(self._on_ui_lang_changed)
        layout.addWidget(
            self._row("Arayüz Dili", self.ui_lang_combo, "Uygulama menüleri ve temel kontroller")
        )

        self.trans_lang_combo = QComboBox()
        self.trans_lang_combo.addItem("Otomatik Algıla (Auto-Detect)", "auto")
        for spec in SUPPORTED_LANGUAGES:
            self.trans_lang_combo.addItem(spec.display_label, spec.code)
        self.trans_lang_combo.currentIndexChanged.connect(self._on_trans_lang_changed)
        layout.addWidget(
            self._row(
                "Transkripsiyon Dili",
                self.trans_lang_combo,
                "Yeni ses/video transkripsiyonlarında varsayılan dil",
            )
        )

        self.lang_info_label = QLabel("")
        self.lang_info_label.setObjectName("MutedSmall")
        self.lang_info_label.setWordWrap(True)
        layout.addWidget(self.lang_info_label)

        theme = QLabel("Koyu stüdyo teması — Siyah & Kırmızı (#090A0C / #E53945)")
        theme.setObjectName("Muted")
        layout.addWidget(self._row("Tema", theme, "Full HD (1920×1080) ve Yüksek DPI uyumlu"))

        self.reduce_motion_check = QCheckBox("Hareket azaltma (animasyonları ve geçişleri sadeleştir)")
        self.reduce_motion_check.setChecked(bool(self.ctx.config.get("reduce_motion", False)))
        self.reduce_motion_check.toggled.connect(
            lambda checked: self._set_config("reduce_motion", bool(checked))
        )
        layout.addWidget(self._row("Animasyonlar", self.reduce_motion_check))
        return card

    def _filter_language_combos(self, query: str) -> None:
        curr_ui = self.ctx.i18n.get_ui_language()
        curr_tr = self.ctx.i18n.get_transcription_language()

        ui_matches = search_languages(query, include_auto=False)
        self.ui_lang_combo.blockSignals(True)
        self.ui_lang_combo.clear()
        for code, label in ui_matches:
            self.ui_lang_combo.addItem(label, code)
        self._select(self.ui_lang_combo, curr_ui)
        self.ui_lang_combo.blockSignals(False)

        tr_matches = search_languages(query, include_auto=True)
        self.trans_lang_combo.blockSignals(True)
        self.trans_lang_combo.clear()
        for code, label in tr_matches:
            self.trans_lang_combo.addItem(label, code)
        self._select(self.trans_lang_combo, curr_tr)
        self.trans_lang_combo.blockSignals(False)

    def _on_ui_lang_changed(self) -> None:
        code = self.ui_lang_combo.currentData()
        if not code:
            return
        self.ctx.i18n.set_ui_language(str(code))
        self._update_lang_info_label()
        self.settingsChanged.emit()

    def _on_trans_lang_changed(self) -> None:
        code = self.trans_lang_combo.currentData()
        if not code:
            return
        self.ctx.i18n.set_transcription_language(str(code))
        self._update_lang_info_label()
        self.settingsChanged.emit()

    def _update_lang_info_label(self) -> None:
        if not hasattr(self, "lang_info_label"):
            return
        ui_spec = get_language_spec(self.ctx.i18n.get_ui_language())
        tr_code = self.ctx.i18n.get_transcription_language()
        tr_txt = (
            "Otomatik Algıla"
            if tr_code == "auto"
            else get_language_spec(tr_code).display_label
        )
        dir_txt = "Sağdan Sola (RTL)" if ui_spec.is_rtl else "Soldan Sağa (LTR)"
        self.lang_info_label.setText(
            f"Aktif Arayüz Dili: {ui_spec.display_label} · Yön: {dir_txt} · Yazı Sistemi: {ui_spec.script.upper()}\n"
            f"Varsayılan Transkripsiyon Dili: {tr_txt}\n"
            f"Unicode Önizleme: {ui_spec.sample_text}"
        )

    # ------------------------------------------------------------ veri kartı
    def _build_data_card(self) -> QWidget:
        card, layout = self._card(
            "VERİ, GÜNLÜKLER VE KURTARMA",
            "Beklenmedik kapanmada kayıtlar .part dosyası olarak kalır. Kurtarma, okunabilen "
            "dosyaları son adına taşır; okunamayan dosyalara dokunmaz.",
        )
        buttons = QHBoxLayout()
        logs = QPushButton("Günlük klasörü")
        logs.setIcon(icons.icon("folder", size=14))
        logs.clicked.connect(lambda: self._open_path(paths.logs_dir()))
        buttons.addWidget(logs)

        config_btn = QPushButton("Ayar dosyası")
        config_btn.setIcon(icons.icon("gear", size=14))
        config_btn.clicked.connect(self._open_config)
        buttons.addWidget(config_btn)

        recover = QPushButton("Bekleyen kayıtları kurtar")
        recover.setIcon(icons.icon("refresh", size=14))
        recover.clicked.connect(self._recover)
        buttons.addWidget(recover)

        backup = QPushButton("Yedekleme")
        backup.setEnabled(False)
        backup.setToolTip("Proje yedekleme Aşama 3'te eklenecek — Geliştiriliyor")
        buttons.addWidget(backup)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.recovery_label = QLabel("")
        self.recovery_label.setObjectName("MutedSmall")
        self.recovery_label.setWordWrap(True)
        layout.addWidget(self.recovery_label)
        return card

    # -------------------------------------------------------- hakkında kartı
    def _build_about_card(self) -> QWidget:
        card, layout = self._card("SÜRÜM, LİSANSLAR VE GİZLİLİK")
        self.version_label = QLabel("")
        self.version_label.setObjectName("MutedSmall")
        self.version_label.setWordWrap(True)
        self.version_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.version_label)

        privacy = QLabel(
            "Telemetri yok. Uygulama hiçbir ses, metin veya kullanım verisini otomatik olarak "
            "göndermez. Bulut AI varsayılan olarak kapalıdır ve bu sürümde uygulanmamıştır."
        )
        privacy.setObjectName("MutedSmall")
        privacy.setWordWrap(True)
        layout.addWidget(privacy)

        row = QHBoxLayout()
        licenses_btn = QPushButton("Lisanslar ve üçüncü taraf bildirimleri")
        licenses_btn.setIcon(icons.icon("info", size=14))
        licenses_btn.clicked.connect(self._show_licenses)
        row.addWidget(licenses_btn)
        row.addStretch(1)
        layout.addLayout(row)
        return card

    # ================================================================ doldur
    def refresh(self) -> None:
        self._refresh_devices()
        self.folder_label.setText(str(self.ctx.recordings_dir()))
        self.countdown_spin.blockSignals(True)
        self.countdown_spin.setValue(self.ctx.config.get_int("countdown_sec", 0))
        self.countdown_spin.blockSignals(False)
        self._select(self.rate_combo, self.ctx.config.get_int("sample_rate", 48000))
        self._select(self.channel_combo, self.ctx.config.get_int("channels", 1))
        self._select(
            self.format_combo, self.ctx.config.get_str("recording_subtype", "PCM_24")
        )
        if hasattr(self, "ui_lang_combo"):
            self.ui_lang_combo.blockSignals(True)
            self._select(self.ui_lang_combo, self.ctx.i18n.get_ui_language())
            self.ui_lang_combo.blockSignals(False)
        if hasattr(self, "trans_lang_combo"):
            self.trans_lang_combo.blockSignals(True)
            self._select(self.trans_lang_combo, self.ctx.i18n.get_transcription_language())
            self.trans_lang_combo.blockSignals(False)
        self._update_lang_info_label()
        if hasattr(self, "reduce_motion_check"):
            self.reduce_motion_check.blockSignals(True)
            self.reduce_motion_check.setChecked(bool(self.ctx.config.get("reduce_motion", False)))
            self.reduce_motion_check.blockSignals(False)
        if hasattr(self, "codec_summary_label"):
            caps = MediaExtractor.get_supported_capabilities()
            nvenc_txt = "Aktif (NVIDIA NVENC H.264)" if caps.get("nvenc_h264") else "CPU Kodlama (libx264)"
            self.codec_summary_label.setText(
                f"FFmpeg Motoru: PyAV {caps['pyav_version']} (libavcodec {caps['libavcodec']}) · "
                f"Donanım Hızlandırma: {nvenc_txt}\n"
                f"Video: {', '.join(C.VIDEO_EXTENSIONS)} · "
                f"Ses: {', '.join(C.AUDIO_EXTENSIONS)} · "
                f"Görsel: {', '.join(C.IMAGE_EXTENSIONS)} · "
                f"Altyazı: {', '.join(C.SUBTITLE_EXTENSIONS)}"
            )
        self.version_label.setText(self._environment_text())

    def _refresh_devices(self) -> None:
        self.ctx.refresh_devices()
        self._fill_combo(self.input_combo, self.ctx.devices.input_devices(), self.ctx.input_device())
        self._fill_combo(self.output_combo, self.ctx.devices.output_devices(), self.ctx.output_device())
        inputs = len(self.ctx.devices.input_devices())
        outputs = len(self.ctx.devices.output_devices())
        self.device_note.setText(
            f"{inputs} giriş, {outputs} çıkış cihazı bulundu. "
            "Cihaz adları WASAPI/DirectSound üzerinden okunur."
        )

    @staticmethod
    def _fill_combo(combo: QComboBox, devices, selected) -> None:
        combo.blockSignals(True)
        combo.clear()
        for dev in devices:
            combo.addItem(f"{dev.name}  ·  {dev.hostapi_name}", dev.index)
        if selected is not None:
            idx = combo.findData(selected.index)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        if combo.count() == 0:
            combo.addItem("Cihaz bulunamadı", None)
        combo.blockSignals(False)

    @staticmethod
    def _select(combo: QComboBox, data) -> None:
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    @staticmethod
    def _environment_text() -> str:
        try:
            import sounddevice as sd
            import soundfile as sf

            pa = sd.get_portaudio_version()
            pa_text = pa[1] if isinstance(pa, tuple) and len(pa) > 1 else str(pa)
            sf_text = sf.__libsndfile_version__
        except Exception:
            pa_text = "bilinmiyor"
            sf_text = "bilinmiyor"
        try:
            import PySide6

            qt_text = PySide6.__version__
        except Exception:
            qt_text = "bilinmiyor"
        try:
            import onnxruntime as ort

            providers = ", ".join(ort.get_available_providers())
        except Exception:
            providers = "yok"

        return (
            f"{C.APP_NAME} {C.APP_VERSION} — {C.APP_SUBTITLE}\n"
            f"Python {platform.python_version()} · Qt/PySide6 {qt_text} · "
            f"PortAudio {pa_text} · libsndfile {sf_text}\n"
            f"ONNX Runtime sağlayıcıları: {providers}\n"
            "Uygulama lisansı: henüz belirlenmedi (dağıtım modeli değerlendiriliyor)."
        )

    # ================================================================ eylemler
    def _set_config(self, key: str, value) -> None:
        self.ctx.config.set(key, value)
        self.settingsChanged.emit()

    def _on_device_changed(self, kind: str) -> None:
        combo = self.input_combo if kind == "input" else self.output_combo
        index = combo.currentData()
        if index is None:
            return
        dev = self.ctx.devices.get(int(index))
        if dev is not None:
            self.ctx.remember_device(dev, kind)
            self.settingsChanged.emit()

    def _choose_recordings_dir(self) -> None:
        current = str(self.ctx.recordings_dir())
        chosen = QFileDialog.getExistingDirectory(self, "Kayıt klasörünü seç", current)
        if not chosen:
            return
        target = Path(chosen)
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(
                self, "Klasör kullanılamıyor", f"Seçilen klasör oluşturulamadı:\n{target}\n\n{exc}"
            )
            return
        self.ctx.config.set_recordings_dir(target)
        self.folder_label.setText(str(target))
        self.settingsChanged.emit()
        self.recordingsChanged.emit()

    def _reset_recordings_dir(self) -> None:
        self.ctx.config.set("recordings_dir", "")
        self.folder_label.setText(str(self.ctx.recordings_dir()))
        self.settingsChanged.emit()
        self.recordingsChanged.emit()

    def _open_config(self) -> None:
        if not self.ctx.config.config_path.exists():
            self.ctx.config.save()
        self._open_path(self.ctx.config.config_path)

    def _open_path(self, path: Path) -> None:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.warning(
                self,
                "Açılamadı",
                f"Yol açılamadı:\n{path}\n\nDosya veya klasör mevcut olmayabilir.",
            )

    def _recover(self) -> None:
        try:
            recovered = recover_pending_recordings(self.ctx.recordings_dir())
        except Exception as exc:
            QMessageBox.critical(
                self, "Kurtarma başarısız", f"Bekleyen kayıtlar kurtarılırken hata oluştu:\n{exc}"
            )
            return
        if recovered:
            self.recovery_label.setText(f"{len(recovered)} kayıt kurtarıldı.")
            QMessageBox.information(
                self,
                "Kurtarma tamamlandı",
                f"{len(recovered)} yarım kayıt kurtarıldı:\n\n"
                + "\n".join(p.name for p in recovered),
            )
            self.recordingsChanged.emit()
        else:
            self.recovery_label.setText("Kurtarılacak yarım kayıt bulunamadı.")
            QMessageBox.information(
                self, "Kurtarma", "Kayıt klasöründe kurtarılabilir .part dosyası bulunamadı."
            )

    def _show_licenses(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Lisanslar ve üçüncü taraf bildirimleri")
        dialog.resize(760, 580)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()
        rows = "".join(
            f"<tr><td style='padding:6px 10px;'><b>{name}</b><br>"
            f"<span style='color:{C.COLOR_TEXT_MUTED};'>{version}</span></td>"
            f"<td style='padding:6px 10px;'>{license_name}</td>"
            f"<td style='padding:6px 10px;'>{note}</td></tr>"
            for name, version, license_name, note in LICENSES
        )
        browser.setHtml(
            f"""
            <h2 style="color:{C.COLOR_TEXT_PRIMARY}; margin-bottom:4px;">Lisanslar</h2>
            <p style="color:{C.COLOR_TEXT_SECONDARY};">
              Bu liste uygulamanın kullandığı bileşenleri ve doğrulanan lisanslarını gösterir.
              Her bağımlılığın ticari kullanım ve yeniden dağıtım koşulları ayrıca incelenmelidir.
            </p>
            <table cellspacing="0" cellpadding="0" width="100%" style="border-collapse:collapse;">
              <tr style="background:{C.COLOR_BG_SURFACE}; color:{C.COLOR_TEXT_SECONDARY};">
                <th align="left" style="padding:8px 10px;">Bileşen</th>
                <th align="left" style="padding:8px 10px;">Lisans</th>
                <th align="left" style="padding:8px 10px;">Not</th>
              </tr>
              {rows}
            </table>
            <h3 style="color:{C.COLOR_TEXT_PRIMARY};">Model ağırlıkları</h3>
            <p style="color:{C.COLOR_TEXT_SECONDARY};">
              VOX Studio bu sürümde hiçbir AI model ağırlığı içermez. Model ağırlıkları
              uygulamayla birlikte dağıtılmaz; ileride yalnızca kullanıcı onayıyla ve lisans
              metni gösterilerek indirilecektir.
            </p>
            <h3 style="color:{C.COLOR_TEXT_PRIMARY};">Uygulama lisansı</h3>
            <p style="color:{C.COLOR_TEXT_SECONDARY};">
              VOX Studio'nun kendi lisansı henüz belirlenmemiştir. Ticari dağıtım planlanıyorsa
              LGPL bileşenleri (Qt, libsndfile) ve FFmpeg derleme seçenekleri için hukuki
              inceleme gereklidir.
            </p>
            """
        )
        layout.addWidget(browser, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _show_codec_dialog(self) -> None:
        caps = MediaExtractor.get_supported_capabilities()
        dialog = QDialog(self)
        dialog.setWindowTitle("Medya ve Kodek Uyumluluk Tablosu")
        dialog.resize(740, 560)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()

        def _dict_rows(d: dict) -> str:
            out = ""
            for k, ok in d.items():
                badge = (
                    f"<span style='color:{C.COLOR_VU_GREEN}; font-weight:bold;'>✓ Destekleniyor</span>"
                    if ok
                    else f"<span style='color:{C.COLOR_VU_RED};'>✗ Yok</span>"
                )
                out += f"<tr><td style='padding:5px 10px;'>{k}</td><td style='padding:5px 10px;'>{badge}</td></tr>"
            return out

        browser.setHtml(
            f"""
            <h2 style="color:{C.COLOR_TEXT_PRIMARY}; margin-bottom:4px;">Medya ve Kodek Uyumluluğu</h2>
            <p style="color:{C.COLOR_TEXT_SECONDARY};">
              PyAV {caps['pyav_version']} · libavcodec {caps['libavcodec']} · libavformat {caps['libavformat']}
            </p>
            <h3 style="color:{C.COLOR_TEXT_PRIMARY};">Video Kod Çözücüler (Decoders)</h3>
            <table width="100%" cellspacing="0" cellpadding="0">{_dict_rows(caps['video_decoders'])}</table>
            <h3 style="color:{C.COLOR_TEXT_PRIMARY};">Video Kodlayıcılar (Encoders & GPU)</h3>
            <table width="100%" cellspacing="0" cellpadding="0">{_dict_rows(caps['video_encoders'])}</table>
            <h3 style="color:{C.COLOR_TEXT_PRIMARY};">Ses Kodekleri</h3>
            <table width="100%" cellspacing="0" cellpadding="0">{_dict_rows(caps['audio_codecs'])}</table>
            <h3 style="color:{C.COLOR_TEXT_PRIMARY};">Görsel ve Altyazı Biçimleri</h3>
            <p style="color:{C.COLOR_TEXT_SECONDARY};">
              <b>Görseller:</b> {', '.join(caps['image_formats'])}<br>
              <b>Altyazılar:</b> {', '.join(caps['subtitle_formats'])}
            </p>
            """
        )
        layout.addWidget(browser, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()
