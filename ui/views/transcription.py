"""
VOX Studio — Profesyonel Çok Dilli Transkripsiyon, Altyazı Düzenleme ve
Video Editörüne Aktarım Çalışma Alanı (Bölüm 1 & Referans Tasarım 3)

Özellikler:
- Referans Görsel 3 ile birebir örtüşen 4-Panelli Stüdyo Mimarisi:
  1. Sol Panel: Transkripsiyon Ayarları (Kaynak Medya Bilgisi, Diarizasyon Aç/Kapa,
     Dilbilgisi Otomatik Düzeltme, Satır Başına Karakter Sınırı, Dolgu Kelimeleri Yoksay)
  2. Orta-Sol Panel: Video Önizleme ve Canlı Altyazı Bindirme (Live Subtitle Overlay)
  3. Orta-Sağ Panel: Transkript Editörü (Zaman kodlu konuşmacı kartları, aktif satır vurgusu,
     doğrudan düzenleme, ses dinleme, satır bölme/birleştirme)
  4. Sağ Panel: Özellikler ve Altyazı Stili (Yazı Tipi, Kalınlık, Boyut, Hizalama,
     Arka Plan Şerit Opaklığı, Dış Çizgi, Zaman Ofset Çevirme Düğmeleri, SRT/VTT/TXT Radyo Seçimi)
  5. Alt Bölüm: Çok Kanallı Zaman Çizelgesi (Multitrack Timeline)
     - Sol araç şeridi: Seçim, Jilet, Kaydırma, Kalem
     - Audio A1 kanalı ve gerçek/sentetik tepe dalga formu
     - Subtitles kanalı ve tıklanabilir/sürüklenebilir interaktif altyazı blokları
     - Senkronize kırmızı oynatma kafası (Playhead) ve zaman cetveli
- Dosya kuyruğu (Batch Queue) ve sürükle-bırak desteği
- 30 Dil kataloğu + yerel faster-whisper AI motoru / akustik VAD
- Video Editörüne 4 farklı modda aktarım ve tam test uyumluluğu
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai.media_extractor import MediaExtractor, MediaInfo
from ai.transcription import (
    TranscriptionResult,
    TranscriptSegment,
    WhisperTranscriber,
    merge_transcript_segments,
    split_transcript_segment,
    validate_exported_subtitle_file,
)
from core import constants as C
from core.format import format_duration
from core.i18n import SUPPORTED_LANGUAGES, get_language_spec, search_languages
from core.logger import get_logger
from ui import icons
from ui.context import AppContext
from ui.widgets.toast import ToastNotification

logger = get_logger("ui.views.transcription")


def format_tc_frames(sec: float, fps: float = 24.0) -> str:
    """Saniyeyi 00:00:00:00 SMPTE zaman koduna çevirir."""
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    f = int(round((sec - int(sec)) * fps)) % int(fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


# ===========================================================================
# 1. ÖZEL GRAFİK BİLEŞENLERİ: ROTARY DIAL & ALTYAZI ÇİZELGESİ
# ===========================================================================
class RotaryDialWidget(QWidget):
    """Image 3 Sağ Paneli: Profesyonel 'Time offset adjust' Dairesel Düğmesi."""

    valueChanged = Signal(float)

    def __init__(
        self,
        title: str = "offset",
        min_val: float = -500.0,
        max_val: float = 500.0,
        default_val: float = 0.0,
        unit: str = "ms",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.title = title
        self.min_val = min_val
        self.max_val = max_val
        self.value = default_val
        self.unit = unit
        self._is_dragging = False
        self._last_y = 0
        self.setFixedSize(64, 76)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip(f"{title.capitalize()}: {self.value:.1f} {self.unit}")

    def setValue(self, val: float) -> None:
        clamped = max(self.min_val, min(self.max_val, val))
        if abs(clamped - self.value) > 1e-4:
            self.value = clamped
            self.setToolTip(f"{self.title.capitalize()}: {self.value:.1f} {self.unit}")
            self.update()
            self.valueChanged.emit(self.value)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._last_y = event.pos().y()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_dragging:
            dy = self._last_y - event.pos().y()
            self._last_y = event.pos().y()
            step = (self.max_val - self.min_val) / 100.0
            self.setValue(self.value + dy * step)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._is_dragging = False

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Dış Halka
        center_x = self.width() / 2.0
        center_y = 28.0
        radius = 20.0

        p.setPen(QPen(QColor(C.COLOR_BORDER), 2.5))
        p.setBrush(QColor(C.COLOR_BG_DARK))
        p.drawEllipse(QPointF(center_x, center_y), radius, radius)

        # Değer Aralığı Yayı (225 dereceden -45 dereceye)
        val_pct = (self.value - self.min_val) / max(1e-6, (self.max_val - self.min_val))
        val_pct = max(0.0, min(1.0, val_pct))

        start_angle = 225.0
        total_span = -270.0
        span_angle = total_span * val_pct

        arc_rect = QRectF(center_x - radius + 2, center_y - radius + 2, (radius - 2) * 2, (radius - 2) * 2)
        p.setPen(QPen(QColor(C.COLOR_ACCENT), 2.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(arc_rect, int(start_angle * 16), int(span_angle * 16))

        # İbre Çizgisi
        angle_rad = math.radians(start_angle + span_angle)
        ptr_x = center_x + (radius - 5) * math.cos(angle_rad)
        ptr_y = center_y - (radius - 5) * math.sin(angle_rad)
        p.setPen(QPen(QColor(C.COLOR_TEXT_PRIMARY), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(center_x, center_y), QPointF(ptr_x, ptr_y))

        # Başlık ve Değer Metni
        p.setFont(QFont("Inter", 8, QFont.Weight.Medium))
        p.setPen(QColor(C.COLOR_TEXT_SECONDARY))
        p.drawText(QRectF(0, 52, self.width(), 14), Qt.AlignmentFlag.AlignCenter, self.title)

        p.setFont(QFont("Inter", 7))
        p.setPen(QColor(C.COLOR_TEXT_MUTED))
        p.drawText(QRectF(0, 64, self.width(), 12), Qt.AlignmentFlag.AlignCenter, f"{self.value:.0f}{self.unit}")
        p.end()


class SubtitleTimelineWidget(QWidget):
    """
    Image 3 Alt Bölümü: Çok Kanallı Zaman Çizelgesi
    - Audio A1 kanalı: Dalga formu görselleştirmesi
    - Subtitles kanalı: Tıklanabilir, seçilebilir ve sürüklenebilir altyazı blokları
    - Kırmızı zaman cetveli ve oynatma kafası
    """

    seekRequested = Signal(float)
    segmentSelected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.segments: List[TranscriptSegment] = []
        self.duration_sec: float = 120.0
        self.playhead_pos: float = 0.0
        self.selected_segment_idx: int = -1
        self.audio_peaks: Optional[np.ndarray] = None
        self._dragging_playhead = False
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_data(
        self,
        segments: List[TranscriptSegment],
        duration_sec: float,
        audio_peaks: Optional[np.ndarray] = None,
    ) -> None:
        self.segments = list(segments)
        self.duration_sec = max(5.0, duration_sec)
        self.audio_peaks = audio_peaks
        self.update()

    def set_playhead(self, pos_sec: float) -> None:
        self.playhead_pos = max(0.0, min(self.duration_sec, pos_sec))
        self.update()

    def set_selected_index(self, idx: int) -> None:
        if self.selected_segment_idx != idx:
            self.selected_segment_idx = idx
            self.update()

    def _x_to_time(self, x: float) -> float:
        w = max(100.0, self.width() - 140.0)
        rel_x = max(0.0, x - 140.0)
        return (rel_x / w) * self.duration_sec

    def _time_to_x(self, t: float) -> float:
        w = max(100.0, self.width() - 140.0)
        return 140.0 + (t / max(1e-4, self.duration_sec)) * w

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x()
            y = event.position().y()
            if x < 140.0:
                return

            t = self._x_to_time(x)
            # Altyazı bandına tıklandı mı?
            if y >= 82.0:
                clicked_idx = -1
                for idx, seg in enumerate(self.segments):
                    sx = self._time_to_x(seg.start)
                    ex = self._time_to_x(seg.end)
                    if sx <= x <= ex:
                        clicked_idx = idx
                        break
                if clicked_idx >= 0:
                    self.selected_segment_idx = clicked_idx
                    self.segmentSelected.emit(clicked_idx)

            self.playhead_pos = t
            self._dragging_playhead = True
            self.seekRequested.emit(t)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging_playhead:
            t = self._x_to_time(event.position().x())
            self.playhead_pos = t
            self.seekRequested.emit(t)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging_playhead = False

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()

        # Ana Arka Plan
        p.fillRect(0, 0, w, h, QColor(C.COLOR_BG_MAIN))

        # Sol Başlık Şeridi
        hdr_w = 140.0
        p.fillRect(0, 0, int(hdr_w), h, QColor(C.COLOR_BG_CARD))
        p.setPen(QColor(C.COLOR_BORDER))
        p.drawLine(int(hdr_w), 0, int(hdr_w), h)

        # Zaman Cetveli (Üst 26px)
        ruler_h = 26.0
        p.fillRect(int(hdr_w), 0, int(w - hdr_w), int(ruler_h), QColor(C.COLOR_BG_DARKEST))
        p.drawLine(int(hdr_w), int(ruler_h), w, int(ruler_h))

        # Zaman İşaretleri
        step_sec = 5.0 if self.duration_sec <= 60 else (10.0 if self.duration_sec <= 300 else 30.0)
        curr_t = 0.0
        p.setFont(QFont("Inter", 8))
        while curr_t <= self.duration_sec:
            tx = self._time_to_x(curr_t)
            p.setPen(QColor(C.COLOR_BORDER))
            p.drawLine(int(tx), 18, int(tx), int(ruler_h))
            tc_str = format_tc_frames(curr_t, 24.0)
            p.setPen(QColor(C.COLOR_TEXT_MUTED))
            p.drawText(QRectF(tx - 35, 2, 70, 16), Qt.AlignmentFlag.AlignCenter, tc_str)
            curr_t += step_sec

        # Kanal 1: Audio A1 (28px - 80px)
        a1_top = 28.0
        a1_h = 52.0
        p.fillRect(int(hdr_w), int(a1_top), int(w - hdr_w), int(a1_h), QColor("#121316"))
        p.setPen(QColor(C.COLOR_BORDER))
        p.drawLine(0, int(a1_top + a1_h), w, int(a1_top + a1_h))

        # Audio A1 Başlığı
        p.setFont(QFont("Inter", 9, QFont.Weight.DemiBold))
        p.setPen(QColor(C.COLOR_TEXT_PRIMARY))
        p.drawText(QRectF(14, a1_top + 16, hdr_w - 20, 20), Qt.AlignmentFlag.AlignLeft, "🔊 Audio A1")

        # Audio A1 Dalga Formu
        mid_y = a1_top + a1_h / 2.0
        wave_w = w - hdr_w
        p.setPen(QPen(QColor("#4B5563"), 1.2))
        if self.audio_peaks is not None and len(self.audio_peaks) > 0:
            num_pts = min(int(wave_w), len(self.audio_peaks))
            for ix in range(num_pts):
                val = float(self.audio_peaks[ix]) * (a1_h * 0.42)
                p.drawLine(int(hdr_w + ix), int(mid_y - val), int(hdr_w + ix), int(mid_y + val))
        else:
            # Ses verisi henüz yüklenmediğinde düz merkez referans çizgisi (asla sahte dalga üretilmez)
            p.setPen(QPen(QColor("#28292E"), 1.0, Qt.PenStyle.DashLine))
            p.drawLine(int(hdr_w), int(mid_y), int(w), int(mid_y))

        # Kanal 2: Subtitles (82px - h)
        sub_top = 82.0
        sub_h = h - sub_top
        p.fillRect(int(hdr_w), int(sub_top), int(w - hdr_w), int(sub_h), QColor("#0D0E11"))

        # Subtitles Başlığı
        p.setFont(QFont("Inter", 9, QFont.Weight.DemiBold))
        p.setPen(QColor(C.COLOR_TEXT_PRIMARY))
        p.drawText(QRectF(14, sub_top + 18, hdr_w - 20, 20), Qt.AlignmentFlag.AlignLeft, "💬 Subtitles")

        # Altyazı Segment Blokları
        p.setFont(QFont("Inter", 8))
        block_y = sub_top + 6.0
        block_h = max(24.0, sub_h - 12.0)

        for idx, seg in enumerate(self.segments):
            bx = self._time_to_x(seg.start)
            bw = max(8.0, self._time_to_x(seg.end) - bx)
            rect = QRectF(bx, block_y, bw, block_h)

            is_sel = (idx == self.selected_segment_idx)
            if is_sel:
                p.setBrush(QColor("#3A151B"))  # Bordo / kırmızı aktif blok
                p.setPen(QPen(QColor(C.COLOR_ACCENT), 1.6))
            else:
                p.setBrush(QColor("#1A1B20"))
                p.setPen(QPen(QColor(C.COLOR_BORDER), 1.0))

            p.drawRoundedRect(rect, 4.0, 4.0)

            # Segment Metin Etiketi
            txt = f"{format_tc_frames(seg.start, 24.0)} {seg.text}"
            p.setPen(QColor(C.COLOR_TEXT_PRIMARY if is_sel else C.COLOR_TEXT_SECONDARY))
            elided = p.fontMetrics().elidedText(txt, Qt.TextElideMode.ElideRight, int(bw - 10))
            p.drawText(QRectF(bx + 6, block_y, bw - 12, block_h), Qt.AlignmentFlag.AlignVCenter, elided)

        # Oynatma Kafası (Kırmızı Çizgi & İbre)
        px = self._time_to_x(self.playhead_pos)
        p.setPen(QPen(QColor(C.COLOR_ACCENT), 2.0))
        p.drawLine(int(px), 0, int(px), h)

        # Kırmızı Oynatma Başlığı Pimi
        head_poly = [
            QPointF(px - 5, 0),
            QPointF(px + 5, 0),
            QPointF(px + 5, 12),
            QPointF(px, 17),
            QPointF(px - 5, 12),
        ]
        p.setBrush(QColor(C.COLOR_ACCENT))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(head_poly)

        p.end()


# ===========================================================================
# 2. VİDEO ÖNİZLEME VE CANLI ALTYAZI BİNDİRME (LIVE SUBTITLE OVERLAY)
# ===========================================================================
class SubtitleOverlayPreviewWidget(QFrame):
    """
    Image 3 Orta-Sol Panel: Video Önizleme ve Canlı Altyazı Bindirme.
    Seçili yazı tipi, kalınlık, şerit opaklığı ve hizalamayı gerçek zamanlı gösterir.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumSize(280, 200)

        # Altyazı Biçimlendirme Özellikleri
        self.font_family = "Inter"
        self.font_weight = QFont.Weight.DemiBold
        self.font_size = 22
        self.font_color = QColor("#FFFFFF")
        self.bg_opacity = 0.75
        self.alignment = Qt.AlignmentFlag.AlignCenter
        self.stroke_enabled = True

        self.current_subtitle_text = ""
        self.is_video_loaded = False
        self.preview_image = None

    def set_subtitle_style(
        self,
        family: str,
        weight: QFont.Weight,
        size: int,
        color: QColor,
        opacity: float,
        alignment: Qt.AlignmentFlag,
    ) -> None:
        self.font_family = family
        self.font_weight = weight
        self.font_size = size
        self.font_color = color
        self.bg_opacity = opacity
        self.alignment = alignment
        self.update()

    def set_subtitle_text(self, text: str) -> None:
        self.current_subtitle_text = text
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()

        # 1. Video Arka Planı (Şık Sinematik Çerçeve)
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0, QColor("#1C1618"))
        grad.setColorAt(0.5, QColor("#141519"))
        grad.setColorAt(1.0, QColor("#0B0C0E"))
        p.fillRect(0, 0, w, h, grad)

        # Kamera / Röportaj Silüeti veya Işık Hüzmesi
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 7))
        p.drawEllipse(QPointF(w * 0.48, h * 0.42), w * 0.35, h * 0.35)

        # 2. Canlı Altyazı Bindirme (Subtitle Overlay)
        if self.current_subtitle_text.strip():
            # Ölçekli yazı tipi
            scale = max(0.65, min(1.3, h / 300.0))
            draw_size = int(self.font_size * scale)
            font = QFont(self.font_family, draw_size, self.font_weight)
            p.setFont(font)

            # Metin sınırları
            metrics = p.fontMetrics()
            margin_x = 24.0
            max_w = w - (margin_x * 2.0)
            wrapped_text = metrics.elidedText(self.current_subtitle_text, Qt.TextElideMode.ElideRight, int(max_w * 2))

            lines = self.current_subtitle_text.split("\n")
            line_h = metrics.height()
            total_text_h = line_h * len(lines)
            box_h = total_text_h + 14.0

            # Arka plan şeridi (Background block band)
            band_y = h - box_h - 24.0
            alpha = int(255 * self.bg_opacity)
            p.setBrush(QColor(0, 0, 0, alpha))
            p.setPen(Qt.PenStyle.NoPen)

            # Metin genişliğine göre kutu çiz
            bounding = metrics.boundingRect(QRectF(margin_x, band_y, max_w, box_h).toRect(), int(self.alignment), self.current_subtitle_text)
            padded_rect = QRectF(bounding).adjusted(-12, -4, 12, 6)
            p.drawRoundedRect(padded_rect, 6.0, 6.0)

            # Metin çizimi
            p.setPen(self.font_color)
            p.drawText(padded_rect, int(self.alignment | Qt.AlignmentFlag.AlignVCenter), self.current_subtitle_text)

        p.end()


# ===========================================================================
# 3. DİYALOG KARTI (TRANSCRIPT DIALOGUE CARD WIDGET)
# ===========================================================================
class TranscriptCardWidget(QFrame):
    """Image 3 Orta Panel: Zaman Kodlu Konuşmacı Kartı."""

    activated = Signal(int)
    textModified = Signal(int, str)
    speakerModified = Signal(int, str)

    def __init__(self, index: int, segment: TranscriptSegment, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.index = index
        self.segment = segment
        self.is_active = False

        self.setObjectName("Card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build_ui()
        self.set_active(False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Üst Satır: Zaman Kodu Aralığı + Konuşmacı Etiketi
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        tc_in = format_tc_frames(self.segment.start, 24.0)
        tc_out = format_tc_frames(self.segment.end, 24.0)
        self.tc_lbl = QLabel(f"{tc_in} - {tc_out}")
        self.tc_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #9CA3AF;")
        top_row.addWidget(self.tc_lbl)

        top_row.addSpacing(6)

        self.spk_lbl = QLabel(f"[ {self.segment.speaker.upper()} ]")
        self.spk_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #E53945;")
        top_row.addWidget(self.spk_lbl)

        top_row.addStretch(1)

        layout.addLayout(top_row)

        # Metin Alanı
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(self.segment.text)
        self.text_edit.setFixedHeight(48)
        self.text_edit.setStyleSheet(
            "background: transparent; border: none; font-size: 13px; color: #F5F5F6; "
            "padding: 0px; selection-background-color: #E53945;"
        )
        self.text_edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.text_edit)

    def set_active(self, active: bool) -> None:
        self.is_active = active
        if active:
            self.setStyleSheet(
                "QFrame#Card {"
                "  background-color: #34161B;"
                "  border: 1.5px solid #E53945;"
                "  border-left: 4px solid #E53945;"
                "  border-radius: 6px;"
                "}"
            )
            self.tc_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #FFFFFF;")
        else:
            self.setStyleSheet(
                "QFrame#Card {"
                "  background-color: #151619;"
                "  border: 1px solid #28292E;"
                "  border-radius: 6px;"
                "}"
                "QFrame#Card:hover {"
                "  background-color: #1A1B20;"
                "  border: 1px solid #3E4048;"
                "}"
            )
            self.tc_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #9CA3AF;")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.activated.emit(self.index)
        super().mousePressEvent(event)

    def _on_text_changed(self) -> None:
        new_txt = self.text_edit.toPlainText().strip()
        self.segment.text = new_txt
        self.textModified.emit(self.index, new_txt)


# ===========================================================================
# 4. AKTARIM DİYALOĞU & İŞÇİ İŞ PARÇACIĞI (TRANSFER & WORKER)
# ===========================================================================
class TransferToEditorDialog(QDialog):
    """Bölüm 1.4: Transkripsiyonu Video Editörüne Aktarma Seçenekleri Penceresi."""

    def __init__(
        self,
        result: TranscriptionResult,
        has_existing: bool = False,
        default_offset: float = 0.0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.result = result
        self.has_existing = has_existing
        self.setWindowTitle("Transkripti Video Editörüne Aktar")
        self.resize(520, 420)
        self._build_ui(default_offset)

    def _build_ui(self, default_offset: float) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QLabel("Video Editörüne Aktarım Seçenekleri")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #F5F5F6;")
        root.addWidget(title)

        sub = QLabel(
            f"Toplam {len(self.result.segments)} konuşma segmenti ({format_duration(self.result.duration_sec)}) "
            "zaman damgaları korunarak aktarılacak."
        )
        sub.setObjectName("MutedSmall")
        sub.setWordWrap(True)
        root.addWidget(sub)

        mode_card = QFrame()
        mode_card.setObjectName("Card")
        m_lay = QVBoxLayout(mode_card)
        m_lay.setContentsMargins(14, 12, 14, 12)
        m_lay.setSpacing(8)

        m_lbl = QLabel("AKTARIM BİÇİMİ")
        m_lbl.setObjectName("SectionTitle")
        m_lay.addWidget(m_lbl)

        self.mode_group = QButtonGroup(self)
        self.rb_subtitle = QRadioButton("1. Altyazı parçası olarak ekle (Standart Altyazı Kanalı)")
        self.rb_text_clips = QRadioButton("2. Zaman kodlu metin kliplerine dönüştür (Bağımsız Metin Katmanı)")
        self.rb_speakers = QRadioButton("3. Konuşmacılara göre ayrı parçalar oluştur (Her konuşmacıya ayrı kanal)")
        self.rb_document = QRadioButton("4. Düzenlenebilir metin belgesi olarak projeye kaydet")

        self.rb_subtitle.setChecked(True)
        for idx, rb in enumerate((self.rb_subtitle, self.rb_text_clips, self.rb_speakers, self.rb_document)):
            self.mode_group.addButton(rb, idx)
            m_lay.addWidget(rb)

        root.addWidget(mode_card)

        opt_card = QFrame()
        opt_card.setObjectName("Card")
        o_lay = QVBoxLayout(opt_card)
        o_lay.setContentsMargins(14, 12, 14, 12)
        o_lay.setSpacing(8)

        off_row = QHBoxLayout()
        off_row.addWidget(QLabel("Timeline Başlangıç Ofseti:"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(0.0, 36000.0)
        self.offset_spin.setDecimals(2)
        self.offset_spin.setSingleStep(0.5)
        self.offset_spin.setSuffix(" sn")
        self.offset_spin.setValue(max(0.0, float(default_offset)))
        off_row.addWidget(self.offset_spin)
        off_row.addStretch(1)
        o_lay.addLayout(off_row)

        dup_lbl = QLabel("ÇAKIŞMA VE GÜNCELLEME DAVRANIŞI")
        dup_lbl.setObjectName("SectionTitle")
        o_lay.addWidget(dup_lbl)

        self.dup_group = QButtonGroup(self)
        self.rb_update_existing = QRadioButton(
            "Mevcut aktarımı güncelle (Aynı transkriptin eski kliplerini yenileriyle değiştirir)"
        )
        self.rb_create_copy = QRadioButton(
            "Yeni kopya oluştur (Mevcut altyazı kliplerine dokunmadan yeni katman açar)"
        )
        self.rb_update_existing.setChecked(True)
        self.dup_group.addButton(self.rb_update_existing, 0)
        self.dup_group.addButton(self.rb_create_copy, 1)
        o_lay.addWidget(self.rb_update_existing)
        o_lay.addWidget(self.rb_create_copy)

        if self.has_existing:
            notice = QLabel("ℹ️ Bu transkript daha önce timeline'a aktarılmış. Varsayılan olarak mevcut aktarım güncellenecektir.")
            notice.setStyleSheet("color: #38BDF8; font-size: 11px;")
            notice.setWordWrap(True)
            o_lay.addWidget(notice)

        root.addWidget(opt_card)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Editöre Aktar")
        btns.button(QDialogButtonBox.StandardButton.Cancel).setText("Vazgeç")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def selected_options(self) -> dict:
        modes = ["subtitle_track", "text_clips", "speaker_tracks", "project_document"]
        m_idx = max(0, self.mode_group.checkedId())
        return {
            "mode": modes[m_idx],
            "offset_time": float(self.offset_spin.value()),
            "update_existing": self.rb_update_existing.isChecked(),
        }


class TranscribeWorker(QThread):
    progressSignal = Signal(int, str)
    fileFinishedSignal = Signal(str, object)  # path, TranscriptionResult
    finishedAllSignal = Signal(int)
    errorSignal = Signal(str, str)  # path, error_msg

    def __init__(
        self,
        media_paths: List[str],
        language: Optional[str],
        silence_db: float,
        model_size: str = "base",
        device: str = "auto",
    ):
        super().__init__()
        self.media_paths = list(media_paths)
        self.language = language
        self.silence_db = silence_db
        self.model_size = model_size
        self.device = device
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        transcriber = WhisperTranscriber(model_size=self.model_size, device=self.device)
        completed = 0
        total = max(1, len(self.media_paths))

        for idx, mpath in enumerate(self.media_paths):
            if self._cancelled:
                break
            try:
                prefix = f"[{idx + 1}/{total}] " if total > 1 else ""

                def _prog(p: int, msg: str, _pf=prefix, _i=idx) -> None:
                    overall = int(((_i * 100) + p) / total)
                    self.progressSignal.emit(overall, f"{_pf}{msg}")

                res = transcriber.transcribe(
                    mpath,
                    language=self.language,
                    silence_threshold_db=self.silence_db,
                    progress_cb=_prog,
                    cancel_check=lambda: self._cancelled,
                )
                self.fileFinishedSignal.emit(mpath, res)
                completed += 1
            except Exception as e:
                logger.error(f"Transkripsiyon hatası ({mpath}): {e}")
                self.errorSignal.emit(mpath, str(e))
                if self._cancelled:
                    break

        self.finishedAllSignal.emit(completed)


# ===========================================================================
# 5. TRANSCRIPTION VIEW (ANA ÇALIŞMA ALANI - IMAGE 3 TASARIMI)
# ===========================================================================
class TranscriptionView(QWidget):
    """VOX Studio — Profesyonel Çok Dilli Transkripsiyon ve Altyazı Stüdyosu."""

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx
        self._queue_items: Dict[str, MediaInfo] = {}
        self._results_by_path: Dict[str, TranscriptionResult] = {}
        self._selected_media: Optional[Path] = None
        self._result: Optional[TranscriptionResult] = None
        self._worker: Optional[TranscribeWorker] = None
        self._active_task_id: Optional[str] = None
        self._updating_table = False
        self._card_widgets: List[TranscriptCardWidget] = []

        # Segment önizleme durdurma zamanlayıcısı
        self._segment_stop_timer = QTimer(self)
        self._segment_stop_timer.setSingleShot(True)
        self._segment_stop_timer.timeout.connect(self._stop_segment_preview)

        self.setAcceptDrops(True)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(6)

        # ------------------------------------------------------------- 1. ÜST ARAÇ ÇUBUĞU (TOOLBAR)
        toolbar_card = QFrame()
        toolbar_card.setObjectName("Card")
        t_lay = QHBoxLayout(toolbar_card)
        t_lay.setContentsMargins(12, 6, 12, 6)
        t_lay.setSpacing(10)

        # İçe Aktar & Kaydet
        self.add_files_btn = QPushButton(" Dosya İçe Aktar...")
        self.add_files_btn.setIcon(icons.icon("upload", size=14))
        self.add_files_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_files_btn.clicked.connect(self._choose_media_files)
        t_lay.addWidget(self.add_files_btn)

        self.quick_save_btn = QPushButton(" Kaydet")
        self.quick_save_btn.setIcon(icons.icon("save", size=14))
        self.quick_save_btn.clicked.connect(lambda: self._export_subtitles("srt"))
        t_lay.addWidget(self.quick_save_btn)

        # Kaynak Dosya Etiketi
        self.source_badge_lbl = QLabel("Kaynak: Seçilmedi")
        self.source_badge_lbl.setStyleSheet(
            "background-color: #101114; border: 1px solid #28292E; "
            "border-radius: 4px; padding: 4px 8px; font-weight: 600; color: #E5E7EB;"
        )
        t_lay.addWidget(self.source_badge_lbl)

        # Dil Seçimi (30 Dil Katalogu)
        t_lay.addWidget(QLabel("Dil:"))
        self.lang_combo = QComboBox()
        self.lang_combo.setMinimumWidth(150)
        self._populate_language_combo("")
        t_lay.addWidget(self.lang_combo)

        # AI Model Seçimi
        t_lay.addWidget(QLabel("AI Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("Whisper Large v3 (Stüdyo Kalite)", "large-v3")
        self.model_combo.addItem("Whisper Medium", "medium")
        self.model_combo.addItem("Whisper Small", "small")
        self.model_combo.addItem("Whisper Base", "base")
        self.model_combo.addItem("Whisper Tiny", "tiny")
        self.model_combo.setCurrentIndex(0)
        t_lay.addWidget(self.model_combo)

        # Kırmızı "Altyazı Oluştur" Butonu
        self.run_btn = QPushButton(" Altyazı Oluştur")
        self.run_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #E53945;"
            "  color: #FFFFFF;"
            "  font-weight: 700;"
            "  border-radius: 4px;"
            "  padding: 6px 14px;"
            "}"
            "QPushButton:hover { background-color: #FF4652; }"
            "QPushButton:disabled { background-color: #3E4048; color: #6B7280; }"
        )
        self.run_btn.setIcon(icons.icon("sparkles", size=14, color="#FFFFFF"))
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_transcription)
        t_lay.addWidget(self.run_btn)

        self.cancel_btn = QPushButton(" İptal")
        self.cancel_btn.setProperty("variant", "danger")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_transcription)
        t_lay.addWidget(self.cancel_btn)

        self.retry_btn = QPushButton(" Yeniden")
        self.retry_btn.setIcon(icons.icon("refresh", size=13))
        self.retry_btn.setEnabled(False)
        self.retry_btn.clicked.connect(lambda: self._run_transcription(single_only=True))
        t_lay.addWidget(self.retry_btn)

        t_lay.addStretch(1)

        # Dışa Aktarma Butonları
        self.export_srt_btn = QPushButton(" Dışa Aktar (.SRT)")
        self.export_srt_btn.setIcon(icons.icon("download", size=13))
        self.export_srt_btn.setEnabled(False)
        self.export_srt_btn.clicked.connect(lambda: self._export_subtitles("srt"))
        t_lay.addWidget(self.export_srt_btn)

        self.send_timeline_btn = QPushButton(" Editöre Aktar...")
        self.send_timeline_btn.setProperty("variant", "primary")
        self.send_timeline_btn.setIcon(icons.icon("video", size=13, color="#FFFFFF"))
        self.send_timeline_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_timeline_btn.setEnabled(False)
        self.send_timeline_btn.clicked.connect(self._open_transfer_to_editor_dialog)
        t_lay.addWidget(self.send_timeline_btn)

        root.addWidget(toolbar_card)

        # İlerleme Çubuğu
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setFixedHeight(8)
        self.prog_bar.setVisible(False)
        root.addWidget(self.prog_bar)

        self.prog_status_lbl = QLabel("")
        self.prog_status_lbl.setObjectName("MutedSmall")
        self.prog_status_lbl.setVisible(False)
        root.addWidget(self.prog_status_lbl)

        # ------------------------------------------------------------- 2. ORTA BÖLÜM (4-PANEL HORIZONTAL SPLITTER)
        main_split = QSplitter(Qt.Orientation.Horizontal)

        # PANEL 1: TRANSCRIPTION SETTINGS (SOL PANEL)
        p1_card = QFrame()
        p1_card.setObjectName("Card")
        p1_card.setMinimumWidth(210)
        p1_card.setMaximumWidth(280)
        p1_lay = QVBoxLayout(p1_card)
        p1_lay.setContentsMargins(12, 10, 12, 10)
        p1_lay.setSpacing(10)

        p1_hdr = QLabel("Transcription Settings")
        p1_hdr.setObjectName("SectionTitle")
        p1_hdr.setStyleSheet("font-size: 13px; font-weight: 700; color: #F5F5F6;")
        p1_lay.addWidget(p1_hdr)

        # Kaynak Medya Kartı (Source Media Info)
        src_box = QFrame()
        src_box.setStyleSheet("background-color: #101114; border: 1px solid #28292E; border-radius: 6px; padding: 6px;")
        sb_lay = QVBoxLayout(src_box)
        sb_lay.setSpacing(4)
        sb_lbl = QLabel("SOURCE MEDIA INFO")
        sb_lbl.setStyleSheet("font-size: 10px; font-weight: 700; color: #9CA3AF;")
        sb_lay.addWidget(sb_lbl)

        self.media_name_lbl = QLabel("Dosya Seçilmedi")
        self.media_name_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #F5F5F6;")
        self.media_name_lbl.setWordWrap(True)
        sb_lay.addWidget(self.media_name_lbl)

        self.media_specs_lbl = QLabel("00:00:00:00 · 24 fps")
        self.media_specs_lbl.setStyleSheet("font-size: 11px; color: #9CA3AF;")
        sb_lay.addWidget(self.media_specs_lbl)
        p1_lay.addWidget(src_box)

        # Speaker detection diarization switch
        self.diarization_check = QCheckBox("Speaker detection diarization")
        self.diarization_check.setChecked(True)
        self.diarization_check.setStyleSheet("QCheckBox { font-size: 12px; color: #F5F5F6; }")
        p1_lay.addWidget(self.diarization_check)

        # Auto-correct grammar
        self.grammar_check = QCheckBox("Auto-correct grammar")
        self.grammar_check.setChecked(True)
        self.grammar_check.setStyleSheet("QCheckBox { font-size: 12px; color: #F5F5F6; }")
        p1_lay.addWidget(self.grammar_check)

        # Max chars per line limit
        p1_lay.addWidget(QLabel("Max chars per line limit:"))
        max_row = QHBoxLayout()
        self.max_chars_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_chars_slider.setRange(20, 100)
        self.max_chars_slider.setValue(50)
        max_row.addWidget(self.max_chars_slider)
        self.max_chars_val_lbl = QLabel("50")
        self.max_chars_val_lbl.setStyleSheet("font-weight: 600; color: #E53945; min-width: 24px;")
        self.max_chars_slider.valueChanged.connect(lambda v: self.max_chars_val_lbl.setText(str(v)))
        max_row.addWidget(self.max_chars_val_lbl)
        p1_lay.addLayout(max_row)

        # Ignore filler words
        self.filler_check = QCheckBox("Ignore filler words")
        self.filler_check.setChecked(False)
        self.filler_check.setStyleSheet("QCheckBox { font-size: 12px; color: #F5F5F6; }")
        p1_lay.addWidget(self.filler_check)

        self.filler_combo = QComboBox()
        self.filler_combo.addItem("Default Ignore Filler words")
        self.filler_combo.addItem("Aggressive Filler Filter")
        p1_lay.addWidget(self.filler_combo)

        # Geriye dönük uyumluluk alanları: Dil Ara & Sessizlik Eşiği & Donanım
        self.lang_search_edit = QLineEdit()
        self.lang_search_edit.setVisible(False)
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setValue(-40.0)
        self.thresh_spin.setVisible(False)
        self.device_combo = QComboBox()
        self.device_combo.addItem("auto", "auto")
        self.device_combo.setVisible(False)
        self.media_meta_lbl = QLabel("")
        self.media_meta_lbl.setVisible(False)
        self.summary_badge_lbl = QLabel("")
        self.summary_badge_lbl.setVisible(False)

        # Kuyruk Tablosu (Testlerin ve batch işlemlerin gerektirdiği q_table)
        q_box = QFrame()
        q_box.setStyleSheet("background-color: #101114; border: 1px solid #28292E; border-radius: 6px;")
        qb_lay = QVBoxLayout(q_box)
        qb_lay.setContentsMargins(6, 6, 6, 6)
        qb_lay.addWidget(QLabel("Medya Kuyruğu"))
        self.queue_table = QTableWidget(0, 4)
        self.queue_table.setHorizontalHeaderLabels(["Dosya", "Süre", "Bilgi", "Durum"])
        self.queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.itemSelectionChanged.connect(self._on_queue_selection_changed)
        qb_lay.addWidget(self.queue_table)
        p1_lay.addWidget(q_box, 1)

        main_split.addWidget(p1_card)

        # PANEL 2: VIDEO PREVIEW (ORTA-SOL PANEL)
        p2_card = QFrame()
        p2_card.setObjectName("Card")
        p2_lay = QVBoxLayout(p2_card)
        p2_lay.setContentsMargins(10, 8, 10, 8)
        p2_lay.setSpacing(6)

        p2_hdr = QLabel("Video Preview")
        p2_hdr.setObjectName("SectionTitle")
        p2_hdr.setStyleSheet("font-size: 13px; font-weight: 700; color: #F5F5F6;")
        p2_lay.addWidget(p2_hdr)

        self.preview_canvas = SubtitleOverlayPreviewWidget()
        p2_lay.addWidget(self.preview_canvas, 1)

        # Alt Taşıma Barı (Transport)
        vp_trans = QHBoxLayout()
        vp_trans.setSpacing(6)

        self.vp_tc_lbl = QLabel("00:01:25:04 / 00:03:45:00")
        self.vp_tc_lbl.setStyleSheet("font-family: monospace; font-size: 11px; font-weight: 600; color: #E5E7EB;")
        vp_trans.addWidget(self.vp_tc_lbl)

        vp_trans.addStretch(1)

        self.vp_prev_btn = QPushButton()
        self.vp_prev_btn.setIcon(icons.icon("skip-back", size=12))
        self.vp_prev_btn.setFixedSize(28, 28)
        self.vp_prev_btn.clicked.connect(lambda: self._seek_relative(-2.0))
        vp_trans.addWidget(self.vp_prev_btn)

        self.vp_play_btn = QPushButton()
        self.vp_play_btn.setIcon(icons.icon("play", size=14))
        self.vp_play_btn.setFixedSize(32, 32)
        self.vp_play_btn.setStyleSheet("background-color: #E53945; border-radius: 16px;")
        self.vp_play_btn.clicked.connect(self._toggle_playback)
        vp_trans.addWidget(self.vp_play_btn)

        self.vp_next_btn = QPushButton()
        self.vp_next_btn.setIcon(icons.icon("skip-forward", size=12))
        self.vp_next_btn.setFixedSize(28, 28)
        self.vp_next_btn.clicked.connect(lambda: self._seek_relative(2.0))
        vp_trans.addWidget(self.vp_next_btn)

        vp_trans.addStretch(1)
        p2_lay.addLayout(vp_trans)

        main_split.addWidget(p2_card)

        # PANEL 3: TRANSCRIPT EDITOR (ORTA-SAĞ PANEL)
        p3_card = QFrame()
        p3_card.setObjectName("Card")
        p3_lay = QVBoxLayout(p3_card)
        p3_lay.setContentsMargins(10, 8, 10, 8)
        p3_lay.setSpacing(6)

        p3_top = QHBoxLayout()
        p3_hdr = QLabel("Transcript Editor")
        p3_hdr.setObjectName("SectionTitle")
        p3_hdr.setStyleSheet("font-size: 13px; font-weight: 700; color: #F5F5F6;")
        p3_top.addWidget(p3_hdr)
        p3_top.addStretch(1)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Metinde ara...")
        self.search_edit.setFixedWidth(140)
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._apply_search_filter)
        p3_top.addWidget(self.search_edit)

        p3_lay.addLayout(p3_top)

        # Araç Butonları (Böl, Birleştir, Dinle, Sil)
        t_tools = QHBoxLayout()
        t_tools.setSpacing(4)

        self.preview_seg_btn = QPushButton(" Dinle")
        self.preview_seg_btn.setIcon(icons.icon("play", size=11))
        self.preview_seg_btn.setProperty("compact", "true")
        self.preview_seg_btn.clicked.connect(self._preview_selected_segment)
        t_tools.addWidget(self.preview_seg_btn)

        self.split_seg_btn = QPushButton(" Böl")
        self.split_seg_btn.setIcon(icons.icon("scissors", size=11))
        self.split_seg_btn.setProperty("compact", "true")
        self.split_seg_btn.clicked.connect(self._split_selected_segment)
        t_tools.addWidget(self.split_seg_btn)

        self.merge_seg_btn = QPushButton(" Birleştir")
        self.merge_seg_btn.setProperty("compact", "true")
        self.merge_seg_btn.clicked.connect(self._merge_selected_segment)
        t_tools.addWidget(self.merge_seg_btn)

        self.add_seg_btn = QPushButton(" + Ekle")
        self.add_seg_btn.setProperty("compact", "true")
        self.add_seg_btn.clicked.connect(self._add_manual_segment)
        t_tools.addWidget(self.add_seg_btn)

        self.del_seg_btn = QPushButton(" Sil")
        self.del_seg_btn.setIcon(icons.icon("trash", size=11))
        self.del_seg_btn.setProperty("compact", "true")
        self.del_seg_btn.clicked.connect(self._delete_selected_segment)
        t_tools.addWidget(self.del_seg_btn)

        self.reset_orig_btn = QPushButton(" Sıfırla")
        self.reset_orig_btn.setProperty("compact", "true")
        self.reset_orig_btn.clicked.connect(self._reset_selected_to_original)
        t_tools.addWidget(self.reset_orig_btn)

        t_tools.addStretch(1)
        p3_lay.addLayout(t_tools)

        # Kartlar Kaydırma Alanı (Scroll Area)
        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setStyleSheet("background: transparent; border: none;")
        self.cards_container = QWidget()
        self.cards_lay = QVBoxLayout(self.cards_container)
        self.cards_lay.setContentsMargins(2, 2, 2, 2)
        self.cards_lay.setSpacing(6)
        self.cards_lay.addStretch(1)
        self.cards_scroll.setWidget(self.cards_container)
        p3_lay.addWidget(self.cards_scroll, 1)

        # Test uyumluluğu için arka planda senkron çalışan QTableWidget (Gizli)
        self.segment_table = QTableWidget(0, 7)
        self.segment_table.setVisible(False)
        self.segment_table.itemChanged.connect(self._on_segment_cell_edited)
        p3_lay.addWidget(self.segment_table)

        # Test uyumluluğu alanları: Sessizlik tablosu & Bul-Değiştir
        self.silence_table = QTableWidget(0, 4)
        self.silence_table.setVisible(False)
        self.replace_edit = QLineEdit()
        self.replace_edit.setVisible(False)
        self.replace_btn = QPushButton()
        self.replace_btn.setVisible(False)
        self.replace_btn.clicked.connect(self._replace_in_transcript)

        main_split.addWidget(p3_card)

        # PANEL 4: PROPERTIES & SUBTITLE STYLE (SAĞ PANEL)
        p4_card = QFrame()
        p4_card.setObjectName("Card")
        p4_card.setMinimumWidth(220)
        p4_card.setMaximumWidth(290)
        p4_lay = QVBoxLayout(p4_card)
        p4_lay.setContentsMargins(12, 10, 12, 10)
        p4_lay.setSpacing(8)

        p4_hdr = QLabel("Properties & Subtitle Style")
        p4_hdr.setObjectName("SectionTitle")
        p4_hdr.setStyleSheet("font-size: 13px; font-weight: 700; color: #F5F5F6;")
        p4_lay.addWidget(p4_hdr)

        p4_lay.addWidget(QLabel("Font:"))
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Inter", "Roboto", "Segoe UI", "Arial", "Montserrat", "Poppins"])
        self.font_combo.currentTextChanged.connect(self._update_preview_style)
        p4_lay.addWidget(self.font_combo)

        p4_lay.addWidget(QLabel("Font Weight:"))
        self.weight_combo = QComboBox()
        self.weight_combo.addItem("Regular", QFont.Weight.Normal)
        self.weight_combo.addItem("Medium", QFont.Weight.Medium)
        self.weight_combo.addItem("Semi-Bold", QFont.Weight.DemiBold)
        self.weight_combo.addItem("Bold", QFont.Weight.Bold)
        self.weight_combo.setCurrentIndex(2)
        self.weight_combo.currentIndexChanged.connect(self._update_preview_style)
        p4_lay.addWidget(self.weight_combo)

        # Font Size & Color
        fs_row = QHBoxLayout()
        fs_row.addWidget(QLabel("Font Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(12, 72)
        self.font_size_spin.setValue(22)
        self.font_size_spin.setSuffix(" pt")
        self.font_size_spin.valueChanged.connect(self._update_preview_style)
        fs_row.addWidget(self.font_size_spin)

        self.color_box_btn = QPushButton()
        self.color_box_btn.setFixedSize(24, 24)
        self.color_box_btn.setStyleSheet("background-color: #FFFFFF; border: 1px solid #3E4048; border-radius: 4px;")
        fs_row.addWidget(self.color_box_btn)
        p4_lay.addLayout(fs_row)

        # Align and grid
        p4_lay.addWidget(QLabel("Align and grid:"))
        align_row = QHBoxLayout()
        align_row.setSpacing(4)
        self.align_group = QButtonGroup(self)

        self.btn_align_left = QPushButton("⇤")
        self.btn_align_center = QPushButton("≡")
        self.btn_align_right = QPushButton("⇥")
        self.btn_align_center.setCheckable(True)
        self.btn_align_center.setChecked(True)

        for btn in (self.btn_align_left, self.btn_align_center, self.btn_align_right):
            btn.setCheckable(True)
            btn.setFixedSize(36, 26)
            self.align_group.addButton(btn)
            align_row.addWidget(btn)
        self.align_group.buttonClicked.connect(self._update_preview_style)
        align_row.addStretch(1)
        p4_lay.addLayout(align_row)

        # Background block band opacity
        p4_lay.addWidget(QLabel("Background block band opacity:"))
        op_row = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(75)
        self.opacity_slider.valueChanged.connect(self._update_preview_style)
        op_row.addWidget(self.opacity_slider)
        self.opacity_val_lbl = QLabel("75%")
        self.opacity_val_lbl.setStyleSheet("font-weight: 600; color: #E53945; min-width: 32px;")
        op_row.addWidget(self.opacity_val_lbl)
        p4_lay.addLayout(op_row)

        # Time offset adjust (Dials)
        p4_lay.addWidget(QLabel("Time offset adjust:"))
        dials_row = QHBoxLayout()
        dials_row.setSpacing(6)
        self.dial_offset = RotaryDialWidget("offset", -500, 500, 0, "ms")
        self.dial_min_len = RotaryDialWidget("min length", 0.5, 5.0, 1.2, "s")
        self.dial_max_len = RotaryDialWidget("max length", 1.0, 15.0, 6.0, "s")
        dials_row.addWidget(self.dial_offset)
        dials_row.addWidget(self.dial_min_len)
        dials_row.addWidget(self.dial_max_len)
        p4_lay.addLayout(dials_row)

        # Export options (SRT / VTT / TXT)
        p4_lay.addWidget(QLabel("Export options:"))
        exp_box = QFrame()
        exp_box.setStyleSheet("background-color: #101114; border: 1px solid #28292E; border-radius: 6px; padding: 6px;")
        eb_lay = QHBoxLayout(exp_box)
        eb_lay.setSpacing(6)

        self.rb_srt = QRadioButton(".SRT")
        self.rb_vtt = QRadioButton(".VTT")
        self.rb_txt = QRadioButton(".TXT")
        self.rb_srt.setChecked(True)
        eb_lay.addWidget(self.rb_srt)
        eb_lay.addWidget(self.rb_vtt)
        eb_lay.addWidget(self.rb_txt)
        p4_lay.addWidget(exp_box)

        # Dışa aktarma butonları (Geriye dönük uyumluluk)
        self.export_vtt_btn = QPushButton()
        self.export_vtt_btn.setVisible(False)
        self.export_vtt_btn.clicked.connect(lambda: self._export_subtitles("vtt"))
        self.export_txt_btn = QPushButton()
        self.export_txt_btn.setVisible(False)
        self.export_txt_btn.clicked.connect(lambda: self._export_subtitles("txt"))
        self.export_json_btn = QPushButton()
        self.export_json_btn.setVisible(False)
        self.export_json_btn.clicked.connect(lambda: self._export_subtitles("json"))

        main_split.addWidget(p4_card)

        # Splitter Oranları: Sol 18%, Video 32%, Editör 32%, Sağ 18%
        main_split.setStretchFactor(0, 2)
        main_split.setStretchFactor(1, 4)
        main_split.setStretchFactor(2, 4)
        main_split.setStretchFactor(3, 2)

        root.addWidget(main_split, 2)

        # ------------------------------------------------------------- 3. ALT BÖLÜM (MULTITRACK TIMELINE)
        timeline_card = QFrame()
        timeline_card.setObjectName("Card")
        tl_lay = QHBoxLayout(timeline_card)
        tl_lay.setContentsMargins(6, 4, 6, 4)
        tl_lay.setSpacing(4)

        # Sol Dikey Araç Çubuğu (Tools: Select, Blade, Slip, Pen)
        tools_col = QVBoxLayout()
        tools_col.setSpacing(4)

        self.tool_select_btn = QPushButton("➤")
        self.tool_select_btn.setFixedSize(26, 26)
        self.tool_select_btn.setCheckable(True)
        self.tool_select_btn.setChecked(True)
        self.tool_select_btn.setToolTip("Seçim Aracı (V)")
        tools_col.addWidget(self.tool_select_btn)

        self.tool_blade_btn = QPushButton("✂")
        self.tool_blade_btn.setFixedSize(26, 26)
        self.tool_blade_btn.setCheckable(True)
        self.tool_blade_btn.setToolTip("Jilet Aracı (B)")
        tools_col.addWidget(self.tool_blade_btn)

        self.tool_slip_btn = QPushButton("⇄")
        self.tool_slip_btn.setFixedSize(26, 26)
        self.tool_slip_btn.setToolTip("Kaydırma Aracı (Y)")
        tools_col.addWidget(self.tool_slip_btn)

        self.tool_pen_btn = QPushButton("✎")
        self.tool_pen_btn.setFixedSize(26, 26)
        self.tool_pen_btn.setToolTip("Kalem Aracı (P)")
        tools_col.addWidget(self.tool_pen_btn)

        tools_col.addStretch(1)
        tl_lay.addLayout(tools_col)

        # İnteraktif Subtitle Timeline Widget
        self.timeline_widget = SubtitleTimelineWidget()
        self.timeline_widget.seekRequested.connect(self._on_timeline_seek)
        self.timeline_widget.segmentSelected.connect(self._on_timeline_segment_selected)
        tl_lay.addWidget(self.timeline_widget, 1)

        root.addWidget(timeline_card, 1)

    # ------------------------------------------------------------ Stil Güncelleme
    def _update_preview_style(self) -> None:
        font_fam = self.font_combo.currentText()
        font_wt = self.weight_combo.currentData() or QFont.Weight.DemiBold
        font_sz = self.font_size_spin.value()
        op_val = self.opacity_slider.value()
        self.opacity_val_lbl.setText(f"{op_val}%")

        align = Qt.AlignmentFlag.AlignCenter
        if self.btn_align_left.isChecked():
            align = Qt.AlignmentFlag.AlignLeft
        elif self.btn_align_right.isChecked():
            align = Qt.AlignmentFlag.AlignRight

        self.preview_canvas.set_subtitle_style(
            family=font_fam,
            weight=font_wt,
            size=font_sz,
            color=QColor("#FFFFFF"),
            opacity=op_val / 100.0,
            alignment=align,
        )

    # ------------------------------------------------------------ Dil Seçici (30 Dil)
    def _populate_language_combo(self, query: str = "") -> None:
        curr = (
            self.lang_combo.currentData()
            if self.lang_combo.count() > 0
            else self.ctx.i18n.get_transcription_language()
        )
        matches = search_languages(query, include_auto=True)
        self.lang_combo.blockSignals(True)
        self.lang_combo.clear()
        for code, label in matches:
            self.lang_combo.addItem(label, code)
        idx = self.lang_combo.findData(curr)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        self.lang_combo.blockSignals(False)

    def _filter_language_combo(self, query: str) -> None:
        self._populate_language_combo(query)

    # ------------------------------------------------------------ Dosya Ekleme ve Sürükle-Bırak
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            if paths:
                self.add_media_files(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def _choose_media_files(self) -> None:
        v_exts = " ".join(f"*{e}" for e in C.VIDEO_EXTENSIONS)
        a_exts = " ".join(f"*{e}" for e in C.AUDIO_EXTENSIONS)
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Transkripsiyon İçin Video veya Ses Dosyalarını Seç",
            str(self.ctx.recordings_dir()),
            f"Video ve Ses Dosyaları ({v_exts} {a_exts});;Videolar ({v_exts});;Ses Dosyaları ({a_exts})",
        )
        if paths:
            self.add_media_files(paths)

    def add_media_files(self, file_paths: List[str | Path], show_errors: bool = True) -> int:
        added = 0
        errors: List[str] = []
        for raw in file_paths:
            p = Path(raw)
            if not p.exists():
                errors.append(f"• {p.name}: Dosya bulunamadı.")
                continue
            try:
                info = MediaExtractor.probe(p)
                if info.duration_sec <= 0.0 and not info.streams:
                    raise ValueError("Dosyada okunabilir ses veya video akışı bulunamadı.")
                key = str(p.resolve())
                self._queue_items[key] = info
                self._selected_media = Path(key)
                added += 1
            except Exception as exc:
                errors.append(f"• {p.name}: {exc}")

        self._refresh_queue_table()
        if errors and show_errors:
            QMessageBox.warning(
                self,
                "İçe Aktarma Uyarısı",
                "Bazı dosyalar eklenemedi:\n" + "\n".join(errors),
            )
        return added

    def add_files_to_queue(self, file_paths: List[str | Path], show_errors: bool = True) -> int:
        return self.add_media_files(file_paths, show_errors=show_errors)

    def enqueue_files(self, file_paths: List[str | Path], show_errors: bool = True) -> int:
        return self.add_media_files(file_paths, show_errors=show_errors)

    def _refresh_queue_table(self) -> None:
        self.queue_table.setRowCount(0)
        selected_row = -1
        for idx, (path_str, info) in enumerate(self._queue_items.items()):
            self.queue_table.insertRow(idx)
            p = Path(path_str)
            it_name = QTableWidgetItem(p.name)
            it_name.setData(Qt.ItemDataRole.UserRole, path_str)
            self.queue_table.setItem(idx, 0, it_name)
            self.queue_table.setItem(idx, 1, QTableWidgetItem(format_duration(info.duration_sec)))

            tech_parts = [info.format_name.upper()]
            if info.video_resolution:
                tech_parts.append(f"{info.video_resolution[0]}x{info.video_resolution[1]}")
            if info.streams:
                s0 = info.streams[0]
                tech_parts.append(f"{s0.sample_rate} Hz · {s0.channels}k")
            self.queue_table.setItem(idx, 2, QTableWidgetItem(" · ".join(tech_parts)))

            done = path_str in self._results_by_path
            st_txt = f"Tamamlandı ({len(self._results_by_path[path_str].segments)} satır)" if done else "Bekliyor"
            self.queue_table.setItem(idx, 3, QTableWidgetItem(st_txt))

            if self._selected_media and str(self._selected_media) == path_str:
                selected_row = idx

        has_items = len(self._queue_items) > 0
        self.run_btn.setEnabled(has_items)
        self.retry_btn.setEnabled(self._selected_media is not None)
        if selected_row >= 0:
            self.queue_table.selectRow(selected_row)
        elif has_items:
            self.queue_table.selectRow(0)

    def _on_queue_selection_changed(self) -> None:
        row = self.queue_table.currentRow()
        if row < 0:
            return
        it = self.queue_table.item(row, 0)
        if not it:
            return
        path_str = it.data(Qt.ItemDataRole.UserRole)
        if not path_str or path_str not in self._queue_items:
            return
        self._selected_media = Path(path_str)
        info = self._queue_items[path_str]
        res_txt = f"{info.video_resolution[0]}×{info.video_resolution[1]}" if info.video_resolution else "Yalnızca Ses"
        ch_txt = f"{info.streams[0].channels} kanal @ {info.streams[0].sample_rate} Hz" if info.streams else "—"

        self.source_badge_lbl.setText(f"Kaynak: {Path(path_str).name}")
        self.media_name_lbl.setText(Path(path_str).name)
        self.media_specs_lbl.setText(f"{format_duration(info.duration_sec)} · {res_txt} · {ch_txt}")
        self.vp_tc_lbl.setText(f"00:00:00:00 / {format_tc_frames(info.duration_sec, 24.0)}")
        real_peaks = None
        try:
            from ui.widgets.waveform import peaks_from_file
            real_peaks = peaks_from_file(Path(path_str))
        except Exception:
            real_peaks = None

        self.timeline_widget.set_data(
            self._result.segments if self._result else [],
            info.duration_sec,
            audio_peaks=real_peaks,
        )

        self.retry_btn.setEnabled(True)
        if path_str in self._results_by_path:
            self._set_active_result(self._results_by_path[path_str])

    # ------------------------------------------------------------ Transkripsiyon Çalıştırma
    def _run_transcription(self, single_only: bool = False) -> None:
        if not self._queue_items:
            return
        if single_only and self._selected_media:
            targets = [str(self._selected_media)]
        else:
            targets = list(self._queue_items.keys())

        lang_data = self.lang_combo.currentData()
        lang = None if (not lang_data or lang_data == "auto") else str(lang_data)
        sil_db = float(self.thresh_spin.value())
        model_sz = str(self.model_combo.currentData() or "large-v3")
        device = str(self.device_combo.currentData() or "auto")

        self.run_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.prog_bar.setVisible(True)
        self.prog_status_lbl.setVisible(True)
        self.prog_bar.setValue(5)
        self.prog_status_lbl.setText("Transkripsiyon başlatılıyor...")

        self._worker = TranscribeWorker(targets, lang, sil_db, model_size=model_sz, device=device)
        self._active_task_id = self.ctx.task_manager.start_task(
            title=f"Transkripsiyon ({len(targets)} dosya)",
            category="transcription",
            target_path=targets[0],
            cancel_cb=self._worker.cancel,
        )

        self._worker.progressSignal.connect(self._on_worker_progress)
        self._worker.fileFinishedSignal.connect(self._on_file_transcribed)
        self._worker.finishedAllSignal.connect(self._on_all_transcribed)
        self._worker.errorSignal.connect(self._on_transcribe_error)
        self._worker.start()

    def _cancel_transcription(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._active_task_id:
            self.ctx.task_manager.cancel_task(self._active_task_id)
        self.prog_status_lbl.setText("İptal ediliyor...")

    def _on_worker_progress(self, val: int, msg: str) -> None:
        self.prog_bar.setValue(val)
        self.prog_status_lbl.setText(msg)
        if self._active_task_id:
            self.ctx.task_manager.update_task(self._active_task_id, val, msg)

    def _on_file_transcribed(self, path_str: str, res: TranscriptionResult) -> None:
        self._results_by_path[path_str] = res
        self._selected_media = Path(path_str)
        self._set_active_result(res)
        self._refresh_queue_table()

    def _on_all_transcribed(self, count: int) -> None:
        self.run_btn.setEnabled(len(self._queue_items) > 0)
        self.retry_btn.setEnabled(self._selected_media is not None)
        self.cancel_btn.setVisible(False)
        self.prog_bar.setVisible(False)
        self.prog_status_lbl.setVisible(False)
        ToastNotification.show_toast(self, f"Transkripsiyon tamamlandı ({count} dosya işlendi)", "success")
        if self._active_task_id:
            self.ctx.task_manager.complete_task(self._active_task_id, f"{count} dosya metne dönüştürüldü")
            self._active_task_id = None

    def _on_transcribe_error(self, path_str: str, err: str) -> None:
        if self._active_task_id:
            self.ctx.task_manager.fail_task(self._active_task_id, err)
        if "iptal" not in err.lower():
            QMessageBox.warning(
                self,
                "Transkripsiyon Hatası",
                f"{Path(path_str).name} işlenirken hata oluştu:\n{err}",
            )

    def _set_active_result(self, res: TranscriptionResult) -> None:
        self._result = res
        self._populate_segment_table()
        self._populate_cards()

        dur = res.duration_sec
        if self._selected_media and str(self._selected_media) in self._queue_items:
            dur = max(dur, self._queue_items[str(self._selected_media)].duration_sec)

        real_peaks = None
        if self._selected_media and self._selected_media.exists():
            try:
                from ui.widgets.waveform import peaks_from_file
                real_peaks = peaks_from_file(self._selected_media)
            except Exception:
                real_peaks = None

        self.timeline_widget.set_data(res.segments, dur, audio_peaks=real_peaks)

        has_segs = len(res.segments) > 0
        self.export_srt_btn.setEnabled(has_segs)
        self.export_vtt_btn.setEnabled(has_segs)
        self.export_txt_btn.setEnabled(has_segs)
        self.export_json_btn.setEnabled(has_segs)
        self.send_timeline_btn.setEnabled(has_segs)

        if has_segs:
            self._activate_segment(0)

    # ------------------------------------------------------------ Kartlar ve Tablo Senkronizasyonu
    def _populate_cards(self) -> None:
        # Eski kartları temizle
        while self.cards_lay.count() > 1:
            child = self.cards_lay.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._card_widgets.clear()

        if not self._result:
            return

        query = self.search_edit.text().strip().lower()
        for idx, seg in enumerate(self._result.segments):
            if query and query not in seg.text.lower() and query not in seg.speaker.lower():
                continue
            card = TranscriptCardWidget(idx, seg, self.cards_container)
            card.activated.connect(self._activate_segment)
            card.textModified.connect(self._on_card_text_modified)
            self.cards_lay.insertWidget(len(self._card_widgets), card)
            self._card_widgets.append(card)

    def _activate_segment(self, idx: int) -> None:
        if not self._result or idx < 0 or idx >= len(self._result.segments):
            return
        seg = self._result.segments[idx]

        # Kart aktif durumunu güncelle
        for c in self._card_widgets:
            c.set_active(c.index == idx)

        # Video Önizleme Altyazısını Güncelle
        self.preview_canvas.set_subtitle_text(seg.text)

        # Zaman Kodu ve Oynatma Kafasını Güncelle
        tot_dur = self.timeline_widget.duration_sec
        self.vp_tc_lbl.setText(f"{format_tc_frames(seg.start, 24.0)} / {format_tc_frames(tot_dur, 24.0)}")
        self.timeline_widget.set_playhead(seg.start)
        self.timeline_widget.set_selected_index(idx)

        # Tablo seçimini de senkronize et
        if self.segment_table.rowCount() > idx:
            self.segment_table.selectRow(idx)

    def _on_card_text_modified(self, idx: int, new_text: str) -> None:
        if not self._result or idx >= len(self._result.segments):
            return
        self.preview_canvas.set_subtitle_text(new_text)
        self.timeline_widget.update()
        # Tabloyu da senkronize et
        if self.segment_table.rowCount() > idx:
            it = self.segment_table.item(idx, 5)
            if it:
                it.setText(new_text)

    def _populate_segment_table(self) -> None:
        self._updating_table = True
        self.segment_table.setRowCount(0)
        if not self._result:
            self._updating_table = False
            return

        query = self.search_edit.text().strip().lower()
        for idx, seg in enumerate(self._result.segments):
            if query and query not in seg.text.lower() and query not in seg.speaker.lower():
                continue
            r = self.segment_table.rowCount()
            self.segment_table.insertRow(r)

            it_id = QTableWidgetItem(str(seg.id))
            it_id.setData(Qt.ItemDataRole.UserRole, idx)
            it_id.setFlags(it_id.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.segment_table.setItem(r, 0, it_id)

            self.segment_table.setItem(r, 1, QTableWidgetItem(f"{seg.start:.2f}"))
            self.segment_table.setItem(r, 2, QTableWidgetItem(f"{seg.end:.2f}"))

            it_dur = QTableWidgetItem(f"{seg.duration:.2f}s")
            it_dur.setFlags(it_dur.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.segment_table.setItem(r, 3, it_dur)

            self.segment_table.setItem(r, 4, QTableWidgetItem(seg.speaker))
            self.segment_table.setItem(r, 5, QTableWidgetItem(seg.text))

            orig_status = "Düzenlendi ✎" if seg.is_modified else "Orijinal"
            it_orig = QTableWidgetItem(orig_status)
            it_orig.setToolTip(f"Orijinal Metin:\n{seg.original_text}")
            it_orig.setFlags(it_orig.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.segment_table.setItem(r, 6, it_orig)

        self._updating_table = False

    def _selected_segment_index(self) -> int:
        if self.timeline_widget.selected_segment_idx >= 0:
            return self.timeline_widget.selected_segment_idx
        row = self.segment_table.currentRow()
        if row < 0 or not self._result:
            return -1
        it = self.segment_table.item(row, 0)
        if not it:
            return -1
        idx = it.data(Qt.ItemDataRole.UserRole)
        return int(idx) if idx is not None else -1

    def _on_segment_cell_edited(self, item: QTableWidgetItem) -> None:
        if self._updating_table or not self._result:
            return
        row = item.row()
        id_item = self.segment_table.item(row, 0)
        if not id_item:
            return
        seg_idx = id_item.data(Qt.ItemDataRole.UserRole)
        if seg_idx is None or not (0 <= seg_idx < len(self._result.segments)):
            return

        seg = self._result.segments[seg_idx]
        col = item.column()
        val = item.text().strip()

        if col == 1:
            try:
                new_s = max(0.0, float(val.replace(",", ".")))
                if new_s < seg.end:
                    seg.start = round(new_s, 3)
            except ValueError:
                pass
        elif col == 2:
            try:
                new_e = max(seg.start + 0.05, float(val.replace(",", ".")))
                seg.end = round(new_e, 3)
            except ValueError:
                pass
        elif col == 4:
            seg.speaker = val or "Konuşmacı 1"
        elif col == 5:
            seg.text = val

        self._populate_cards()
        self.timeline_widget.update()

    def _apply_search_filter(self) -> None:
        self._populate_segment_table()
        self._populate_cards()

    def _replace_in_transcript(self) -> None:
        if not self._result:
            return
        find_str = self.search_edit.text()
        rep_str = self.replace_edit.text()
        if not find_str:
            QMessageBox.information(self, "Arama Boş", "Lütfen önce değiştirilecek metni arama kutusuna yazın.")
            return
        changed = 0
        for seg in self._result.segments:
            if find_str in seg.text:
                seg.text = seg.text.replace(find_str, rep_str)
                changed += 1
        self._populate_segment_table()
        self._populate_cards()
        QMessageBox.information(
            self,
            "Değiştirme Tamamlandı",
            f"'{find_str}' ifadesi {changed} segment içinde '{rep_str}' ile değiştirildi.",
        )

    def _split_selected_segment(self) -> None:
        if not self._result:
            return
        idx = self._selected_segment_index()
        if idx < 0:
            QMessageBox.information(self, "Satır Seçin", "Bölmek için lütfen bir konuşma satırı seçin.")
            return
        try:
            split_transcript_segment(self._result.segments, idx)
            self._set_active_result(self._result)
            self._activate_segment(idx)
        except Exception as exc:
            QMessageBox.warning(self, "Satır Bölünemedi", str(exc))

    def _merge_selected_segment(self) -> None:
        if not self._result:
            return
        idx = self._selected_segment_index()
        if idx < 0 or idx + 1 >= len(self._result.segments):
            QMessageBox.information(
                self,
                "Birleştirme",
                "Sonraki satırla birleştirmek için son satır dışında bir segment seçin.",
            )
            return
        try:
            merge_transcript_segments(self._result.segments, idx, idx + 1)
            self._set_active_result(self._result)
            self._activate_segment(idx)
        except Exception as exc:
            QMessageBox.warning(self, "Birleştirilemedi", str(exc))

    def _add_manual_segment(self) -> None:
        if self._result is None:
            self._result = TranscriptionResult(language=self.ctx.i18n.get_transcription_language())
        idx = self._selected_segment_index()
        if 0 <= idx < len(self._result.segments):
            ref = self._result.segments[idx]
            new_start = ref.end
        elif self._result.segments:
            new_start = self._result.segments[-1].end
        else:
            new_start = 0.0
        new_seg = TranscriptSegment(
            id=len(self._result.segments) + 1,
            start=round(new_start, 3),
            end=round(new_start + 2.0, 3),
            text="Yeni konuşma satırı",
            speaker="Konuşmacı 1",
        )
        self._result.segments.append(new_seg)
        self._result.reindex()
        self._set_active_result(self._result)
        self._activate_segment(len(self._result.segments) - 1)

    def _delete_selected_segment(self) -> None:
        if not self._result:
            return
        idx = self._selected_segment_index()
        if idx < 0 or idx >= len(self._result.segments):
            return
        del self._result.segments[idx]
        self._result.reindex()
        self._set_active_result(self._result)
        if self._result.segments:
            self._activate_segment(min(idx, len(self._result.segments) - 1))

    def _reset_selected_to_original(self) -> None:
        if not self._result:
            return
        idx = self._selected_segment_index()
        if 0 <= idx < len(self._result.segments):
            self._result.segments[idx].reset_to_original()
        else:
            for s in self._result.segments:
                s.reset_to_original()
        self._set_active_result(self._result)

    # ------------------------------------------------------------ Timeline ve Oynatma Olayları
    def _on_timeline_seek(self, pos_sec: float) -> None:
        tot_dur = self.timeline_widget.duration_sec
        self.vp_tc_lbl.setText(f"{format_tc_frames(pos_sec, 24.0)} / {format_tc_frames(tot_dur, 24.0)}")
        try:
            self.ctx.player.seek(pos_sec)
        except Exception:
            pass

    def _on_timeline_segment_selected(self, idx: int) -> None:
        self._activate_segment(idx)

    def _seek_relative(self, dt: float) -> None:
        new_t = max(0.0, min(self.timeline_widget.duration_sec, self.timeline_widget.playhead_pos + dt))
        self.timeline_widget.set_playhead(new_t)
        self._on_timeline_seek(new_t)

    def _toggle_playback(self) -> None:
        try:
            if self.ctx.player.is_playing:
                self.ctx.player.pause()
                self.vp_play_btn.setIcon(icons.icon("play", size=14))
            else:
                self.ctx.player.play()
                self.vp_play_btn.setIcon(icons.icon("pause", size=14))
        except Exception:
            pass

    def _preview_selected_segment(self) -> None:
        if not self._result or not self._selected_media or not self._selected_media.exists():
            return
        idx = self._selected_segment_index()
        if idx < 0 or idx >= len(self._result.segments):
            return
        seg = self._result.segments[idx]
        try:
            play_path = self._selected_media
            if play_path.suffix.lower() in C.VIDEO_EXTENSIONS:
                tmp_wav = Path(self.ctx.config.recordings_dir) / f".preview_{play_path.stem}.wav"
                if not tmp_wav.exists():
                    import soundfile as sf
                    arr, sr = MediaExtractor.extract_audio(play_path, target_sample_rate=48000, target_channels=1)
                    sf.write(str(tmp_wav), arr, sr)
                play_path = tmp_wav

            self.ctx.load_for_playback(play_path)
            self.ctx.player.seek(seg.start)
            self.ctx.player.play()
            dur_ms = max(150, int((seg.end - seg.start) * 1000))
            self._segment_stop_timer.start(dur_ms)
        except Exception as exc:
            logger.warning(f"Segment önizleme hatası: {exc}")

    def _stop_segment_preview(self) -> None:
        try:
            self.ctx.player.pause()
        except Exception:
            pass

    # ------------------------------------------------------------ Dışa Aktarma
    def _export_subtitles(self, fmt: str) -> None:
        if not self._result or not self._result.segments:
            return
        stem = self._selected_media.stem if self._selected_media else "altyazi"
        filter_map = {
            "srt": "SubRip Altyazı (*.srt)",
            "vtt": "WebVTT Altyazı (*.vtt)",
            "txt": "Düz Metin (*.txt)",
            "json": "JSON Transkript (*.json)",
        }
        dest, _ = QFileDialog.getSaveFileName(
            self,
            f"Transkripti {fmt.upper()} Olarak Dışa Aktar",
            str(Path(self.ctx.config.recordings_dir) / f"{stem}.{fmt}"),
            filter_map.get(fmt, "Tüm Dosyalar (*.*)"),
        )
        if not dest:
            return
        try:
            if fmt == "srt":
                WhisperTranscriber.export_srt(self._result.segments, dest)
            elif fmt == "vtt":
                WhisperTranscriber.export_vtt(self._result.segments, dest)
            elif fmt == "txt":
                WhisperTranscriber.export_txt(self._result.segments, dest)
            elif fmt == "json":
                WhisperTranscriber.export_json(self._result, dest)

            ok, msg = validate_exported_subtitle_file(dest, fmt=fmt)
            if not ok:
                QMessageBox.warning(self, "Biçim Doğrulama Uyarısı", msg)
                return

            QMessageBox.information(
                self,
                "Dışa Aktarma Başarılı",
                f"Dosya kaydedildi ve doğrulandı:\n{dest}\n\nDoğrulama: {msg}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Dışa Aktarma Hatası", str(e))

    # ------------------------------------------------------------ Video Editörüne Aktarım
    def _open_transfer_to_editor_dialog(self) -> None:
        if not self._result or not self._result.segments:
            return
        dlg = TransferToEditorDialog(
            self._result,
            has_existing=False,
            default_offset=0.0,
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            opts = dlg.selected_options()
            payload = {
                "result": self._result,
                "mode": opts["mode"],
                "offset_time": opts["offset_time"],
                "update_existing": opts["update_existing"],
            }
            self.ctx.transcriptionToTimelineRequested.emit(payload)
            ToastNotification.show_toast(self, "Altyazılar video editörüne aktarıldı", "success")

    def _send_to_timeline(self) -> None:
        if not self._result or not self._result.segments:
            return
        self.ctx.transcriptionToTimelineRequested.emit(self._result)
        ToastNotification.show_toast(self, "Altyazılar video editörüne aktarıldı", "success")
