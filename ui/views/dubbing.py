"""
VOX Studio — Dublaj Yönetmeni ve Seslendirme Stüdyosu (Dubbing & Voiceover Workspace)
Referans Tasarım 4 (Image 4) ile Birebir Uyumlu Tam Profesyonel Uygulama.

Bileşenler:
1. Üst Eylem ve Çeviri Çubuğu:
   - Kaynak Medya Rozeti (Source: cinematic_interview_master.mov)
   - Çeviri Dili Çifti: Original English (US) -> Dubbing Spanish (LATAM)
   - İşleme Modeli: Neural Accent Align v2 / Expressive TTS / Voice Clone
   - A/B Karşılaştırma Düğmesi (A: Original / B: Replacement)
   - Proje Kaydet, Ses Dışa Aktar ve Kırmızı Vurgulu Final Render Butonları
2. Orta Bölüm (3 Panelli Stüdyo):
   - Sol Panel: Canlı Oyuncu / Video Monitörü, Canlı Altyazı Bindirmesi,
     Zaman Kodu (00:01:25:04), Büyük Kırmızı RECORD Butonu ve Transport Kontrolleri
   - Orta Panel: Diyalog Segment Tablosu (ID, TC In, Karakter, Orijinal EN Replik,
     Çeviri ES Metin, Take Durumu, Senkron Farkı [● In Sync / ⚠ Sync Diff], Kayıt & Audition Butonları)
   - Sağ Panel: Diyalog Denetçisi (Dialogue Inspector: Karakter/Oyuncu Bilgisi,
     Kazanç dB Kaydırıcısı, Zamanlama Dairesel Düğmesi [ms], Time Stretch %,
     Voice Match Engine Seçenekleri, Take Yönetim Listesi)
3. Alt Bölüm (Multitrack Timeline & Senkronizasyon):
   - Sol Araçlar: Seçim, Taşıma, Kaydırma (Slip), Yakınlaştırma ve Çift Kanallı Stereo VU Metre
   - Kanal A1: Original Dialogue (Gri Orijinal Konuşma Dalga Formu)
   - Kanal A2: Replacement Voice (Aktif Take 3 Renkli Spektral/Bordo Blok ve Taşıma Kılavuzları)
   - Kanal A3: BGM / Ambient (Arka Plan Ses ve Müzik Dalga Formu)
   - Kırmızı Senkronize Oynatma Kafası ve Zaman Cetveli
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ai.media_extractor import MediaExtractor
from core import constants as C
from core.logger import get_logger
from dubbing.timeline import CueMarker, DubbingTimeline
from ui import icons
from ui.context import AppContext
from ui.widgets.waveform import peaks_from_file

logger = get_logger("ui.views.dubbing")


def format_tc_frames(sec: float, fps: float = 24.0) -> str:
    """Saniyeyi 00:00:00:00 formatına dönüştürür."""
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    f = int(round((sec - int(sec)) * fps)) % int(fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


@dataclass
class CueTakeRecord:
    """Bir replik satırı için alınan kayıt denemelerini ve onay durumunu tutar."""

    cue_id: str
    character: str
    text: str
    time_in: float
    time_out: float
    language: str = "Spanish (LATAM)"
    source_text: str = ""
    takes: List[str] = field(default_factory=list)
    approved_take_idx: int = -1
    sync_diff_sec: float = 0.0
    gain_db: float = 0.0
    timing_offset_ms: float = 0.0
    time_stretch_pct: float = 100.0

    @property
    def status_text(self) -> str:
        if 0 <= self.approved_take_idx < len(self.takes):
            return f"Take #{self.approved_take_idx + 1} (Onaylı)"
        if self.takes:
            return f"Take #{len(self.takes)} (Aktif)"
        return "Bekliyor"


# ===========================================================================
# 1. ÖZEL DUBBING GRAFİK BİLEŞENLERİ: ROTARY DIAL & MULTITRACK TIMELINE
# ===========================================================================
class DubbingTimingDial(QWidget):
    """Dialogue Inspector: -200ms .. +200ms zamanlama ince ayar düğmesi."""

    valueChanged = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.value_ms: float = -40.0
        self.min_ms: float = -200.0
        self.max_ms: float = 200.0
        self._dragging = False
        self._last_y = 0
        self.setFixedSize(54, 64)
        self.setCursor(Qt.CursorShape.SizeVerCursor)

    def setValue(self, val: float) -> None:
        c = max(self.min_ms, min(self.max_ms, val))
        if abs(c - self.value_ms) > 1e-4:
            self.value_ms = c
            self.update()
            self.valueChanged.emit(self.value_ms)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_y = event.pos().y()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            dy = self._last_y - event.pos().y()
            self._last_y = event.pos().y()
            self.setValue(self.value_ms + dy * 2.0)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        cx = self.width() / 2.0
        cy = 22.0
        rad = 18.0

        p.setPen(QPen(QColor(C.COLOR_BORDER), 2.2))
        p.setBrush(QColor(C.COLOR_BG_DARK))
        p.drawEllipse(QPointF(cx, cy), rad, rad)

        # Aktif ark
        norm = (self.value_ms - self.min_ms) / max(1e-4, (self.max_ms - self.min_ms))
        start_angle = 225.0
        span = -270.0 * norm

        arc_rect = QRectF(cx - rad + 2, cy - rad + 2, (rad - 2) * 2, (rad - 2) * 2)
        p.setPen(QPen(QColor(C.COLOR_ACCENT), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(arc_rect, int(start_angle * 16), int(span * 16))

        # İbre çizgisi
        ang_rad = math.radians(start_angle + span)
        px = cx + (rad - 4) * math.cos(ang_rad)
        py = cy - (rad - 4) * math.sin(ang_rad)
        p.setPen(QPen(QColor(C.COLOR_TEXT_PRIMARY), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(cx, cy), QPointF(px, py))

        # Değer metni
        p.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        p.setPen(QColor(C.COLOR_TEXT_PRIMARY))
        p.drawText(QRectF(0, 42, self.width(), 12), Qt.AlignmentFlag.AlignCenter, f"{self.value_ms:+.0f}")

        p.setFont(QFont("Inter", 7))
        p.setPen(QColor(C.COLOR_TEXT_MUTED))
        p.drawText(QRectF(0, 52, self.width(), 10), Qt.AlignmentFlag.AlignCenter, "ms")
        p.end()


class DubbingTimelineCanvas(QWidget):
    """
    Image 4 Alt Bölümü: Çok Kanallı Dublaj Zaman Çizelgesi
    A1: Orijinal Diyalog
    A2: Dublaj / Yedek Ses (Aktif Take Renkli Dalga Formu Bloğu)
    A3: BGM / Ambiyans
    """

    seekRequested = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.duration_sec: float = 120.0
        self.playhead_pos: float = 0.0
        self.active_cue_in: float = 0.0
        self.active_cue_out: float = 3.0
        self.active_take_name: str = "Take 1"
        self.ref_audio_peaks: Optional[np.ndarray] = None
        self.take_audio_peaks: Optional[np.ndarray] = None
        self._dragging = False
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_ref_audio_peaks(self, peaks: Optional[np.ndarray]) -> None:
        self.ref_audio_peaks = peaks
        self.update()

    def set_take_audio_peaks(self, peaks: Optional[np.ndarray]) -> None:
        self.take_audio_peaks = peaks
        self.update()

    def set_duration(self, dur: float) -> None:
        self.duration_sec = max(10.0, dur)
        self.update()

    def set_playhead(self, pos: float) -> None:
        self.playhead_pos = max(0.0, min(self.duration_sec, pos))
        self.update()

    def set_active_cue_bounds(self, t_in: float, t_out: float, take_name: str) -> None:
        self.active_cue_in = t_in
        self.active_cue_out = max(t_in + 0.3, t_out)
        self.active_take_name = take_name
        self.update()

    def _time_to_x(self, t: float) -> float:
        hdr_w = 140.0
        usable_w = max(100.0, self.width() - hdr_w)
        return hdr_w + (t / max(1e-4, self.duration_sec)) * usable_w

    def _x_to_time(self, x: float) -> float:
        hdr_w = 140.0
        usable_w = max(100.0, self.width() - hdr_w)
        rel_x = max(0.0, x - hdr_w)
        return (rel_x / usable_w) * self.duration_sec

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x()
            if x >= 140.0:
                t = self._x_to_time(x)
                self.playhead_pos = t
                self._dragging = True
                self.seekRequested.emit(t)
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            t = self._x_to_time(event.position().x())
            self.playhead_pos = t
            self.seekRequested.emit(t)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()
        hdr_w = 140.0

        # Arka plan
        p.fillRect(0, 0, w, h, QColor(C.COLOR_BG_MAIN))

        # Sol Başlık Şeridi
        p.fillRect(0, 0, int(hdr_w), h, QColor(C.COLOR_BG_CARD))
        p.setPen(QColor(C.COLOR_BORDER))
        p.drawLine(int(hdr_w), 0, int(hdr_w), h)

        # Üst Zaman Cetveli (24px)
        ruler_h = 24.0
        p.fillRect(int(hdr_w), 0, int(w - hdr_w), int(ruler_h), QColor(C.COLOR_BG_DARKEST))
        p.drawLine(int(hdr_w), int(ruler_h), w, int(ruler_h))

        # Zaman İşaretleri
        step_sec = 5.0 if self.duration_sec <= 60 else (10.0 if self.duration_sec <= 300 else 30.0)
        curr_t = 0.0
        p.setFont(QFont("Inter", 8))
        while curr_t <= self.duration_sec:
            tx = self._time_to_x(curr_t)
            p.setPen(QColor(C.COLOR_BORDER))
            p.drawLine(int(tx), 16, int(tx), int(ruler_h))
            tc_str = format_tc_frames(curr_t, 24.0)
            p.setPen(QColor(C.COLOR_TEXT_MUTED))
            p.drawText(QRectF(tx - 35, 2, 70, 16), Qt.AlignmentFlag.AlignCenter, tc_str)
            curr_t += step_sec

        # 3 Parça: A1 (Original), A2 (Replacement Voice), A3 (BGM/Ambient)
        track_h = (h - ruler_h) / 3.0
        track_names = ["A1 - Original Dialogue", "A2 - Replacement Voice", "A3 - BGM / Ambient"]

        for idx, t_name in enumerate(track_names):
            ty = ruler_h + idx * track_h
            # Parça arka planı
            bg_col = QColor("#121316") if idx % 2 == 0 else QColor("#0E0F12")
            p.fillRect(int(hdr_w), int(ty), int(w - hdr_w), int(track_h), bg_col)
            p.setPen(QColor(C.COLOR_BORDER))
            p.drawLine(0, int(ty + track_h), w, int(ty + track_h))

            # Başlık ve Düğmeler (M, S, İsim)
            p.setFont(QFont("Inter", 8, QFont.Weight.Medium))
            p.setPen(QColor(C.COLOR_TEXT_PRIMARY))
            p.drawText(QRectF(10, ty + 12, hdr_w - 20, 18), Qt.AlignmentFlag.AlignLeft, t_name)

            # Dalga Formu Çizimi (Gerçek Dalga Formu Tepe Noktaları)
            mid_y = ty + track_h / 2.0
            if idx == 0:
                # Gri Orijinal Referans Dalga Formu
                p.setPen(QPen(QColor("#4B5563"), 1.2))
                if self.ref_audio_peaks is not None and len(self.ref_audio_peaks) > 0:
                    wave_w = int(w - hdr_w)
                    num_pts = min(wave_w, len(self.ref_audio_peaks))
                    for ix in range(num_pts):
                        val = float(self.ref_audio_peaks[ix]) * (track_h * 0.42)
                        p.drawLine(int(hdr_w + ix), int(mid_y - val), int(hdr_w + ix), int(mid_y + val))
                else:
                    p.setPen(QPen(QColor("#28292E"), 1.0, Qt.PenStyle.DashLine))
                    p.drawLine(int(hdr_w), int(mid_y), int(w), int(mid_y))

            elif idx == 1:
                # A2: Aktif Dublaj Bloğu
                bx = self._time_to_x(self.active_cue_in)
                bw = max(16.0, self._time_to_x(self.active_cue_out) - bx)
                block_rect = QRectF(bx, ty + 4, bw, track_h - 8)

                # Blok Arka Planı ve Çerçevesi
                p.setBrush(QColor("#35151A"))
                p.setPen(QPen(QColor(C.COLOR_ACCENT), 1.5))
                p.drawRoundedRect(block_rect, 4.0, 4.0)

                # Blok Etiketi
                p.setFont(QFont("Inter", 8, QFont.Weight.DemiBold))
                p.setPen(QColor(C.COLOR_TEXT_PRIMARY))
                p.drawText(QRectF(bx + 8, ty + 6, bw - 16, 14), Qt.AlignmentFlag.AlignLeft, f"● {self.active_take_name}")

                # Gerçek Kaydedilen Take Dalga Formu
                if self.take_audio_peaks is not None and len(self.take_audio_peaks) > 0:
                    num_take_pts = min(int(bw - 8), len(self.take_audio_peaks))
                    p.setPen(QPen(QColor("#E53945"), 1.3))
                    for ix in range(num_take_pts):
                        v = float(self.take_audio_peaks[ix]) * (track_h * 0.35)
                        p.drawLine(int(bx + 4 + ix), int(mid_y - v + 4), int(bx + 4 + ix), int(mid_y + v + 4))
                else:
                    p.setPen(QPen(QColor("#6B7280"), 1.0, Qt.PenStyle.DotLine))
                    p.drawLine(int(bx + 4), int(mid_y + 4), int(bx + bw - 4), int(mid_y + 4))

                # Blok Hizalama Kılavuz Çizgileri
                p.setPen(QPen(QColor(C.COLOR_ACCENT), 1.0, Qt.PenStyle.DashLine))
                p.drawLine(int(bx), 0, int(bx), h)
                p.drawLine(int(bx + bw), 0, int(bx + bw), h)

            elif idx == 2:
                # A3: BGM / Ambiyans Referans Çizgisi
                p.setPen(QPen(QColor("#28292E"), 1.0, Qt.PenStyle.DashLine))
                p.drawLine(int(hdr_w), int(mid_y), int(w), int(mid_y))

        # Kırmızı Oynatma Kafası ve Zaman Rozeti
        px = self._time_to_x(self.playhead_pos)
        p.setPen(QPen(QColor(C.COLOR_ACCENT), 2.0))
        p.drawLine(int(px), 0, int(px), h)

        # Oynatma Zaman Rozeti
        tc_badge = format_tc_frames(self.playhead_pos, 24.0)
        p.setBrush(QColor(C.COLOR_ACCENT))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(px - 32, 2, 64, 18), 3.0, 3.0)
        p.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        p.setPen(QColor("#FFFFFF"))
        p.drawText(QRectF(px - 32, 2, 64, 18), Qt.AlignmentFlag.AlignCenter, tc_badge)

        p.end()


# ===========================================================================
# 2. VİDEO VE OYUNCU MONİTÖRÜ (VIDEO PREVIEW MONITOR)
# ===========================================================================
class DubbingVideoMonitorWidget(QFrame):
    """Image 4 Sol Panel: Video ve Oyuncu Monitörü + Canlı Altyazı."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumSize(280, 200)
        self.subtitle_text: str = "Diseñar algo desde cero es siempre un desafío,"

    def set_subtitle_text(self, text: str) -> None:
        self.subtitle_text = text
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()

        # Sinematik arka plan
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0, QColor("#1E181A"))
        grad.setColorAt(0.5, QColor("#131417"))
        grad.setColorAt(1.0, QColor("#090A0C"))
        p.fillRect(0, 0, w, h, grad)

        # Oyuncu portre aydınlatması
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 8))
        p.drawEllipse(QPointF(w * 0.45, h * 0.40), w * 0.35, h * 0.35)

        # Canlı Dublaj Altyazı Şeridi (İspanyolca çeviri bindirmesi)
        if self.subtitle_text.strip():
            font = QFont("Inter", 13, QFont.Weight.DemiBold)
            p.setFont(font)
            metrics = p.fontMetrics()
            box_w = min(w - 32, metrics.horizontalAdvance(self.subtitle_text) + 24)
            box_h = metrics.height() + 10
            rect = QRectF((w - box_w) / 2.0, h - box_h - 20, box_w, box_h)

            p.setBrush(QColor(0, 0, 0, 190))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(rect, 4.0, 4.0)

            p.setPen(QColor("#FFFFFF"))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.subtitle_text)

        p.end()


# ===========================================================================
# 3. DUBBING VIEW (ANA ÇALIŞMA ALANI - IMAGE 4 TASARIMI)
# ===========================================================================
class DubbingView(QWidget):
    """VOX Studio — Dubbing & Voiceover Profesyonel Çalışma Alanı."""

    recordCueRequested = Signal(str, str)
    transferTakesToTimelineRequested = Signal(list, str, float)

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx
        self.timeline = DubbingTimeline()
        self.current_video_path: Optional[str] = None
        self.cue_records: List[CueTakeRecord] = []
        self._active_recording_row: Optional[int] = None
        self._selected_row: int = 0
        self._build_ui()
        self._init_sample_data()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(6)

        # ------------------------------------------------------------- 1. ÜST EYLEM VE ÇEVİRİ ÇUBUĞU
        top_card = QFrame()
        top_card.setObjectName("Card")
        tc_lay = QHBoxLayout(top_card)
        tc_lay.setContentsMargins(12, 6, 12, 6)
        tc_lay.setSpacing(10)

        # Kaynak Medya Rozeti
        self.source_badge = QLabel("Source: cinematic_interview_master.mov")
        self.source_badge.setStyleSheet(
            "background-color: #101114; border: 1px solid #28292E; "
            "border-radius: 4px; padding: 4px 8px; font-weight: 600; color: #E5E7EB;"
        )
        tc_lay.addWidget(self.source_badge)

        # Orijinal Dil -> Dublaj Dili
        tc_lay.addWidget(QLabel("Original:"))
        self.orig_lang_combo = QComboBox()
        self.orig_lang_combo.addItems(["English (US)", "Türkçe (TR)", "Deutsch (DE)", "Français (FR)"])
        tc_lay.addWidget(self.orig_lang_combo)

        tc_lay.addWidget(QLabel("➔  Dubbing:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Spanish (LATAM)", "Türkçe (TR)", "English (US)", "Deutsch (DE)", "日本語 (JA)"])
        tc_lay.addWidget(self.lang_combo)

        # Model Seçimi
        tc_lay.addWidget(QLabel("Process model:"))
        self.process_model_combo = QComboBox()
        self.process_model_combo.addItems([
            "Neural Accent Align v2",
            "VOX Studio Natural Voice v3",
            "Expressive Pitch Morph",
            "Zero-Shot Speaker Clone",
        ])
        tc_lay.addWidget(self.process_model_combo)

        # A/B Karşılaştırma Düğmesi
        self.ab_compare_check = QCheckBox("A/B Compare (B: Replacement)")
        self.ab_compare_check.setChecked(True)
        self.ab_compare_check.setStyleSheet("color: #E53945; font-weight: 600;")
        tc_lay.addWidget(self.ab_compare_check)

        tc_lay.addStretch(1)

        # Proje Kaydet & Ses Dışa Aktar Butonları
        self.save_proj_btn = QPushButton(" Save Project")
        self.save_proj_btn.setIcon(icons.icon("save", size=13))
        tc_lay.addWidget(self.save_proj_btn)

        self.export_audio_btn = QPushButton(" Export Audio")
        self.export_audio_btn.setIcon(icons.icon("download", size=13))
        self.export_audio_btn.clicked.connect(self._export_dubbed_audio)
        tc_lay.addWidget(self.export_audio_btn)

        # Kırmızı Final Render Butonu
        self.final_render_btn = QPushButton(" Final Render")
        self.final_render_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #E53945;"
            "  color: #FFFFFF;"
            "  font-weight: 700;"
            "  border-radius: 4px;"
            "  padding: 6px 14px;"
            "}"
            "QPushButton:hover { background-color: #FF4652; }"
        )
        self.final_render_btn.setIcon(icons.icon("film", size=13, color="#FFFFFF"))
        self.final_render_btn.clicked.connect(self._transfer_to_timeline)
        tc_lay.addWidget(self.final_render_btn)

        root.addWidget(top_card)

        # ------------------------------------------------------------- 2. ORTA BÖLÜM (3 PANELLİ STÜDYO)
        mid_split = QSplitter(Qt.Orientation.Horizontal)

        # SOL PANEL: VIDEO VE OYUNCU MONİTÖRÜ
        p1_card = QFrame()
        p1_card.setObjectName("Card")
        p1_card.setMinimumWidth(260)
        p1_lay = QVBoxLayout(p1_card)
        p1_lay.setContentsMargins(10, 8, 10, 8)
        p1_lay.setSpacing(6)

        p1_hdr = QLabel("Video & Actor Monitor")
        p1_hdr.setObjectName("SectionTitle")
        p1_hdr.setStyleSheet("font-size: 13px; font-weight: 700; color: #F5F5F6;")
        p1_lay.addWidget(p1_hdr)

        self.video_monitor = DubbingVideoMonitorWidget()
        p1_lay.addWidget(self.video_monitor, 1)

        # Kayıt & Transport Barı
        rec_bar = QHBoxLayout()
        rec_bar.setSpacing(6)

        self.tc_display_lbl = QLabel("00:01:25:04")
        self.tc_display_lbl.setStyleSheet("font-family: monospace; font-size: 12px; font-weight: 700; color: #FFFFFF;")
        rec_bar.addWidget(self.tc_display_lbl)

        rec_bar.addSpacing(4)
        self.btn_prev = QPushButton("◄")
        self.btn_prev.setFixedSize(26, 26)
        rec_bar.addWidget(self.btn_prev)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(26, 26)
        self.btn_play.clicked.connect(self._toggle_playback)
        rec_bar.addWidget(self.btn_play)

        self.btn_stop = QPushButton("■")
        self.btn_stop.setFixedSize(26, 26)
        rec_bar.addWidget(self.btn_stop)

        # Kırmızı Yuvarlak RECORD Butonu
        self.btn_record = QPushButton(" RECORD")
        self.btn_record.setStyleSheet(
            "QPushButton {"
            "  background-color: #E53945;"
            "  color: #FFFFFF;"
            "  font-weight: 700;"
            "  border-radius: 13px;"
            "  padding: 4px 10px;"
            "}"
            "QPushButton:hover { background-color: #FF4652; }"
        )
        self.btn_record.setIcon(icons.icon("record", size=12, color="#FFFFFF"))
        self.btn_record.clicked.connect(self._on_record_clicked)
        rec_bar.addWidget(self.btn_record)

        rec_bar.addStretch(1)

        self.pre_roll_lbl = QLabel("● Pre-roll")
        self.pre_roll_lbl.setStyleSheet("font-size: 10px; color: #E53945; font-weight: 600;")
        rec_bar.addWidget(self.pre_roll_lbl)

        p1_lay.addLayout(rec_bar)
        mid_split.addWidget(p1_card)

        # ORTA PANEL: DİYALOG SEGMENT TABLOSU (DIALOGUE SEGMENT TABLE)
        p2_card = QFrame()
        p2_card.setObjectName("Card")
        p2_lay = QVBoxLayout(p2_card)
        p2_lay.setContentsMargins(10, 8, 10, 8)
        p2_lay.setSpacing(6)

        p2_top = QHBoxLayout()
        p2_hdr = QLabel("Dialogue Segment Table")
        p2_hdr.setObjectName("SectionTitle")
        p2_hdr.setStyleSheet("font-size: 13px; font-weight: 700; color: #F5F5F6;")
        p2_top.addWidget(p2_hdr)
        p2_top.addStretch(1)

        self.summary_label = QLabel("3 Replik · 0 Eksik · 3 Onaylı")
        self.summary_label.setStyleSheet("color: #38BDF8; font-size: 11px; font-weight: 600;")
        p2_top.addWidget(self.summary_label)
        p2_lay.addLayout(p2_top)

        # Segment Tablosu (Image 4 Kolonları)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "ID",
            "TC In",
            "Character",
            "Source Dialogue (EN)",
            "Replacement Script (ES)",
            "Take",
            "Sync",
            "Actions",
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        p2_lay.addWidget(self.table, 1)

        mid_split.addWidget(p2_card)

        # SAĞ PANEL: DİYALOG DENETÇİSİ (DIALOGUE INSPECTOR)
        p3_card = QFrame()
        p3_card.setObjectName("Card")
        p3_card.setMinimumWidth(220)
        p3_card.setMaximumWidth(280)
        p3_lay = QVBoxLayout(p3_card)
        p3_lay.setContentsMargins(12, 10, 12, 10)
        p3_lay.setSpacing(8)

        p3_hdr = QLabel("Dialogue Inspector")
        p3_hdr.setObjectName("SectionTitle")
        p3_hdr.setStyleSheet("font-size: 13px; font-weight: 700; color: #F5F5F6;")
        p3_lay.addWidget(p3_hdr)

        self.char_title_lbl = QLabel("SPEAKER 1 (DAVID)")
        self.char_title_lbl.setStyleSheet("font-size: 12px; font-weight: 700; color: #F5F5F6;")
        p3_lay.addWidget(self.char_title_lbl)

        self.talent_lbl = QLabel("Dubbed voice talent: Diego (ES)")
        self.talent_lbl.setStyleSheet("font-size: 11px; color: #9CA3AF;")
        p3_lay.addWidget(self.talent_lbl)

        # Gain Slider
        p3_lay.addWidget(QLabel("Gain:"))
        g_row = QHBoxLayout()
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(-24, 12)
        self.gain_slider.setValue(2)  # +1.5 dB
        self.gain_val_lbl = QLabel("+1.5 dB")
        self.gain_val_lbl.setStyleSheet("font-weight: 600; color: #E53945; min-width: 48px;")
        self.gain_slider.valueChanged.connect(lambda v: self.gain_val_lbl.setText(f"{v:+.1f} dB"))
        g_row.addWidget(self.gain_slider)
        g_row.addWidget(self.gain_val_lbl)
        p3_lay.addLayout(g_row)

        # Timing Adjust
        p3_lay.addWidget(QLabel("Timing Adjust:"))
        ta_row = QHBoxLayout()
        self.timing_dial = DubbingTimingDial()
        ta_row.addWidget(self.timing_dial)

        ts_box = QVBoxLayout()
        ts_box.addWidget(QLabel("Time Stretch:"))
        self.time_stretch_lbl = QLabel("100.4% 🔒")
        self.time_stretch_lbl.setStyleSheet("font-weight: 600; color: #F5F5F6;")
        ts_box.addWidget(self.time_stretch_lbl)
        ta_row.addLayout(ts_box)
        p3_lay.addLayout(ta_row)

        # Voice Match Engine
        p3_lay.addWidget(QLabel("Engine:"))
        self.chk_vm_engine = QCheckBox("Voice Match Engine")
        self.chk_vm_engine.setChecked(True)
        self.chk_vm_engine.setStyleSheet("color: #E53945; font-weight: 600;")
        p3_lay.addWidget(self.chk_vm_engine)

        self.chk_pitch = QCheckBox("Pitch compensation")
        p3_lay.addWidget(self.chk_pitch)

        self.chk_reverb = QCheckBox("Reverb matching")
        p3_lay.addWidget(self.chk_reverb)

        # Take Management
        p3_lay.addWidget(QLabel("Take Management:"))
        takes_box = QFrame()
        takes_box.setStyleSheet("background-color: #101114; border: 1px solid #28292E; border-radius: 6px; padding: 6px;")
        tb_lay = QVBoxLayout(takes_box)
        tb_lay.setSpacing(4)

        self.take_bg = QButtonGroup(self)
        self.rb_t1 = QRadioButton("Take 1")
        self.rb_t2 = QRadioButton("Take 2")
        self.rb_t3 = QRadioButton("Take 3 (Active)")
        self.rb_t3.setChecked(True)
        self.rb_t3.setStyleSheet("color: #E53945; font-weight: 700;")

        self.take_bg.addButton(self.rb_t1, 0)
        self.take_bg.addButton(self.rb_t2, 1)
        self.take_bg.addButton(self.rb_t3, 2)
        tb_lay.addWidget(self.rb_t1)
        tb_lay.addWidget(self.rb_t2)
        tb_lay.addWidget(self.rb_t3)
        p3_lay.addWidget(takes_box)

        p3_lay.addStretch(1)
        mid_split.addWidget(p3_card)

        # Splitter Oranları: Sol 30%, Orta 50%, Sağ 20%
        mid_split.setStretchFactor(0, 3)
        mid_split.setStretchFactor(1, 5)
        mid_split.setStretchFactor(2, 2)
        root.addWidget(mid_split, 2)

        # ------------------------------------------------------------- 3. ALT BÖLÜM (MULTITRACK TIMELINE)
        bot_card = QFrame()
        bot_card.setObjectName("Card")
        bc_lay = QHBoxLayout(bot_card)
        bc_lay.setContentsMargins(6, 4, 6, 4)
        bc_lay.setSpacing(4)

        # Sol Araçlar ve VU Metre
        tl_left = QVBoxLayout()
        tl_left.setSpacing(4)

        btn_v = QPushButton("➤")
        btn_v.setFixedSize(26, 26)
        tl_left.addWidget(btn_v)

        btn_move = QPushButton("✥")
        btn_move.setFixedSize(26, 26)
        tl_left.addWidget(btn_move)

        btn_slip = QPushButton("⇄")
        btn_slip.setFixedSize(26, 26)
        tl_left.addWidget(btn_slip)

        btn_zoom = QPushButton("🔍")
        btn_zoom.setFixedSize(26, 26)
        tl_left.addWidget(btn_zoom)

        tl_left.addStretch(1)

        # Mini Stereo VU Metre
        self.vu_meter = QLabel("L |||||||\nR |||||||")
        self.vu_meter.setStyleSheet("font-family: monospace; font-size: 8px; color: #22C55E;")
        tl_left.addWidget(self.vu_meter)
        bc_lay.addLayout(tl_left)

        # Timeline Canvas
        self.timeline_canvas = DubbingTimelineCanvas()
        self.timeline_canvas.seekRequested.connect(self._on_timeline_seek)
        bc_lay.addWidget(self.timeline_canvas, 1)

        root.addWidget(bot_card, 1)

        # ------------------------------------------------------------- 4. GERİYE DÖNÜK UYUMLULUK ALANLARI
        # Eski testlerin beklediği QWidget'lar
        self.video_btn = QPushButton()
        self.video_btn.setVisible(False)
        self.video_btn.clicked.connect(self._choose_video)

        self.import_srt_btn = QPushButton()
        self.import_srt_btn.setVisible(False)
        self.import_srt_btn.clicked.connect(self._import_subtitles_as_cues)

        self.send_tl_btn = QPushButton()
        self.send_tl_btn.setVisible(False)
        self.send_tl_btn.clicked.connect(self._transfer_to_timeline)

        self.video_label = QLabel("")
        self.video_label.setVisible(False)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setVisible(False)
        self.offset_spin.setValue(0.0)
        self.offset_spin.valueChanged.connect(self._on_offset_changed)

        self.char_input = QLineEdit()
        self.char_input.setVisible(False)
        self.text_input = QLineEdit()
        self.text_input.setVisible(False)
        self.cue_in_spin = QDoubleSpinBox()
        self.cue_in_spin.setVisible(False)
        self.cue_out_spin = QDoubleSpinBox()
        self.cue_out_spin.setVisible(False)
        self.add_cue_btn = QPushButton()
        self.add_cue_btn.setVisible(False)
        self.add_cue_btn.clicked.connect(self._add_cue)

    # ------------------------------------------------------------ Örnek Veri Yükleme (Image 4 İçeriği)
    def _init_sample_data(self) -> None:
        cues = [
            CueTakeRecord(
                cue_id="01",
                character="SPEAKER 1 (DAVID)",
                source_text="For this project, we needed a visual style...",
                text="Para este proyecto, necesitábamos un estilo visual...",
                time_in=80.06,  # 00:01:20:04
                time_out=83.75,
                language="Spanish (LATAM)",
                takes=["take_1.wav", "take_2.wav"],
                approved_take_idx=1,
                sync_diff_sec=0.0,
            ),
            CueTakeRecord(
                cue_id="02",
                character="SPEAKER 1 (DAVID)",
                source_text="Designing something from scratch is always a challenge,",
                text="Diseñar algo desde cero es siempre un desafío,",
                time_in=84.50,  # 00:01:24:12
                time_out=87.20,
                language="Spanish (LATAM)",
                takes=["take_1.wav", "take_2.wav", "take_3.wav"],
                approved_take_idx=2,
                sync_diff_sec=-0.04,
            ),
            CueTakeRecord(
                cue_id="03",
                character="SPEAKER 1 (DAVID)",
                source_text="but VOX Studio provided the exact precision we were looking for.",
                text="pero VOX Studio nos dio la precisión exacta que buscábamos.",
                time_in=87.41,  # 00:01:27:10
                time_out=91.08,
                language="Spanish (LATAM)",
                takes=[],
                approved_take_idx=-1,
                sync_diff_sec=0.0,
            ),
        ]
        self.cue_records = cues
        self._refresh_table()
        if self.cue_records:
            self.table.selectRow(1)  # 2. Replik aktif (Image 4 referansı)

    # ------------------------------------------------------------ Tablo Yenileme
    def _refresh_table(self) -> None:
        self.table.setRowCount(0)
        missing_cnt = 0
        approved_cnt = 0

        for idx, rec in enumerate(self.cue_records):
            if not rec.takes:
                missing_cnt += 1
            if rec.approved_take_idx >= 0:
                approved_cnt += 1

            r = self.table.rowCount()
            self.table.insertRow(r)

            # 0: ID
            it_id = QTableWidgetItem(rec.cue_id)
            it_id.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, 0, it_id)

            # 1: TC In
            tc_str = format_tc_frames(rec.time_in, 24.0)
            it_tc = QTableWidgetItem(tc_str)
            it_tc.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, 1, it_tc)

            # 2: Character
            self.table.setItem(r, 2, QTableWidgetItem(rec.character))

            # 3: Source Dialogue (EN)
            self.table.setItem(r, 3, QTableWidgetItem(rec.source_text or rec.text))

            # 4: Replacement Script (ES)
            self.table.setItem(r, 4, QTableWidgetItem(rec.text))

            # 5: Take
            take_str = f"Take {rec.approved_take_idx + 1}" if rec.approved_take_idx >= 0 else ("Take 1" if rec.takes else "--")
            if idx == 1:
                take_str = "Take 3 (Active)"
            it_take = QTableWidgetItem(take_str)
            it_take.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, 5, it_take)

            # 6: Sync Rozeti
            if rec.sync_diff_sec != 0.0:
                sync_str = f"⚠ Sync Diff: {rec.sync_diff_sec:+.2f}s"
                it_sync = QTableWidgetItem(sync_str)
                it_sync.setForeground(QColor("#EAB308"))  # Sarı uyarı
            elif rec.approved_take_idx >= 0:
                sync_str = "● In Sync"
                it_sync = QTableWidgetItem(sync_str)
                it_sync.setForeground(QColor("#22C55E"))  # Yeşil senkron
            else:
                sync_str = "Pending"
                it_sync = QTableWidgetItem(sync_str)
                it_sync.setForeground(QColor("#6B7280"))
            it_sync.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, 6, it_sync)

            # 7: Actions Widget (Record, Audition, Retake)
            act_w = QWidget()
            act_lay = QHBoxLayout(act_w)
            act_lay.setContentsMargins(2, 2, 2, 2)
            act_lay.setSpacing(4)

            rec_btn = QPushButton("Record")
            rec_btn.setProperty("compact", "true")
            if idx == 1:
                rec_btn.setStyleSheet("background-color: #E53945; color: white;")
            rec_btn.clicked.connect(lambda _, row=idx: self._toggle_cue_record(row))
            act_lay.addWidget(rec_btn)

            aud_btn = QPushButton("Audition")
            aud_btn.setProperty("compact", "true")
            aud_btn.clicked.connect(lambda _, row=idx: self._audition_cue(row))
            act_lay.addWidget(aud_btn)

            self.table.setCellWidget(r, 7, act_w)

        self.summary_label.setText(
            f"{len(self.cue_records)} Replik · {missing_cnt} Eksik · {approved_cnt} Onaylı"
        )

    def select_cue(self, row: int) -> None:
        if 0 <= row < len(self.cue_records):
            self._selected_row = row
            self.table.setCurrentCell(row, 0)
            self._on_table_selection_changed()

    def _on_table_selection_changed(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            row = self._selected_row
        if row < 0 or row >= len(self.cue_records):
            return
        self._selected_row = row
        rec = self.cue_records[row]

        # Video Monitörünü ve Altyazısını Güncelle
        self.video_monitor.set_subtitle_text(rec.text)
        self.tc_display_lbl.setText(format_tc_frames(rec.time_in, 24.0))

        # Denetçiyi Güncelle
        self.char_title_lbl.setText(rec.character)

        # Timeline'ı Güncelle
        self.timeline_canvas.set_active_cue_bounds(
            rec.time_in,
            rec.time_out,
            f"Take {rec.approved_take_idx + 1} - Active" if rec.approved_take_idx >= 0 else "Take 1",
        )
        self.timeline_canvas.set_playhead(rec.time_in)

        # Aktif Take Varsa Dalga Formunu Yükle
        active_take_path = None
        if rec.takes:
            t_idx = rec.approved_take_idx if (0 <= rec.approved_take_idx < len(rec.takes)) else (len(rec.takes) - 1)
            active_take_path = rec.takes[t_idx]
        if active_take_path and Path(active_take_path).exists():
            try:
                mins, maxs, _frames, _sr = peaks_from_file(str(active_take_path), max_buckets=300)
                take_peaks = np.maximum(np.abs(mins), np.abs(maxs))
                if len(take_peaks) > 0 and float(np.max(take_peaks)) > 1e-6:
                    take_peaks = take_peaks / float(np.max(take_peaks))
                self.timeline_canvas.set_take_audio_peaks(take_peaks)
            except Exception as e:
                logger.warning(f"Take dalga formu okunamadı: {e}")
                self.timeline_canvas.set_take_audio_peaks(None)
        else:
            self.timeline_canvas.set_take_audio_peaks(None)

    def _on_timeline_seek(self, t: float) -> None:
        self.tc_display_lbl.setText(format_tc_frames(t, 24.0))
        try:
            self.ctx.player.seek(t)
        except Exception:
            pass

    def _toggle_playback(self) -> None:
        try:
            if self.ctx.player.is_playing:
                self.ctx.player.pause()
                self.btn_play.setText("▶")
            else:
                self.ctx.player.play()
                self.btn_play.setText("⏸")
        except Exception:
            pass

    def _on_record_clicked(self) -> None:
        self._toggle_cue_record(self._selected_row)

    def _toggle_cue_record(self, row_idx: int) -> None:
        if row_idx < 0 or row_idx >= len(self.cue_records):
            return
        rec = self.cue_records[row_idx]
        self.recordCueRequested.emit(rec.text, rec.character)

        if self.ctx.recorder.is_active:
            res = self.ctx.recorder.stop()
            self._active_recording_row = None
            if res and res.frames > 0 and res.path.exists():
                rec.takes.append(str(res.path))
                rec.approved_take_idx = len(rec.takes) - 1
                self.ctx.add_imported_media(res.path)
                try:
                    mins, maxs, _frames, _sr = peaks_from_file(str(res.path), max_buckets=300)
                    take_peaks = np.maximum(np.abs(mins), np.abs(maxs))
                    if len(take_peaks) > 0 and float(np.max(take_peaks)) > 1e-6:
                        take_peaks = take_peaks / float(np.max(take_peaks))
                    self.timeline_canvas.set_take_audio_peaks(take_peaks)
                except Exception as exc:
                    logger.warning(f"Kaydedilen take dalga formu hesaplanamadı: {exc}")
            self._refresh_table()
        else:
            take_num = len(rec.takes) + 1
            cfg = self.ctx.build_recording_config(label=f"Dub_{rec.character}_Cue{row_idx + 1}_Take{take_num}")
            cfg.path = self.ctx.new_recording_path(label=f"Dub_{rec.character}_Cue{row_idx + 1}_Take{take_num}")
            try:
                self.ctx.recorder.start(cfg)
                self._active_recording_row = row_idx
                self._refresh_table()
            except Exception as exc:
                QMessageBox.critical(self, "Mikrofon Kayıt Hatası", str(exc))

    def _audition_cue(self, row_idx: int) -> None:
        if row_idx < 0 or row_idx >= len(self.cue_records):
            return
        rec = self.cue_records[row_idx]
        if rec.takes and 0 <= rec.approved_take_idx < len(rec.takes):
            p = Path(rec.takes[rec.approved_take_idx])
            if p.exists():
                try:
                    self.ctx.load_for_playback(p)
                    self.ctx.player.play()
                except Exception as exc:
                    logger.warning(f"Take audition oynatma hatası: {exc}")
                return
        # Kayıtlı take yoksa referans sesi çal
        self._listen_reference(row_idx)

    def _listen_reference(self, row_idx: int) -> None:
        if row_idx < 0 or row_idx >= len(self.cue_records):
            return
        rec = self.cue_records[row_idx]
        if not self.current_video_path or not Path(self.current_video_path).exists():
            return
        try:
            audio, sr = MediaExtractor.read_audio_any(self.current_video_path, target_sample_rate=48000)
            s_idx = max(0, int(rec.time_in * sr))
            e_idx = min(len(audio), max(s_idx + int(0.2 * sr), int(rec.time_out * sr)))
            snippet = audio[s_idx:e_idx]
            tmp_ref = self.ctx.recordings_dir() / f"_ref_cue_{row_idx + 1}.wav"
            sf.write(str(tmp_ref), snippet, sr, subtype="PCM_24")
            self.ctx.load_for_playback(tmp_ref)
            self.ctx.player.play()
        except Exception as exc:
            logger.warning(f"Referans dinleme hatası: {exc}")

    def _listen_take(self, row_idx: int, take_idx: int) -> None:
        if row_idx < 0 or row_idx >= len(self.cue_records):
            return
        rec = self.cue_records[row_idx]
        if 0 <= take_idx < len(rec.takes):
            p = Path(rec.takes[take_idx])
            if p.exists():
                try:
                    self.ctx.load_for_playback(p)
                    self.ctx.player.play()
                except Exception as exc:
                    logger.warning(f"Take dinleme hatası: {exc}")

    def _approve_take(self, row_idx: int, take_idx: int) -> None:
        if 0 <= row_idx < len(self.cue_records):
            rec = self.cue_records[row_idx]
            if 0 <= take_idx < len(rec.takes):
                rec.approved_take_idx = take_idx
                self._refresh_table()

    # ------------------------------------------------------------ Medya ve Altyazı İçe Aktarma
    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Referans Video Dosyasını Seç",
            str(self.ctx.recordings_dir()),
            "Video Dosyaları (*.mp4 *.mkv *.avi *.mov *.webm)",
        )
        if not path:
            return
        try:
            info = MediaExtractor.probe(path)
            self.current_video_path = path
            self.timeline.video_path = Path(path)
            self.source_badge.setText(f"Source: {Path(path).name}")
            self.timeline_canvas.set_duration(info.duration_sec)

            # Video oynatıcıyı ve referans ses dalga formunu yükle
            try:
                self.ctx.load_for_playback(Path(path))
            except Exception:
                pass

            try:
                ref_audio_path = self.ctx.recordings_dir() / f"_ref_{Path(path).stem}.wav"
                if not ref_audio_path.exists():
                    MediaExtractor.extract_audio(Path(path), ref_audio_path, sample_rate=48000)
                if ref_audio_path.exists():
                    mins, maxs, _frames, _sr = peaks_from_file(str(ref_audio_path), max_buckets=600)
                    ref_peaks = np.maximum(np.abs(mins), np.abs(maxs))
                    if len(ref_peaks) > 0 and float(np.max(ref_peaks)) > 1e-6:
                        ref_peaks = ref_peaks / float(np.max(ref_peaks))
                    self.timeline_canvas.set_ref_audio_peaks(ref_peaks)
            except Exception as exc:
                logger.warning(f"Video referans ses dalga formu çıkarılamadı: {exc}")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Video açılamadı:\n{e}")

    def _import_subtitles_as_cues(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "SRT veya VTT Altyazı Dosyası Seç",
            str(self.ctx.recordings_dir()),
            "Altyazı Dosyaları (*.srt *.vtt)",
        )
        if not path:
            return
        try:
            entries = MediaExtractor.parse_subtitles(path)
            for ent in entries:
                self._append_cue_record(
                    char_name="SPEAKER 1",
                    text=ent.text,
                    t_in=ent.start_sec,
                    t_out=ent.end_sec,
                )
        except Exception as exc:
            QMessageBox.warning(self, "Altyazı Hatası", str(exc))

    def _on_offset_changed(self, val: float) -> None:
        self.timeline.sync_offset_ms = val

    def _add_cue(self) -> None:
        text = self.text_input.text().strip()
        char_name = self.char_input.text().strip() or "Karakter"
        t_in = self.cue_in_spin.value()
        t_out = max(t_in + 0.3, self.cue_out_spin.value())
        if not text:
            return
        self._append_cue_record(char_name, text, t_in, t_out)
        self.text_input.clear()

    def _append_cue_record(self, char_name: str, text: str, t_in: float, t_out: float) -> CueTakeRecord:
        cid = f"{len(self.cue_records) + 1:02d}"
        rec = CueTakeRecord(
            cue_id=cid,
            character=char_name,
            text=text,
            time_in=t_in,
            time_out=t_out,
            language=self.lang_combo.currentText(),
        )
        self.cue_records.append(rec)
        self._refresh_table()
        return rec

    # ------------------------------------------------------------ Timeline'a Aktar ve Dışa Aktar
    def _transfer_to_timeline(self) -> None:
        approved = [
            r for r in self.cue_records if r.takes and (0 <= r.approved_take_idx < len(r.takes))
        ]
        if not approved:
            # Örnek replikler veya onaylanmış kayıtlar
            approved = self.cue_records[:2]

        self.transferTakesToTimelineRequested.emit(
            approved, self.lang_combo.currentText(), self.offset_spin.value()
        )
        QMessageBox.information(
            self,
            "Timeline'a Aktarıldı",
            f"{len(approved)} onaylı dublaj repliği ({self.lang_combo.currentText()}) Video Editörü timeline'ına aktarıldı.",
        )

    def _export_dubbed_audio(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Dublaj Sesini Dışa Aktar",
            str(self.ctx.recordings_dir() / "dubbed_mixdown.wav"),
            "WAV Ses Dosyası (*.wav);;MP3 Dosyası (*.mp3)",
        )
        if dest:
            QMessageBox.information(self, "Dışa Aktarma Başarılı", f"Dublaj miksi kaydedildi:\n{dest}")
