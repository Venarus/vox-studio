"""
VOX Studio — Profesyonel CapCut Tarzı Akıllı Çok Kanallı Timeline Bileşeni
Mıknatıslı hizalama (Magnetic Snap), klip içi ses dalga formu (waveform) ve dB kazanç çizgisi,
video küçük resim şeridi, katman kilitleme, işaretçiler (markers), B tuşuyla Kesme Aracı (Blade Tool),
imleç merkezli yakınlaştırma/uzaklaştırma (Zoom Sistemi) ve görünür alan kırpması (Frustum Culling).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ai.media_extractor import MediaExtractor
from core.logger import get_logger
from ui import icons
from ui.widgets.waveform import peaks_from_file
from video_editor.engine import TimelineEngine
from video_editor.models import ClipType, TimelineClip, TimelineTrack, TrackType

logger = get_logger("ui.widgets.timeline")

TRACK_HEIGHT = 58
RULER_HEIGHT = 28
HEADER_WIDTH = 236
MIN_ZOOM_PX = 10.0
MAX_ZOOM_PX = 300.0


class TimelineCanvas(QWidget):
    """Zaman çizelgesi kanallarını, dalga formlarını, küçük resimleri ve klipleri çizen interaktif tuval."""

    playheadMoved = Signal(float)
    clipSelected = Signal(object)  # TimelineClip or None
    clipModified = Signal(object)
    toolModeChanged = Signal(str)  # "select" or "blade"
    zoomRequested = Signal(float, float)  # (new_px_per_sec, anchor_time)

    def __init__(self, engine: TimelineEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine = engine
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.px_per_sec = 60.0
        self.selected_clip_id: Optional[str] = None
        self._tool_mode: str = "select"  # "select" | "blade"
        self._blade_hover_x: Optional[float] = None
        self._blade_hover_clip_id: Optional[str] = None

        # Etkileşim durumları
        self._is_dragging_playhead = False
        self._drag_mode: Optional[str] = None  # 'move', 'trim_left', 'trim_right', 'volume'
        self._drag_clip_id: Optional[str] = None
        self._drag_start_x: float = 0.0
        self._drag_start_y: float = 0.0
        self._drag_orig_in: float = 0.0
        self._drag_orig_out: float = 0.0
        self._drag_orig_volume: float = 1.0
        self._snapshot_taken_for_drag: bool = False

        # Ses Seviyesi (Volume Envelope) Sürükleme ve Hover Durumu
        self._hover_volume_clip_id: Optional[str] = None
        self._hover_volume_x: float = 0.0
        self._hover_volume_y: float = 0.0
        self._volume_tooltip_pos: Optional[Tuple[float, float]] = None

        # Sürükle-bırak hedef önizlemesi
        self._drop_preview_rect: Optional[QRectF] = None

        # Dalga formu ve küçük resim önbellekleri (bellek dostu)
        self._waveform_cache: Dict[str, Optional[np.ndarray]] = {}
        self._thumb_cache: Dict[str, Optional[QPixmap]] = {}

        self.engine.on_changed(self.update)

    @property
    def tool_mode(self) -> str:
        return self._tool_mode

    def set_tool_mode(self, mode: str) -> None:
        normalized = "blade" if str(mode).lower() == "blade" else "select"
        if self._tool_mode == normalized:
            return
        self._tool_mode = normalized
        if normalized == "select":
            self._blade_hover_x = None
            self._blade_hover_clip_id = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        self.toolModeChanged.emit(self._tool_mode)
        self.update()

    def set_zoom(self, px_per_sec: float) -> None:
        self.px_per_sec = max(MIN_ZOOM_PX, min(MAX_ZOOM_PX, float(px_per_sec)))
        self._update_geometry()
        self.update()

    def time_to_x(self, t: float) -> float:
        return float(t) * self.px_per_sec

    def x_to_time(self, x: float) -> float:
        return max(0.0, float(x) / max(1e-6, self.px_per_sec))

    def _update_geometry(self) -> None:
        dur = max(60.0, self.engine.project.duration + 30.0)
        total_w = int(dur * self.px_per_sec) + 120
        total_h = RULER_HEIGHT + (len(self.engine.project.tracks) * TRACK_HEIGHT) + 40
        self.setMinimumSize(total_w, total_h)
        self.resize(total_w, total_h)

    def get_selected_clip(self) -> Optional[TimelineClip]:
        if not self.selected_clip_id:
            return None
        for t in self.engine.project.tracks:
            for c in t.clips:
                if c.id == self.selected_clip_id:
                    return c
        return None

    def _get_clip_peaks(self, media_path: str) -> Optional[np.ndarray]:
        if not media_path:
            return None
        if media_path in self._waveform_cache:
            return self._waveform_cache[media_path]
        try:
            p = Path(media_path)
            if p.exists() and p.stat().st_size > 0 and p.stat().st_size < 120 * 1024 * 1024:
                mins, maxs, _frames, _sr = peaks_from_file(str(p), max_buckets=160)
                arr = np.maximum(np.abs(mins), np.abs(maxs))
                m = float(np.max(arr)) if arr.size else 0.0
                if m > 1e-6:
                    arr = arr / m
                self._waveform_cache[media_path] = arr
                return arr
        except Exception:
            pass
        self._waveform_cache[media_path] = None
        return None

    def _get_clip_thumb(self, media_path: str) -> Optional[QPixmap]:
        if not media_path:
            return None
        if media_path in self._thumb_cache:
            return self._thumb_cache[media_path]
        try:
            tp = MediaExtractor.extract_thumbnail(media_path, size=(88, 48))
            if tp and tp.exists():
                pm = QPixmap(str(tp))
                if not pm.isNull():
                    self._thumb_cache[media_path] = pm
                    return pm
        except Exception:
            pass
        self._thumb_cache[media_path] = None
        return None

    # ------------------------------------------------------------------- Çizim
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        h = self.height()

        # Görünür yatay alan (Frustum Culling — uzun projelerde donmayı önler)
        vis_rect = event.rect() if event is not None else self.visibleRegion().boundingRect()
        vis_x1 = max(0, vis_rect.left() - 80)
        vis_x2 = min(w, vis_rect.right() + 80) if vis_rect.right() > 0 else w

        # 1. Arka Plan (VOX Studio koyu antrasit çalışma alanı)
        painter.fillRect(vis_x1, 0, max(1, vis_x2 - vis_x1), h, QColor("#090A0C"))

        # 2. Cetvel (Time Ruler & Markers)
        self._draw_ruler(painter, w, vis_x1, vis_x2)

        # 3. Kanallar ve Klipler
        y = RULER_HEIGHT
        for track in self.engine.project.tracks:
            self._draw_track(painter, track, y, w, vis_x1, vis_x2)
            y += TRACK_HEIGHT

        # 4. Sürükle-bırak önizleme kutusu
        if self._drop_preview_rect is not None:
            painter.setPen(QPen(QColor("#00D2D3"), 2, Qt.PenStyle.DashLine))
            painter.fillRect(self._drop_preview_rect, QColor(0, 210, 211, 40))
            painter.drawRect(self._drop_preview_rect)

        # 5. Kesme Aracı (Blade Tool) kırmızı kesim önizleme çizgisi
        if self._tool_mode == "blade" and self._blade_hover_x is not None:
            self._draw_blade_preview(painter, h)

        # 6. Playhead (İmleç İğnesi)
        self._draw_playhead(painter, h)

    def _draw_ruler(self, painter: QPainter, width: int, vis_x1: int = 0, vis_x2: Optional[int] = None):
        if vis_x2 is None:
            vis_x2 = width
        painter.fillRect(vis_x1, 0, max(1, vis_x2 - vis_x1), RULER_HEIGHT, QColor("#101114"))
        painter.setPen(QPen(QColor("#28292E"), 1))
        painter.drawLine(vis_x1, RULER_HEIGHT, vis_x2, RULER_HEIGHT)

        font = QFont("Segoe UI", 8)
        painter.setFont(font)

        # Ölçeğe göre çakışmayan dinamik cetvel adımı
        if self.px_per_sec < 18:
            step_sec = 10.0
            major_mult = 3
        elif self.px_per_sec < 35:
            step_sec = 5.0
            major_mult = 4
        elif self.px_per_sec < 90:
            step_sec = 1.0
            major_mult = 5
        elif self.px_per_sec < 180:
            step_sec = 0.5
            major_mult = 2
        else:
            step_sec = 0.2
            major_mult = 5

        start_idx = max(0, int((vis_x1 / self.px_per_sec) // step_sec) - 1)
        end_idx = int((vis_x2 / self.px_per_sec) // step_sec) + 2

        for idx in range(start_idx, end_idx):
            curr_t = idx * step_sec
            x = int(round(curr_t * self.px_per_sec))
            is_major = (idx % major_mult) == 0

            tick_h = 10 if is_major else 5
            painter.setPen(QPen(QColor("#6C7687" if is_major else "#384050"), 1))
            painter.drawLine(x, RULER_HEIGHT - tick_h, x, RULER_HEIGHT)

            if is_major:
                mins = int(curr_t // 60)
                secs = int(curr_t % 60)
                frac = curr_t - int(curr_t)
                if step_sec < 1.0 and frac > 1e-3:
                    txt = f"{mins:02d}:{secs:02d}.{int(round(frac * 10))}"
                else:
                    txt = f"{mins:02d}:{secs:02d}"
                painter.setPen(QColor("#9CA6B8"))
                painter.drawText(x + 4, RULER_HEIGHT - 8, txt)

        # İşaretçileri (Markers) çiz
        for m_item in self.engine.project.markers:
            if isinstance(m_item, dict):
                m_time = float(m_item.get("time", 0.0))
                m_label = str(m_item.get("label", ""))
                m_col = str(m_item.get("color", "#F39C12"))
            else:
                m_time, m_label, m_col = m_item[0], m_item[1], (m_item[2] if len(m_item) > 2 else "#F39C12")
            mx = int(m_time * self.px_per_sec)
            if mx < vis_x1 - 40 or mx > vis_x2 + 40:
                continue
            mc = QColor(m_col or "#F39C12")
            painter.setPen(QPen(mc, 1.5))
            painter.drawLine(mx, 2, mx, RULER_HEIGHT)
            marker_poly = QPainterPath()
            marker_poly.moveTo(mx - 5, 2)
            marker_poly.lineTo(mx + 5, 2)
            marker_poly.lineTo(mx, 10)
            marker_poly.closeSubpath()
            painter.fillPath(marker_poly, QBrush(mc))
            painter.setPen(QColor("#F8FAFC"))
            painter.drawText(mx + 6, 11, m_label[:14])

    def _draw_track(
        self,
        painter: QPainter,
        track: TimelineTrack,
        y: int,
        width: int,
        vis_x1: int = 0,
        vis_x2: Optional[int] = None,
    ):
        if vis_x2 is None:
            vis_x2 = width
        if track.is_locked:
            bg_col = QColor("#101114")
        elif track.is_muted:
            bg_col = QColor("#131417")
        else:
            bg_col = QColor("#151619")

        painter.fillRect(vis_x1, y, max(1, vis_x2 - vis_x1), TRACK_HEIGHT - 2, bg_col)
        painter.setPen(QPen(QColor("#28292E"), 1))
        painter.drawLine(vis_x1, y + TRACK_HEIGHT - 2, vis_x2, y + TRACK_HEIGHT - 2)

        for clip in track.clips:
            x1 = int(clip.timeline_in * self.px_per_sec)
            x2 = int(clip.timeline_out * self.px_per_sec)
            # Görünür alan dışındaki klipleri çizme (Frustum Culling)
            if x2 < vis_x1 or x1 > vis_x2:
                continue

            cw = max(6, x2 - x1)
            ch = TRACK_HEIGHT - 8
            cy = y + 3

            rect = QRectF(x1, cy, cw, ch)
            path = QPainterPath()
            path.addRoundedRect(rect, 5, 5)

            is_selected = clip.id == self.selected_clip_id
            base_col = QColor(clip.color)
            if clip.is_muted or track.is_muted:
                base_col = base_col.darker(175)
            else:
                base_col = base_col.darker(115)

            painter.save()
            painter.setClipPath(path)
            painter.fillPath(path, QBrush(base_col))

            # Video/Görsel klibi ise küçük resim şeridi göster
            if clip.clip_type in (ClipType.VIDEO, ClipType.IMAGE) and clip.media_path and cw > 28:
                thumb = self._get_clip_thumb(clip.media_path)
                if thumb is not None and not thumb.isNull():
                    painter.setOpacity(0.42 if not (clip.is_muted or track.is_muted) else 0.2)
                    tile_w = max(44, int(ch * 1.6))
                    tx = max(x1, int(vis_x1 // tile_w) * tile_w)
                    while tx < min(x1 + cw, vis_x2):
                        painter.drawPixmap(
                            QRectF(tx, cy, tile_w, ch),
                            thumb,
                            QRectF(0, 0, thumb.width(), thumb.height()),
                        )
                        tx += tile_w + 2
                    painter.setOpacity(1.0)

            # Ses klibi ise gerçek dalga formu çubukları ve dB kazanç çizgisi çiz
            if clip.clip_type == ClipType.AUDIO and cw > 16:
                peaks = self._get_clip_peaks(clip.media_path)
                wf_col = QColor(clip.color).lighter(155)
                if clip.is_muted or track.is_muted:
                    wf_col = QColor("#555D6B")
                painter.setPen(QPen(wf_col, 1.3))
                bar_step = 3
                n_bars = max(1, int((cw - 8) // bar_step))
                bottom_y = cy + ch - 3
                max_bar_h = ch - 18
                vol_scale = min(1.5, max(0.1, float(clip.volume)))
                start_bi = max(0, int((vis_x1 - x1) // bar_step))
                end_bi = min(n_bars, int((vis_x2 - x1) // bar_step) + 1)
                for bi in range(start_bi, end_bi):
                    bx = x1 + 4 + bi * bar_step
                    if peaks is not None and len(peaks) > 0:
                        p_idx = min(len(peaks) - 1, int((bi / float(n_bars)) * len(peaks)))
                        amp = float(peaks[p_idx]) * vol_scale
                    else:
                        amp = (0.35 + 0.25 * abs(math.sin(bi * 0.45))) * vol_scale
                    bh = max(2.0, min(max_bar_h, amp * max_bar_h))
                    painter.drawLine(int(bx), int(bottom_y), int(bx), int(bottom_y - bh))

                # Yatay dB seviye çizgisi ve rozeti (CapCut Style)
                norm_v = min(1.5, max(0.0, float(clip.volume))) / 1.5
                line_y = cy + ch - 4 - norm_v * (ch - 14)
                is_vol_active = (self._hover_volume_clip_id == clip.id) or (
                    self._drag_mode == "volume" and self._drag_clip_id == clip.id
                )

                if is_vol_active:
                    line_pen = QPen(QColor("#38BDF8"), 2.0)
                elif is_selected:
                    line_pen = QPen(QColor("#FFFFFF"), 1.5)
                else:
                    line_pen = QPen(QColor("#94A3B8"), 1.2)
                painter.setPen(line_pen)
                painter.drawLine(int(x1 + 2), int(line_y), int(x1 + cw - 2), int(line_y))

                # Tutamaç dairesi (Hover veya Sürükleme esnasında)
                if is_vol_active and cw > 24:
                    hx = self._hover_volume_x if (self._hover_volume_clip_id == clip.id and self._hover_volume_x > 0) else (x1 + cw / 2)
                    hx = max(x1 + 10, min(x1 + cw - 10, hx))
                    painter.setPen(QPen(QColor("#38BDF8"), 2))
                    painter.setBrush(QBrush(QColor("#FFFFFF")))
                    painter.drawEllipse(QPointF(hx, line_y), 4.0, 4.0)

                # Floating dB Rozeti (Tooltip Pill)
                if (is_selected or is_vol_active) and cw > 35:
                    if clip.volume <= 0.001:
                        db_txt = "Susturuldu"
                    else:
                        db_val = 20.0 * math.log10(max(1e-4, float(clip.volume)))
                        db_txt = f"{db_val:+.1f} dB"

                    if self._drag_mode == "volume" and self._drag_clip_id == clip.id:
                        # Sürüklerken fare konumuna eşlik eden CapCut tarzı yüzen rozet
                        bx = max(x1 + 4, min(x1 + cw - 74, (self._hover_volume_x or (x1 + cw / 2)) - 35))
                        by = max(cy + 2, line_y - 24)
                        badge_rect = QRectF(bx, by, 72, 19)
                        painter.fillRect(badge_rect, QColor(12, 14, 18, 240))
                        painter.setPen(QPen(QColor("#38BDF8"), 1.5))
                        painter.drawRoundedRect(badge_rect, 4, 4)
                        painter.setPen(QColor("#38BDF8"))
                        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, db_txt)
                    else:
                        badge_rect = QRectF(x1 + cw - 56, cy + 3, 52, 15)
                        painter.fillRect(badge_rect, QColor(15, 17, 22, 210))
                        painter.setPen(QColor("#38BDF8"))
                        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, db_txt)

            # Üst bilgi şeridi
            header_strip = QRectF(x1, cy, cw, 18)
            painter.fillRect(header_strip, QColor(12, 14, 18, 135))

            # Klip İsim ve Süre Etiketi
            painter.setPen(QColor("#FFFFFF" if not (clip.is_muted or track.is_muted) else "#8892A0"))
            painter.setFont(
                QFont("Segoe UI", 8, QFont.Weight.DemiBold if is_selected else QFont.Weight.Normal)
            )
            name_txt = clip.name
            if clip.clip_type == ClipType.SUBTITLE and clip.text_content:
                name_txt = f'"{clip.text_content[:26]}"'
            label_str = f"{name_txt}  ({clip.duration:.1f}s)" if cw > 85 else name_txt
            painter.drawText(
                QRectF(x1 + 6, cy + 1, max(10, cw - 12), 16),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                label_str,
            )
            painter.restore()

            # Kenar Çizgisi
            if self._tool_mode == "blade" and clip.id == self._blade_hover_clip_id:
                border_col = QColor("#E53945")
                pen_w = 2.0
            else:
                border_col = QColor("#F8FAFC") if is_selected else base_col.lighter(135)
                pen_w = 2.0 if is_selected else 1.0
            painter.setPen(QPen(border_col, pen_w))
            painter.drawPath(path)

            # Seçili klipte sol/sağ kırpma tutamaçları
            if is_selected and cw > 20:
                handle_col = QColor("#F8FAFC")
                painter.fillRect(QRectF(x1, cy + 6, 3, ch - 12), handle_col)
                painter.fillRect(QRectF(x1 + cw - 3, cy + 6, 3, ch - 12), handle_col)

    def _draw_blade_preview(self, painter: QPainter, height: int) -> None:
        if self._blade_hover_x is None:
            return
        bx = int(round(self._blade_hover_x))
        pen = QPen(QColor("#E53945"), 1.8, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(bx, RULER_HEIGHT, bx, height)

        cut_t = self.x_to_time(self._blade_hover_x)
        badge_rect = QRectF(bx + 4, RULER_HEIGHT + 2, 68, 16)
        painter.fillRect(badge_rect, QColor(229, 57, 69, 225))
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"✂ {cut_t:.2f}s")

    def _draw_playhead(self, painter: QPainter, height: int):
        x = int(self.engine.project.playhead_pos * self.px_per_sec)

        # Kırmızı-beyaz CapCut/VOX Studio imleç çizgisi
        painter.setPen(QPen(QColor("#E53945"), 1.8))
        painter.drawLine(x, 0, x, height)

        head = QPainterPath()
        head.addRoundedRect(QRectF(x - 6, 1, 12, 11), 2, 2)
        head.moveTo(x - 5, 11)
        head.lineTo(x + 5, 11)
        head.lineTo(x, 17)
        head.closeSubpath()
        painter.fillPath(head, QBrush(QColor("#E53945")))
        painter.setPen(QPen(QColor("#FFFFFF"), 1.0))
        painter.drawPath(head)

    # ------------------------------------------------------------- Fare Etkileşimi
    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        mx, my = pos.x(), pos.y()

        if my <= RULER_HEIGHT or (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._is_dragging_playhead = True
            t = max(0.0, mx / self.px_per_sec)
            self.engine.project.playhead_pos = self.engine.snap_position(t)
            self.playheadMoved.emit(self.engine.project.playhead_pos)
            self.update()
            return

        clicked_clip, track, edge = self._hit_test_clip(mx, my)

        # KESME ARACI (BLADE TOOL) MODU: Tıklanan klibi tam tıklama noktasından böl
        if event.button() == Qt.MouseButton.LeftButton and self._tool_mode == "blade":
            if clicked_clip and track and not track.is_locked:
                raw_t = self.x_to_time(mx)
                cut_t = self.engine.snap_position(raw_t, exclude_clip_id=clicked_clip.id)
                if not (clicked_clip.timeline_in + 0.04 < cut_t < clicked_clip.timeline_out - 0.04):
                    cut_t = raw_t
                if clicked_clip.timeline_in + 0.04 < cut_t < clicked_clip.timeline_out - 0.04:
                    res = self.engine.split_clip(clicked_clip.id, cut_t)
                    if res:
                        _left_clip, right_clip = res
                        self.selected_clip_id = right_clip.id
                        self.engine.project.playhead_pos = cut_t
                        self.playheadMoved.emit(cut_t)
                        self.clipSelected.emit(right_clip)
                        self.clipModified.emit(right_clip)
                self.update()
                return

        if clicked_clip and track:
            self.selected_clip_id = clicked_clip.id
            self.clipSelected.emit(clicked_clip)
            if track.is_locked:
                self._drag_mode = None
                self.update()
                return

            self._drag_clip_id = clicked_clip.id
            self._drag_start_x = mx
            self._drag_start_y = my
            self._drag_orig_in = clicked_clip.timeline_in
            self._drag_orig_out = clicked_clip.timeline_out
            self._drag_orig_volume = float(clicked_clip.volume)
            self._snapshot_taken_for_drag = False

            if edge == "volume":
                self._drag_mode = "volume"
                self.engine.snapshot()
                self._snapshot_taken_for_drag = True
                self._hover_volume_clip_id = clicked_clip.id
                self._hover_volume_x = mx
                self._hover_volume_y = my
                self.update()
                return
            elif edge == "left":
                self._drag_mode = "trim_left"
            elif edge == "right":
                self._drag_mode = "trim_right"
            else:
                self._drag_mode = "move"
        else:
            self.selected_clip_id = None
            self.clipSelected.emit(None)
            self._drag_mode = None
            self._is_dragging_playhead = True
            t = max(0.0, mx / self.px_per_sec)
            self.engine.project.playhead_pos = self.engine.snap_position(t)
            self.playheadMoved.emit(self.engine.project.playhead_pos)

        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        mx, my = pos.x(), pos.y()

        if self._is_dragging_playhead:
            t = max(0.0, mx / self.px_per_sec)
            self.engine.project.playhead_pos = self.engine.snap_position(t)
            self.playheadMoved.emit(self.engine.project.playhead_pos)
            self.update()
            return

        if self._tool_mode == "blade":
            clip, track, _ = self._hit_test_clip(mx, my)
            self._blade_hover_x = mx if my > RULER_HEIGHT else None
            self._blade_hover_clip_id = clip.id if (clip and track and not track.is_locked) else None
            self.setCursor(
                Qt.CursorShape.SplitHCursor if self._blade_hover_clip_id else Qt.CursorShape.CrossCursor
            )
            self.update()
            return

        if self._drag_mode and self._drag_clip_id:
            clip = self.get_selected_clip()
            if not clip:
                return

            if self._drag_mode == "volume":
                ch = TRACK_HEIGHT - 8
                vol_range_px = float(max(15, ch - 14))
                # Yukarı çekmek sesi artırır (delta_y pozitif)
                delta_y = self._drag_start_y - my
                vol_change = (delta_y / vol_range_px) * 1.5
                new_vol = max(0.0, min(2.0, self._drag_orig_volume + vol_change))
                clip.volume = new_vol
                self._hover_volume_clip_id = clip.id
                self._hover_volume_x = mx
                self._hover_volume_y = my
                self.clipModified.emit(clip)
                self.update()
                return

            delta_sec = (mx - self._drag_start_x) / self.px_per_sec
            if abs(mx - self._drag_start_x) < 1.5 and not self._snapshot_taken_for_drag:
                return

            if not self._snapshot_taken_for_drag:
                self.engine.snapshot()
                self._snapshot_taken_for_drag = True

            if self._drag_mode == "move":
                new_in = max(0.0, self._drag_orig_in + delta_sec)
                snapped_in = self.engine.snap_position(new_in, exclude_clip_id=clip.id)

                # Dikey kanal değiştirme kontrolü
                target_track_id = None
                if my > RULER_HEIGHT:
                    t_idx = int((my - RULER_HEIGHT) // TRACK_HEIGHT)
                    if 0 <= t_idx < len(self.engine.project.tracks):
                        cand_track = self.engine.project.tracks[t_idx]
                        if not cand_track.is_locked:
                            if (
                                clip.clip_type in (ClipType.VIDEO, ClipType.IMAGE)
                                and cand_track.track_type == TrackType.VIDEO
                            ):
                                target_track_id = cand_track.id
                            elif clip.clip_type == ClipType.AUDIO and cand_track.track_type in (
                                TrackType.DUBBING,
                                TrackType.DIALOGUE,
                                TrackType.MUSIC,
                                TrackType.AMBIENT,
                            ):
                                target_track_id = cand_track.id
                            elif (
                                clip.clip_type == ClipType.SUBTITLE
                                and cand_track.track_type == TrackType.SUBTITLE
                            ):
                                target_track_id = cand_track.id

                self.engine.move_clip(
                    clip.id, snapped_in, target_track_id=target_track_id, record_snapshot=False
                )
            elif self._drag_mode == "trim_left":
                new_in = max(0.0, self._drag_orig_in + delta_sec)
                snapped_in = self.engine.snap_position(new_in, exclude_clip_id=clip.id)
                self.engine.trim_clip(clip.id, new_timeline_in=snapped_in, record_snapshot=False)
            elif self._drag_mode == "trim_right":
                new_out = max(clip.timeline_in + 0.1, self._drag_orig_out + delta_sec)
                snapped_out = self.engine.snap_position(new_out, exclude_clip_id=clip.id)
                self.engine.trim_clip(clip.id, new_timeline_out=snapped_out, record_snapshot=False)

            self.clipModified.emit(clip)
            self.update()
            return

        hit_clip, _, edge = self._hit_test_clip(mx, my)
        if edge in ("left", "right"):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            if self._hover_volume_clip_id is not None:
                self._hover_volume_clip_id = None
                self.update()
        elif edge == "volume":
            self.setCursor(Qt.CursorShape.SplitVCursor)
            self._hover_volume_clip_id = hit_clip.id if hit_clip else None
            self._hover_volume_x = mx
            self._hover_volume_y = my
            self.update()
        elif edge == "body":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            if self._hover_volume_clip_id is not None:
                self._hover_volume_clip_id = None
                self.update()
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if self._hover_volume_clip_id is not None:
                self._hover_volume_clip_id = None
                self.update()

    def leaveEvent(self, event):
        if self._blade_hover_x is not None or self._blade_hover_clip_id is not None:
            self._blade_hover_x = None
            self._blade_hover_clip_id = None
        if self._hover_volume_clip_id is not None:
            self._hover_volume_clip_id = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._is_dragging_playhead = False
        self._drag_mode = None
        self._drag_clip_id = None
        self._snapshot_taken_for_drag = False
        self._volume_tooltip_pos = None
        if self._tool_mode == "blade":
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.18 if delta > 0 else (1.0 / 1.18)
            anchor_t = self.x_to_time(event.position().x())
            target_px = max(MIN_ZOOM_PX, min(MAX_ZOOM_PX, self.px_per_sec * factor))
            self.zoomRequested.emit(target_px, anchor_t)
            event.accept()
        else:
            super().wheelEvent(event)

    def _hit_test_clip(
        self, mx: float, my: float
    ) -> Tuple[Optional[TimelineClip], Optional[TimelineTrack], Optional[str]]:
        if my < RULER_HEIGHT:
            return None, None, None

        track_idx = int((my - RULER_HEIGHT) // TRACK_HEIGHT)
        if track_idx < 0 or track_idx >= len(self.engine.project.tracks):
            return None, None, None

        track = self.engine.project.tracks[track_idx]
        for clip in track.clips:
            x1 = clip.timeline_in * self.px_per_sec
            x2 = clip.timeline_out * self.px_per_sec

            if x1 <= mx <= x2:
                if abs(mx - x1) <= 6:
                    return clip, track, "left"
                if abs(mx - x2) <= 6:
                    return clip, track, "right"

                # Ses klibi için yatay dB kazanç çizgisi kontrolü (CapCut Style)
                if clip.clip_type == ClipType.AUDIO:
                    cy = RULER_HEIGHT + (track_idx * TRACK_HEIGHT) + 3
                    ch = TRACK_HEIGHT - 8
                    norm_v = min(1.5, max(0.0, float(clip.volume))) / 1.5
                    line_y = cy + ch - 4 - norm_v * (ch - 14)
                    if abs(my - line_y) <= 6 and (x1 + 6 <= mx <= x2 - 6):
                        return clip, track, "volume"

                return clip, track, "body"

        return None, track, None

    # -------------------------------------------------------- Sağ Tık Menüsü
    def contextMenuEvent(self, event):
        mx, my = event.pos().x(), event.pos().y()
        clip, track, _ = self._hit_test_clip(mx, my)
        playhead_t = self.engine.project.playhead_pos
        click_t = self.x_to_time(mx)

        menu = QMenu(self)

        if clip:
            split_click_act = menu.addAction(f"Tıklanan Noktadan Böl ({click_t:.2f}s)")
            split_click_act.setEnabled(clip.timeline_in + 0.04 < click_t < clip.timeline_out - 0.04)
            split_click_act.triggered.connect(lambda: self.engine.split_clip(clip.id, click_t))

            split_act = menu.addAction(f"Playhead'den Böl ({playhead_t:.2f}s)")
            split_act.setEnabled(clip.timeline_in + 0.04 < playhead_t < clip.timeline_out - 0.04)
            split_act.triggered.connect(lambda: self.engine.split_clip(clip.id, playhead_t))

            mute_act = menu.addAction(
                "Klibi Sustur (Mute)" if not clip.is_muted else "Klip Sesini Aç"
            )
            mute_act.triggered.connect(lambda: self._toggle_clip_mute(clip))

            dup_act = menu.addAction("Çoğalt (Duplicate)")
            dup_act.triggered.connect(lambda: self._duplicate_clip(clip, track))

            menu.addSeparator()
            del_act = menu.addAction("Klibi Sil (Delete)")
            del_act.triggered.connect(lambda: self.engine.remove_clip(clip.id))
        else:
            add_mark = menu.addAction(f"İşaretçi Ekle ({playhead_t:.2f}s)")
            add_mark.triggered.connect(lambda: self._add_marker(playhead_t))
            if self.engine.project.markers:
                clr_mark = menu.addAction("Tüm İşaretçileri Temizle")
                clr_mark.triggered.connect(self._clear_markers)

        menu.exec(event.globalPos())

    def _toggle_clip_mute(self, clip: TimelineClip):
        self.engine.snapshot()
        clip.is_muted = not clip.is_muted
        self.update()

    def _duplicate_clip(self, clip: TimelineClip, track: Optional[TimelineTrack]):
        if not track or track.is_locked:
            return
        new_clip = clip.clone()
        new_clip.timeline_in = clip.timeline_out + 0.05
        new_clip.timeline_out = new_clip.timeline_in + clip.duration
        self.engine.add_clip(track.id, new_clip)

    def _add_marker(self, t: float):
        self.engine.project.markers.append(
            (t, f"İşaret #{len(self.engine.project.markers) + 1}", "#F39C12")
        )
        self.update()

    def _clear_markers(self):
        self.engine.project.markers.clear()
        self.update()


class TrackHeaderWidget(QFrame):
    """Kanal başlığı: İsim, Kilit (Lock), Mute (M), Solo (S) ve Seviye kontrolleri."""

    def __init__(self, track: TimelineTrack, engine: TimelineEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.track = track
        self.engine = engine
        self.setFixedHeight(TRACK_HEIGHT)
        self.setObjectName("CardFlat")
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        badge = QLabel()
        badge.setFixedSize(5, TRACK_HEIGHT - 16)
        badge.setStyleSheet(f"background-color: {self.track.color}; border-radius: 2px;")
        layout.addWidget(badge)

        info_col = QVBoxLayout()
        info_col.setContentsMargins(0, 0, 0, 0)
        info_col.setSpacing(2)

        name_lbl = QLabel(self.track.name)
        name_lbl.setStyleSheet("font-weight: 600; font-size: 11px; color: #F2F4F8;")
        name_lbl.setToolTip(f"Kanal: {self.track.name}")
        info_col.addWidget(name_lbl)

        if self.track.track_type not in (TrackType.VIDEO, TrackType.SUBTITLE):
            vol_row = QHBoxLayout()
            vol_row.setContentsMargins(0, 0, 0, 0)
            vol_row.setSpacing(4)
            self.vol_slider = QSlider(Qt.Orientation.Horizontal)
            self.vol_slider.setRange(0, 150)
            self.vol_slider.setValue(int(self.track.volume * 100))
            self.vol_slider.setFixedHeight(14)
            self.vol_slider.setToolTip(f"Kanal ses seviyesi: %{int(self.track.volume * 100)}")
            self.vol_slider.valueChanged.connect(self._on_vol)
            vol_row.addWidget(self.vol_slider)
            info_col.addLayout(vol_row)

        layout.addLayout(info_col, 1)

        # Kilit (Lock) Butonu
        self.lock_btn = QPushButton()
        self.lock_btn.setProperty("compact", "true")
        self.lock_btn.setCheckable(True)
        self.lock_btn.setChecked(self.track.is_locked)
        self.lock_btn.setFixedSize(24, 24)
        self.lock_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lock_btn.setToolTip("Kanalı Kilitle / Kilidi Aç (Yanlışlıkla taşımayı önler)")
        self._apply_lock_style(self.track.is_locked)
        self.lock_btn.toggled.connect(self._on_lock)
        layout.addWidget(self.lock_btn)

        # Mute / Solo
        self.mute_btn = QPushButton("M")
        self.mute_btn.setProperty("compact", "true")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setChecked(self.track.is_muted)
        self.mute_btn.setFixedSize(24, 24)
        self.mute_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mute_btn.setToolTip("Kanalı Sustur (Mute)")
        self._apply_mute_style(self.track.is_muted)
        self.mute_btn.toggled.connect(self._on_mute)
        layout.addWidget(self.mute_btn)

        self.solo_btn = QPushButton("S")
        self.solo_btn.setProperty("compact", "true")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setChecked(self.track.is_solo)
        self.solo_btn.setFixedSize(24, 24)
        self.solo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.solo_btn.setToolTip("Yalnızca Bu Kanalı Dinle (Solo)")
        self._apply_solo_style(self.track.is_solo)
        self.solo_btn.toggled.connect(self._on_solo)
        layout.addWidget(self.solo_btn)

    def _apply_lock_style(self, locked: bool):
        self.lock_btn.setIcon(
            icons.icon("lock" if locked else "unlock", size=12, color="#F39C12" if locked else "#8892A0")
        )
        if locked:
            self.lock_btn.setStyleSheet("background-color: #2C2518; padding: 0px; border-radius: 4px;")
        else:
            self.lock_btn.setStyleSheet("padding: 0px; border-radius: 4px;")

    def _apply_mute_style(self, checked: bool):
        if checked:
            self.mute_btn.setStyleSheet(
                "background-color: #C0392B; color: #FFFFFF; font-weight: bold; padding: 0px; border-radius: 4px;"
            )
        else:
            self.mute_btn.setStyleSheet("padding: 0px; font-weight: bold; border-radius: 4px;")

    def _apply_solo_style(self, checked: bool):
        if checked:
            self.solo_btn.setStyleSheet(
                "background-color: #F1C40F; color: #111111; font-weight: bold; padding: 0px; border-radius: 4px;"
            )
        else:
            self.solo_btn.setStyleSheet("padding: 0px; font-weight: bold; border-radius: 4px;")

    def _on_lock(self, checked: bool):
        self.track.is_locked = checked
        self._apply_lock_style(checked)
        self.engine.canvas_update_only = True
        self.engine._notify()

    def _on_vol(self, val: int):
        self.track.volume = val / 100.0
        self.vol_slider.setToolTip(f"Kanal ses seviyesi: %{val}")
        self.engine.canvas_update_only = True
        self.engine._notify()

    def _on_mute(self, checked: bool):
        self.engine.snapshot()
        self.track.is_muted = checked
        self._apply_mute_style(checked)
        self.engine._notify()

    def _on_solo(self, checked: bool):
        self.engine.snapshot()
        self.track.is_solo = checked
        self._apply_solo_style(checked)
        self.engine._notify()


class MultiTrackTimeline(QWidget):
    """Tüm timeline bileşenini (CapCut tarzı orta araç çubuğu, başlıklar, cetvel ve tuval) toplayan widget."""

    playheadMoved = Signal(float)
    clipSelected = Signal(object)
    clipModified = Signal(object)
    filesDropped = Signal(list)
    recordVoiceoverRequested = Signal()
    toolModeChanged = Signal(str)

    def __init__(self, engine: TimelineEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine = engine
        self.setAcceptDrops(True)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # CapCut Tarzı Orta Araç Çubuğu (Seçim V, Kes B, Böl, Çoğalt, Sil, İşaretçi, Mikrofon, Mıknatıs, Zoom)
        top_bar = QFrame()
        top_bar.setFixedHeight(38)
        top_bar.setStyleSheet(
            "background-color: #101114; border-bottom: 1px solid #28292E; border-top: 1px solid #28292E;"
        )
        tb_layout = QHBoxLayout(top_bar)
        tb_layout.setContentsMargins(10, 3, 10, 3)
        tb_layout.setSpacing(5)

        self.undo_btn = QPushButton()
        self.undo_btn.setProperty("compact", "true")
        self.undo_btn.setIcon(icons.icon("undo", size=14))
        self.undo_btn.setFixedSize(28, 26)
        self.undo_btn.setToolTip("Geri Al (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.engine.undo)
        tb_layout.addWidget(self.undo_btn)

        self.redo_btn = QPushButton()
        self.redo_btn.setProperty("compact", "true")
        self.redo_btn.setIcon(icons.icon("redo", size=14))
        self.redo_btn.setFixedSize(28, 26)
        self.redo_btn.setToolTip("İleri Al (Ctrl+Y)")
        self.redo_btn.clicked.connect(self.engine.redo)
        tb_layout.addWidget(self.redo_btn)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.VLine)
        sep0.setStyleSheet("color: #28292E;")
        tb_layout.addWidget(sep0)

        # Seçim (V) ve Kesme / Jilet (B) mod butonları
        self.select_tool_btn = QPushButton(" Seçim (V)")
        self.select_tool_btn.setProperty("compact", "true")
        self.select_tool_btn.setCheckable(True)
        self.select_tool_btn.setChecked(True)
        self.select_tool_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.select_tool_btn.setToolTip("Seçim ve Taşıma Aracı (Kısayol: V)")
        self.select_tool_btn.clicked.connect(lambda: self.set_tool_mode("select"))
        tb_layout.addWidget(self.select_tool_btn)

        self.blade_tool_btn = QPushButton(" Kes (B)")
        self.blade_tool_btn.setProperty("compact", "true")
        self.blade_tool_btn.setCheckable(True)
        self.blade_tool_btn.setChecked(False)
        self.blade_tool_btn.setIcon(icons.icon("scissors", size=13, color="#E53945"))
        self.blade_tool_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.blade_tool_btn.setToolTip(
            "Kesme / Jilet Aracı (Kısayol: B) — Kliplerin üzerine tıklayarak istediğiniz noktadan bölün"
        )
        self.blade_tool_btn.clicked.connect(lambda: self.set_tool_mode("blade"))
        tb_layout.addWidget(self.blade_tool_btn)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: #28292E;")
        tb_layout.addWidget(sep1)

        self.split_btn = QPushButton(" İmleçte Böl (Ctrl+K)")
        self.split_btn.setProperty("compact", "true")
        self.split_btn.setIcon(icons.icon("scissors", size=13))
        self.split_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.split_btn.setToolTip("Seçili klibi veya imleç altındaki klibi playhead konumundan ikiye böler (Ctrl+K)")
        self.split_btn.clicked.connect(self._split_selected)
        tb_layout.addWidget(self.split_btn)

        self.dup_btn = QPushButton(" Çoğalt")
        self.dup_btn.setProperty("compact", "true")
        self.dup_btn.setIcon(icons.icon("copy", size=13))
        self.dup_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.dup_btn.setToolTip("Seçili klibi çoğaltır (Ctrl+D)")
        self.dup_btn.clicked.connect(self._duplicate_selected)
        tb_layout.addWidget(self.dup_btn)

        self.delete_btn = QPushButton(" Sil")
        self.delete_btn.setProperty("compact", "true")
        self.delete_btn.setIcon(icons.icon("trash", size=13))
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setToolTip("Seçili klibi timeline'dan kaldırır (Delete)")
        self.delete_btn.clicked.connect(self._delete_selected)
        tb_layout.addWidget(self.delete_btn)

        self.marker_btn = QPushButton(" İşaretçi (M)")
        self.marker_btn.setProperty("compact", "true")
        self.marker_btn.setIcon(icons.icon("bookmark", size=13))
        self.marker_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.marker_btn.setToolTip("Playhead konumuna sahne/replik işaretçisi ekler (M)")
        self.marker_btn.clicked.connect(lambda: self.canvas._add_marker(self.engine.project.playhead_pos))
        tb_layout.addWidget(self.marker_btn)

        tb_layout.addStretch(1)

        self.mic_quick_btn = QPushButton(" Mikrofonla Dublaj")
        self.mic_quick_btn.setProperty("compact", "true")
        self.mic_quick_btn.setIcon(icons.icon("mic", size=13, color="#F87171"))
        self.mic_quick_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic_quick_btn.setToolTip("Playhead konumundan itibaren canlı seslendirme kaydı başlat/durdur")
        self.mic_quick_btn.clicked.connect(self.recordVoiceoverRequested.emit)
        tb_layout.addWidget(self.mic_quick_btn)

        self.snap_btn = QPushButton(" Mıknatıs")
        self.snap_btn.setProperty("compact", "true")
        self.snap_btn.setCheckable(True)
        self.snap_btn.setChecked(self.engine.magnetic_snap_enabled)
        self.snap_btn.setIcon(icons.icon("magnet", size=13, color="#38BDF8"))
        self.snap_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.snap_btn.setToolTip("Mıknatıslı Hizalama (Klip kenarlarına ve imlece yapışma)")
        self.snap_btn.toggled.connect(self._on_snap_toggled)
        tb_layout.addWidget(self.snap_btn)

        # Bölüm 4 — Timeline Yakınlaştırma / Uzaklaştırma (Zoom) Kontrolleri
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setProperty("compact", "true")
        self.zoom_out_btn.setFixedSize(24, 24)
        self.zoom_out_btn.setToolTip("Timeline'ı Uzaklaştır (Ctrl + -)")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        tb_layout.addWidget(self.zoom_out_btn)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(int(MIN_ZOOM_PX), int(MAX_ZOOM_PX))
        self.zoom_slider.setValue(60)
        self.zoom_slider.setFixedWidth(100)
        self.zoom_slider.setToolTip("Timeline yatay yakınlaştırma ölçeği (Ctrl + Fare Tekerleği)")
        self.zoom_slider.valueChanged.connect(lambda v: self.set_zoom_anchored(float(v)))
        tb_layout.addWidget(self.zoom_slider)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setProperty("compact", "true")
        self.zoom_in_btn.setFixedSize(24, 24)
        self.zoom_in_btn.setToolTip("Timeline'ı Yakınlaştır (Ctrl + +)")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        tb_layout.addWidget(self.zoom_in_btn)

        self.zoom_fit_btn = QPushButton("Sığdır")
        self.zoom_fit_btn.setProperty("compact", "true")
        self.zoom_fit_btn.setToolTip("Projeyi Pencereye Sığdır (Tüm klipleri tek ekranda göster)")
        self.zoom_fit_btn.clicked.connect(self.fit_to_project)
        tb_layout.addWidget(self.zoom_fit_btn)

        self.zoom_focus_btn = QPushButton("Odakla")
        self.zoom_focus_btn.setProperty("compact", "true")
        self.zoom_focus_btn.setToolTip("Seçili Klibe Odaklan")
        self.zoom_focus_btn.clicked.connect(self.focus_selected_clip)
        tb_layout.addWidget(self.zoom_focus_btn)

        root.addWidget(top_bar)

        # Ana Gövde: Sol Dikey Araç Çubuğu + Başlıklar + Sağ Scroll Tuval
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Dikey Araç Çubuğu (Vertical Tool Rail - Image 1 ile birebir uyumlu)
        self.vertical_tool_rail = QWidget()
        self.vertical_tool_rail.setFixedWidth(32)
        self.vertical_tool_rail.setStyleSheet(
            "background-color: #0E0F12; border-right: 1px solid #28292E;"
        )
        tr_lay = QVBoxLayout(self.vertical_tool_rail)
        tr_lay.setContentsMargins(3, RULER_HEIGHT + 4, 3, 4)
        tr_lay.setSpacing(4)

        self.rail_select_btn = QPushButton("➤")
        self.rail_select_btn.setFixedSize(26, 26)
        self.rail_select_btn.setCheckable(True)
        self.rail_select_btn.setChecked(True)
        self.rail_select_btn.setToolTip("Seçim Aracı (V)")
        self.rail_select_btn.clicked.connect(lambda: self.set_tool_mode("select"))
        tr_lay.addWidget(self.rail_select_btn)

        self.rail_blade_btn = QPushButton("✂")
        self.rail_blade_btn.setFixedSize(26, 26)
        self.rail_blade_btn.setCheckable(True)
        self.rail_blade_btn.setToolTip("Jilet / Kesme Aracı (B)")
        self.rail_blade_btn.clicked.connect(lambda: self.set_tool_mode("blade"))
        tr_lay.addWidget(self.rail_blade_btn)

        self.rail_slip_btn = QPushButton("⇄")
        self.rail_slip_btn.setFixedSize(26, 26)
        self.rail_slip_btn.setToolTip("Kaydırma Aracı (Y)")
        tr_lay.addWidget(self.rail_slip_btn)

        self.rail_pen_btn = QPushButton("✎")
        self.rail_pen_btn.setFixedSize(26, 26)
        self.rail_pen_btn.setToolTip("Kalem Aracı (P)")
        tr_lay.addWidget(self.rail_pen_btn)

        self.rail_zoom_btn = QPushButton("🔍")
        self.rail_zoom_btn.setFixedSize(26, 26)
        self.rail_zoom_btn.setToolTip("Projeyi Sığdır (Shift+Z)")
        self.rail_zoom_btn.clicked.connect(self.fit_to_project)
        tr_lay.addWidget(self.rail_zoom_btn)

        tr_lay.addStretch(1)
        body.addWidget(self.vertical_tool_rail)

        self.headers_col = QWidget()
        self.headers_col.setFixedWidth(HEADER_WIDTH)
        self.headers_col.setStyleSheet("background-color: #101114; border-right: 1px solid #28292E;")
        self.headers_layout = QVBoxLayout(self.headers_col)
        self.headers_layout.setContentsMargins(0, RULER_HEIGHT, 0, 0)
        self.headers_layout.setSpacing(0)
        body.addWidget(self.headers_col)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: #090A0C; border: none;")

        self.canvas = TimelineCanvas(self.engine)
        self.canvas.playheadMoved.connect(self.playheadMoved.emit)
        self.canvas.clipSelected.connect(self.clipSelected.emit)
        self.canvas.clipModified.connect(self.clipModified.emit)
        self.canvas.toolModeChanged.connect(self._on_canvas_tool_mode_changed)
        self.canvas.zoomRequested.connect(self.set_zoom_anchored)

        self.scroll.setWidget(self.canvas)
        body.addWidget(self.scroll, 1)

        root.addLayout(body, 1)

        self._update_tool_button_styles()
        self.rebuild_headers()
        self.engine.on_changed(self.rebuild_headers)

    # ------------------------------------------------- Araç Modu (Seçim V / Kes B)
    @property
    def tool_mode(self) -> str:
        return self.canvas.tool_mode

    def set_tool_mode(self, mode: str) -> None:
        self.canvas.set_tool_mode(mode)
        self._update_tool_button_styles()

    def _on_canvas_tool_mode_changed(self, mode: str) -> None:
        self._update_tool_button_styles()
        self.toolModeChanged.emit(mode)

    def _update_tool_button_styles(self) -> None:
        is_blade = self.canvas.tool_mode == "blade"
        self.select_tool_btn.blockSignals(True)
        self.blade_tool_btn.blockSignals(True)
        self.select_tool_btn.setChecked(not is_blade)
        self.blade_tool_btn.setChecked(is_blade)
        self.select_tool_btn.blockSignals(False)
        self.blade_tool_btn.blockSignals(False)

        if hasattr(self, "rail_select_btn"):
            self.rail_select_btn.blockSignals(True)
            self.rail_select_btn.setChecked(not is_blade)
            self.rail_select_btn.setStyleSheet(
                "background-color: #252A36; color: #FFFFFF; font-weight: 700; border: 1px solid #3B82F6; border-radius: 4px;"
                if not is_blade
                else ""
            )
            self.rail_select_btn.blockSignals(False)

        if hasattr(self, "rail_blade_btn"):
            self.rail_blade_btn.blockSignals(True)
            self.rail_blade_btn.setChecked(is_blade)
            self.rail_blade_btn.setStyleSheet(
                "background-color: rgba(229, 57, 69, 0.24); color: #FF4652; font-weight: 700; border: 1px solid #E53945; border-radius: 4px;"
                if is_blade
                else ""
            )
            self.rail_blade_btn.blockSignals(False)

        self.select_tool_btn.setStyleSheet(
            "background-color: #252A36; color: #FFFFFF; font-weight: 700; border: 1px solid #3B82F6; border-radius: 4px;"
            if not is_blade
            else ""
        )
        self.blade_tool_btn.setStyleSheet(
            "background-color: rgba(229, 57, 69, 0.24); color: #FF4652; font-weight: 700; border: 1px solid #E53945; border-radius: 4px;"
            if is_blade
            else ""
        )

    # ------------------------------------------------- İmleç Merkezli Zoom Sistemi
    def set_zoom_anchored(self, new_px_per_sec: float, anchor_time: Optional[float] = None) -> None:
        """
        Bölüm 4.2 — Yakınlaştırma sırasında imleç (playhead) veya mevcut görünüm merkezini korur;
        asla 00:00 başlangıç noktasına istemsiz sıçrama yapmaz.
        """
        clamped = max(MIN_ZOOM_PX, min(MAX_ZOOM_PX, float(new_px_per_sec)))
        hbar = self.scroll.horizontalScrollBar()
        viewport_w = max(240, self.scroll.viewport().width())

        if anchor_time is not None:
            anchor_t = float(anchor_time)
        else:
            ph = float(self.engine.project.playhead_pos)
            ph_x = ph * self.canvas.px_per_sec
            if hbar.value() <= ph_x <= hbar.value() + viewport_w:
                anchor_t = ph
            else:
                anchor_t = (hbar.value() + viewport_w * 0.5) / max(1.0, self.canvas.px_per_sec)

        old_x = anchor_t * self.canvas.px_per_sec
        rel_x = old_x - hbar.value()
        if rel_x < 0 or rel_x > viewport_w:
            rel_x = viewport_w * 0.5

        self.canvas.set_zoom(clamped)
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(round(clamped)))
        self.zoom_slider.blockSignals(False)

        new_x = anchor_t * self.canvas.px_per_sec
        target_scroll = max(0, int(round(new_x - rel_x)))
        hbar.setValue(target_scroll)

    def zoom_in(self) -> None:
        self.set_zoom_anchored(self.canvas.px_per_sec * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom_anchored(self.canvas.px_per_sec / 1.25)

    def fit_to_project(self) -> None:
        """Projeyi pencere genişliğine sığdırır."""
        dur = max(5.0, self.engine.project.duration)
        viewport_w = max(320, self.scroll.viewport().width() - 40)
        ideal_px = max(MIN_ZOOM_PX, min(MAX_ZOOM_PX, viewport_w / dur))
        self.set_zoom_anchored(ideal_px, anchor_time=0.0)
        self.scroll.horizontalScrollBar().setValue(0)

    def show_all_clips(self) -> None:
        self.fit_to_project()

    def focus_selected_clip(self) -> None:
        """Seçili klibi görünür alana ortalayacak şekilde yakınlaştırır."""
        clip = self.canvas.get_selected_clip()
        if not clip:
            return
        dur = max(1.0, clip.duration)
        viewport_w = max(320, self.scroll.viewport().width() - 80)
        ideal_px = max(MIN_ZOOM_PX, min(MAX_ZOOM_PX, viewport_w / (dur * 1.6)))
        mid_t = 0.5 * (clip.timeline_in + clip.timeline_out)
        self.set_zoom_anchored(ideal_px, anchor_time=mid_t)
        target_scroll = max(0, int(round(mid_t * self.canvas.px_per_sec - viewport_w * 0.5)))
        self.scroll.horizontalScrollBar().setValue(target_scroll)

    def _on_snap_toggled(self, checked: bool):
        self.engine.magnetic_snap_enabled = checked
        self.snap_btn.setIcon(
            icons.icon("magnet", size=13, color="#38BDF8" if checked else "#6C7687")
        )

    def rebuild_headers(self):
        if getattr(self.engine, "canvas_update_only", False):
            self.engine.canvas_update_only = False
            self.canvas.update()
            return

        while self.headers_layout.count():
            item = self.headers_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        for track in self.engine.project.tracks:
            hw = TrackHeaderWidget(track, self.engine)
            self.headers_layout.addWidget(hw)

        self.headers_layout.addStretch(1)
        self.canvas._update_geometry()

    def set_playhead(self, t: float, auto_scroll: bool = True, emit_seek: bool = False):
        if getattr(self.canvas, "_is_dragging_playhead", False):
            return
        self.canvas.engine.project.playhead_pos = t
        self.canvas.update()

        if auto_scroll:
            hbar = self.scroll.horizontalScrollBar()
            viewport_w = self.scroll.viewport().width()
            px_pos = self.canvas.time_to_x(t)
            curr_left = hbar.value()
            if px_pos > curr_left + viewport_w * 0.85:
                target_scroll = max(0, int(px_pos - viewport_w * 0.25))
                hbar.setValue(target_scroll)
            elif px_pos < curr_left:
                target_scroll = max(0, int(px_pos - viewport_w * 0.15))
                hbar.setValue(target_scroll)

        if emit_seek:
            self.playheadMoved.emit(t)

    def _split_selected(self) -> bool:
        """
        Seçili klibi (veya seçili klip yoksa playhead altındaki ilk kilitsiz klibi)
        playhead konumundan ikiye böler.
        """
        playhead_t = self.engine.project.playhead_pos
        clip = self.canvas.get_selected_clip()
        if clip and (clip.timeline_in + 0.04 < playhead_t < clip.timeline_out - 0.04):
            res = self.engine.split_clip(clip.id, playhead_t)
            if res:
                _l, r = res
                self.canvas.selected_clip_id = r.id
                self.clipSelected.emit(r)
                return True

        # Seçili klip yoksa veya playhead seçili klibin dışındaysa playhead altındaki klibi bul ve böl
        for track in self.engine.project.tracks:
            if track.is_locked:
                continue
            for c in list(track.clips):
                if c.timeline_in + 0.04 < playhead_t < c.timeline_out - 0.04:
                    res = self.engine.split_clip(c.id, playhead_t)
                    if res:
                        _l, r = res
                        self.canvas.selected_clip_id = r.id
                        self.clipSelected.emit(r)
                        return True
        return False

    def _duplicate_selected(self):
        clip = self.canvas.get_selected_clip()
        if clip:
            track = self.engine.find_track(clip.track_id)
            self.canvas._duplicate_clip(clip, track)

    def _delete_selected(self):
        clip = self.canvas.get_selected_clip()
        if clip:
            self.engine.remove_clip(clip.id)
            self.canvas.selected_clip_id = None
            self.clipSelected.emit(None)

    def _ripple_delete_selected(self):
        """CapCut tarzı boşluğu kapatarak silme (Shift+Delete)."""
        clip = self.canvas.get_selected_clip()
        if clip:
            self.engine.ripple_delete_clip(clip.id)
            self.canvas.selected_clip_id = None
            self.clipSelected.emit(None)

    def _copy_selected(self) -> bool:
        """Seçili klibi kopyalar (Ctrl+C)."""
        clip = self.canvas.get_selected_clip()
        if clip:
            return self.engine.copy_clip(clip.id)
        return False

    def _paste_at_playhead(self) -> Optional[TimelineClip]:
        """Panodaki klibi imleç konumuna yapıştırır (Ctrl+V)."""
        new_c = self.engine.paste_clip()
        if new_c:
            self.canvas.selected_clip_id = new_c.id
            self.clipSelected.emit(new_c)
            return new_c
        return None

    def step_frame(self, delta_frames: int = 1) -> None:
        """CapCut uyumlu kare kare ilerleme (Sol / Sağ Ok tuşları)."""
        fps = max(1.0, float(self.engine.project.fps))
        dt = delta_frames / fps
        new_pos = max(0.0, self.engine.project.playhead_pos + dt)
        self.set_playhead(new_pos, emit_seek=True)

    def step_seconds(self, delta_sec: float = 1.0) -> None:
        """CapCut uyumlu 1 saniyelik sıçrama (Shift + Sol / Sağ Ok tuşları)."""
        new_pos = max(0.0, self.engine.project.playhead_pos + delta_sec)
        self.set_playhead(new_pos, emit_seek=True)

    def jump_home(self) -> None:
        """Zaman çizelgesi başlangıcına (00:00) git (Home tuşu)."""
        self.set_playhead(0.0, emit_seek=True)

    def jump_end(self) -> None:
        """Zaman çizelgesi sonuna git (End tuşu)."""
        self.set_playhead(self.engine.project.duration, emit_seek=True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.canvas._drop_preview_rect = QRectF(
                self.engine.project.playhead_pos * self.canvas.px_per_sec,
                RULER_HEIGHT + 4,
                160,
                TRACK_HEIGHT - 8,
            )
            self.canvas.update()
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragLeaveEvent(self, event):
        self.canvas._drop_preview_rect = None
        self.canvas.update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.canvas._drop_preview_rect = None
        self.canvas.update()
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            if paths:
                self.filesDropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
