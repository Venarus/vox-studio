"""Seviye ölçer bileşeni (peak + hold + clip göstergesi)."""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from core import constants as C

_MIN_DB = -60.0


def _norm(db: float) -> float:
    if db is None or not math.isfinite(db):
        return 0.0
    return max(0.0, min(1.0, (db - _MIN_DB) / (0.0 - _MIN_DB)))


class LevelMeter(QWidget):
    """-60..0 dBFS ölçeğinde peak/hold göstergesi."""

    def __init__(
        self,
        parent=None,
        orientation: Qt.Orientation = Qt.Orientation.Horizontal,
        show_scale: bool = True,
        show_clip: bool = True,
    ):
        super().__init__(parent)
        self._orientation = orientation
        self._show_scale = show_scale
        self._show_clip = show_clip
        self._peak_db = -math.inf
        self._hold_db = -math.inf
        self._clipped = False
        self.setMinimumSize(QSize(120, 14) if orientation == Qt.Orientation.Horizontal else QSize(18, 90))

    def set_level(self, peak_db: float, hold_db: float, clipped: bool) -> None:
        self._peak_db = peak_db
        self._hold_db = hold_db
        self._clipped = clipped
        self.update()

    def reset(self) -> None:
        self._peak_db = -math.inf
        self._hold_db = -math.inf
        self._clipped = False
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(200, 16) if self._orientation == Qt.Orientation.Horizontal else QSize(20, 120)

    # ------------------------------------------------------------------ çizim
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = 3.0 if self._orientation == Qt.Orientation.Horizontal else 3.0

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(C.COLOR_BG_SURFACE))
        painter.drawRoundedRect(rect, radius, radius)

        clip_w = 10.0 if self._show_clip else 0.0
        gap = 4.0 if self._show_clip else 0.0
        bar = QRectF(rect)
        if self._orientation == Qt.Orientation.Horizontal:
            bar.setWidth(max(0.0, rect.width() - clip_w - gap))
        else:
            bar.setHeight(max(0.0, rect.height() - clip_w - gap))

        norm = _norm(self._peak_db)
        if norm > 0.001:
            value = QRectF(bar)
            if self._orientation == Qt.Orientation.Horizontal:
                value.setWidth(bar.width() * norm)
                x = bar.left()
                gradient_zone = bar.width()
                color = self._zone_color(norm, gradient_zone, x, value.width())
            else:
                value.setTop(bar.bottom() - bar.height() * norm)
                color = self._zone_color(norm, bar.height(), bar.bottom(), value.height())
            painter.setBrush(color)
            painter.drawRoundedRect(value, 2.0, 2.0)

        # Hold çizgisi
        hold_norm = _norm(self._hold_db)
        if hold_norm > 0.001:
            painter.setPen(QPen(QColor(C.COLOR_TEXT_PRIMARY), 1.6))
            if self._orientation == Qt.Orientation.Horizontal:
                x = bar.left() + bar.width() * min(hold_norm, 1.0)
                painter.drawLine(int(x), int(bar.top()) + 1, int(x), int(bar.bottom()) - 1)
            else:
                y = bar.bottom() - bar.height() * min(hold_norm, 1.0)
                painter.drawLine(int(bar.left()) + 1, int(y), int(bar.right()) - 1, int(y))

        # Clip göstergesi
        if self._show_clip:
            clip_rect = (
                QRectF(rect.right() - clip_w, rect.top(), clip_w, rect.height())
                if self._orientation == Qt.Orientation.Horizontal
                else QRectF(rect.left(), rect.top(), rect.width(), clip_w)
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(C.COLOR_VU_RED if self._clipped else C.COLOR_BG_SURFACE))
            painter.drawRoundedRect(clip_rect, 2.0, 2.0)

        # Ölçek çizgileri
        if self._show_scale and self._orientation == Qt.Orientation.Horizontal:
            painter.setPen(QPen(QColor(C.COLOR_BORDER_LIGHT), 1))
            for db in (-6.0, -12.0, -24.0, -48.0):
                x = bar.left() + bar.width() * _norm(db)
                painter.drawLine(int(x), int(bar.top()) + 2, int(x), int(bar.bottom()) - 2)

        painter.end()

    @staticmethod
    def _zone_color(norm: float, _length: float, _origin: float, _size: float) -> QColor:
        db = _MIN_DB + norm * (0.0 - _MIN_DB)
        if db >= -3.0:
            return QColor(C.COLOR_VU_RED)
        if db >= -12.0:
            return QColor(C.COLOR_VU_YELLOW)
        return QColor(C.COLOR_VU_GREEN)
