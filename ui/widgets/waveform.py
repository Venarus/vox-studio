"""
Waveform bileşeni.

* Peak (min/max) verisinden çizer; tam çözünürlüklü örnek tutmaz.
* Genişliğe göre sütunlara indirger, yeniden çizimde önbellek kullanır.
* Tıklama/sürükleme ile ``seekRequested`` yayar; çalan bölüm vurgulanır.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from audio.waveform import WaveformPeaks
from core import constants as C
from core.format import format_duration

_RULER_HEIGHT = 20


class WaveformWidget(QWidget):
    seekRequested = Signal(float)
    selectionChanged = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mins = np.zeros(0, np.float32)
        self._maxs = np.zeros(0, np.float32)
        self._duration = 0.0
        self._position = 0.0
        self._sel_start: Optional[float] = None
        self._sel_end: Optional[float] = None
        self._drag_start_x: Optional[float] = None
        self._message = "Waveform için bir kayıt veya dosya seçin"
        self._columns: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._columns_width = 0
        self._path = None
        self._path_key = None
        self._active = False
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ------------------------------------------------------------------ veri
    def set_peaks(self, mins: np.ndarray, maxs: np.ndarray, duration: float) -> None:
        self._mins = np.asarray(mins, dtype=np.float32)
        self._maxs = np.asarray(maxs, dtype=np.float32)
        self._duration = float(duration)
        self._columns = None
        self._path = None
        self._active = self._mins.size > 0
        self.update()

    def set_audio(self, data: np.ndarray, samplerate: int, max_buckets: int = 4000) -> None:
        """NumPy ses dizisinden doğrudan tepe özeti üretip waveform'u günceller."""
        arr = np.asarray(data, dtype=np.float32)
        if arr.size == 0 or samplerate <= 0:
            self.clear("Ses verisi boş")
            return
        frames = arr.shape[0]
        channels = 1 if arr.ndim == 1 else arr.shape[1]
        bucket = max(32, int(np.ceil(frames / float(max_buckets))))
        peaks = WaveformPeaks(bucket_frames=bucket, channels=channels)
        peaks.add(arr)
        peaks.flush_partial()
        mins, maxs = peaks.extent()
        self.set_peaks(mins, maxs, float(frames) / float(samplerate))

    def clear(self, message: str = "") -> None:
        self._mins = np.zeros(0, np.float32)
        self._maxs = np.zeros(0, np.float32)
        self._duration = 0.0
        self._position = 0.0
        self._active = False
        self._columns = None
        self._path = None
        if message:
            self._message = message
        self.update()

    def set_position(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        if abs(seconds - self._position) < 1e-4:
            return
        self._position = seconds
        self.update()

    def set_playhead(self, seconds: float) -> None:
        """set_position için anlaşılır takma ad."""
        self.set_position(seconds)

    def selection_in(self) -> Optional[float]:
        if self._sel_start is not None and self._sel_end is not None:
            return min(self._sel_start, self._sel_end)
        return None

    def selection_out(self) -> Optional[float]:
        if self._sel_start is not None and self._sel_end is not None:
            return max(self._sel_start, self._sel_end)
        return None

    def set_selection(self, start: Optional[float], end: Optional[float]) -> None:
        self._sel_start = start
        self._sel_end = end
        self.update()
        self.selectionChanged.emit(self.selection_in(), self.selection_out())

    def clear_selection(self) -> None:
        self.set_selection(None, None)

    @property
    def duration(self) -> float:
        return self._duration

    # ------------------------------------------------------------ indirgeme
    def _build_columns(self, width: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._columns is not None and self._columns_width == width:
            return self._columns
        count = self._mins.size
        if count == 0 or width <= 0:
            self._columns = (np.zeros(0, np.float32), np.zeros(0, np.float32))
            self._columns_width = width
            return self._columns

        if count <= width:
            # Noktaları sütunlara yay
            idx = np.linspace(0, count - 1, width).astype(int)
            cols_min = self._mins[idx]
            cols_max = self._maxs[idx]
        else:
            group = count / float(width)
            starts = (np.arange(width) * group).astype(int)
            cols_min = np.minimum.reduceat(self._mins, starts)
            cols_max = np.maximum.reduceat(self._maxs, starts)
            # reduceat son öğeyi taşırabilir; güvenli kırpma
            cols_min = cols_min[:width]
            cols_max = cols_max[:width]
            if cols_min.size < width:
                cols_min = np.pad(cols_min, (0, width - cols_min.size), mode="edge")
                cols_max = np.pad(cols_max, (0, width - cols_max.size), mode="edge")

        self._columns = (cols_min, cols_max)
        self._columns_width = width
        return self._columns

    # ------------------------------------------------------------------ çizim
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = QRectF(self.rect())
        painter.fillRect(rect, QColor(C.COLOR_BG_PANEL))

        wave_rect = QRectF(rect.left(), rect.top() + 6, rect.width(), rect.height() - _RULER_HEIGHT - 10)
        painter.fillRect(wave_rect, QColor(C.COLOR_BG_DARKEST))

        if not self._active or not math_is_positive(self._duration):
            self._paint_message(painter, rect)
            self._paint_ruler(painter, rect, None)
            painter.end()
            return

        center_y = wave_rect.center().y()
        painter.setPen(QPen(QColor(C.COLOR_BORDER), 1))
        painter.drawLine(int(wave_rect.left()), int(center_y), int(wave_rect.right()), int(center_y))

        width = max(1, int(wave_rect.width()))
        cols_min, cols_max = self._build_columns(width)
        if cols_min.size:
            half = wave_rect.height() / 2.0 - 2.0
            base_y = center_y

            min_arr = np.clip(cols_min, -1.0, 1.0) * half
            max_arr = np.clip(cols_max, -1.0, 1.0) * half
            top = base_y - max_arr
            bottom = np.maximum(base_y - min_arr, top + 1.0)

            path = self._build_path(wave_rect, top, bottom)
            play_x = wave_rect.left() + (self._position / self._duration) * wave_rect.width() if self._duration else wave_rect.left()
            play_x = max(wave_rect.left(), min(wave_rect.right(), play_x))

            # çalınmamış bölüm
            painter.save()
            painter.setClipRect(QRectF(play_x, wave_rect.top(), wave_rect.right() - play_x, wave_rect.height()))
            painter.setPen(QPen(QColor(C.WAVEFORM_COLOR), 1))
            painter.drawPath(path)
            painter.restore()

            # çalınan bölüm
            painter.save()
            painter.setClipRect(QRectF(wave_rect.left(), wave_rect.top(), play_x - wave_rect.left(), wave_rect.height()))
            painter.setPen(QPen(QColor(C.WAVEFORM_COLOR_PLAYED), 1))
            painter.drawPath(path)
            painter.restore()

            # playhead
            painter.setPen(QPen(QColor(C.COLOR_TEXT_PRIMARY), 1.4))
            painter.drawLine(int(play_x), int(wave_rect.top()), int(play_x), int(wave_rect.bottom()))

            # seçim alanı
            sel_in = self.selection_in()
            sel_out = self.selection_out()
            if sel_in is not None and sel_out is not None and self._duration > 0:
                sx1 = wave_rect.left() + (sel_in / self._duration) * wave_rect.width()
                sx2 = wave_rect.left() + (sel_out / self._duration) * wave_rect.width()
                sx1 = max(wave_rect.left(), min(wave_rect.right(), sx1))
                sx2 = max(wave_rect.left(), min(wave_rect.right(), sx2))
                if sx2 > sx1:
                    sel_rect = QRectF(sx1, wave_rect.top(), sx2 - sx1, wave_rect.height())
                    painter.fillRect(sel_rect, QColor(230, 57, 70, 50))
                    painter.setPen(QPen(QColor(230, 57, 70, 180), 1.2))
                    painter.drawLine(int(sx1), int(wave_rect.top()), int(sx1), int(wave_rect.bottom()))
                    painter.drawLine(int(sx2), int(wave_rect.top()), int(sx2), int(wave_rect.bottom()))

        self._paint_ruler(painter, rect, self._duration)
        painter.end()

    def _build_path(self, wave_rect: QRectF, top: np.ndarray, bottom: np.ndarray):
        key = (id(self._mins), wave_rect.width(), wave_rect.height(), float(self._mins.size))
        if self._path is not None and self._path_key == key:
            return self._path
        from PySide6.QtGui import QPainterPath

        path = QPainterPath()
        left = wave_rect.left()
        for x in range(top.size):
            px = left + x + 0.5
            path.moveTo(px, float(top[x]))
            path.lineTo(px, float(bottom[x]))
        self._path = path
        self._path_key = key
        return path

    def _paint_message(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QColor(C.COLOR_TEXT_MUTED))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._message)

    def _paint_ruler(self, painter: QPainter, rect: QRectF, duration: Optional[float]) -> None:
        ruler_top = rect.bottom() - _RULER_HEIGHT
        painter.setPen(QPen(QColor(C.COLOR_BORDER), 1))
        painter.drawLine(int(rect.left()), int(ruler_top), int(rect.right()), int(ruler_top))

        if not duration or duration <= 0:
            return
        painter.setPen(QColor(C.COLOR_TEXT_MUTED))
        fm = QFontMetrics(self.font())
        target_ticks = max(2, int(rect.width() / 110))
        raw_step = duration / target_ticks
        step = _nice_step(raw_step)
        t = 0.0
        while t <= duration + 1e-6:
            x = rect.left() + (t / duration) * rect.width()
            painter.setPen(QPen(QColor(C.COLOR_BORDER_LIGHT), 1))
            painter.drawLine(int(x), int(ruler_top + 2), int(x), int(ruler_top + 6))
            painter.setPen(QColor(C.COLOR_TEXT_MUTED))
            label = format_duration(t, millis=False)
            painter.drawText(int(x) + 3, int(rect.bottom() - 3), label)
            t += step

    # ------------------------------------------------------------- etkileşim
    def _x_to_time(self, x: float) -> float:
        if self._duration <= 0:
            return 0.0
        rect = self.rect()
        ratio = (x - rect.left()) / max(1.0, float(rect.width()))
        ratio = max(0.0, min(1.0, ratio))
        return ratio * self._duration

    def _seek_from_x(self, x: float) -> None:
        if self._duration <= 0:
            return
        t = self._x_to_time(x)
        self.seekRequested.emit(t)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_x = event.position().x()
            self._sel_start = self._x_to_time(self._drag_start_x)
            self._sel_end = self._sel_start
            self._seek_from_x(self._drag_start_x)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            cur_x = event.position().x()
            if self._drag_start_x is not None and abs(cur_x - self._drag_start_x) > 4:
                self._sel_end = self._x_to_time(cur_x)
                self.update()
                self.selectionChanged.emit(self.selection_in(), self.selection_out())
            else:
                self._seek_from_x(cur_x)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start_x is not None:
            cur_x = event.position().x()
            if abs(cur_x - self._drag_start_x) <= 4:
                self._sel_start = None
                self._sel_end = None
                self.update()
                self.selectionChanged.emit(None, None)
            self._drag_start_x = None
        super().mouseReleaseEvent(event)


def math_is_positive(value: float) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _nice_step(raw: float) -> float:
    """0.5, 1, 2, 5, 10, 15, 30, 60... biçiminde okunabilir adım."""
    if raw <= 0:
        return 1.0
    for step in (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
        if raw <= step:
            return float(step)
    return float(int(raw))


def peaks_from_file(
    path: str, max_buckets: int = 6000
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Dosyadan görüntüleme için tepe özetleri üretir.

    Dönen: (mins, maxs, frames, samplerate). Bellek kullanımı sabittir.
    libsndfile tarafından doğrudan okunamayan biçimlerde (M4A, AAC, Opus, WMA, MP4 vb.)
    PyAV üzerinden ses akışını çözer.
    """
    import soundfile as sf

    try:
        info = sf.info(str(path))
        if info.frames <= 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32), 0, int(info.samplerate)
        bucket = max(64, int(np.ceil(info.frames / float(max_buckets))))
        peaks = WaveformPeaks(bucket_frames=bucket, channels=int(info.channels))
        with sf.SoundFile(str(path), mode="r") as f:
            while True:
                block = f.read(65536, dtype="float32", always_2d=True)
                if block.shape[0] == 0:
                    break
                peaks.add(block)
        peaks.flush_partial()
        mins, maxs = peaks.extent()
        return mins, maxs, int(info.frames), int(info.samplerate)
    except Exception:
        from ai.media_extractor import MediaExtractor

        arr, sr = MediaExtractor.read_audio_any(path, always_2d=True)
        frames = int(arr.shape[0])
        if frames <= 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32), 0, int(sr)
        bucket = max(64, int(np.ceil(frames / float(max_buckets))))
        peaks = WaveformPeaks(bucket_frames=bucket, channels=int(arr.shape[1]))
        peaks.add(arr)
        peaks.flush_partial()
        mins, maxs = peaks.extent()
        return mins, maxs, frames, int(sr)


WaveformView = WaveformWidget


def peaks_from_samples(
    samples: np.ndarray, max_buckets: int = 1600, points: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy ses dizisinden (mins, maxs) tepe özetlerini üretir."""
    if points is not None:
        max_buckets = points
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    frames = int(arr.shape[0])
    bucket = max(16, int(np.ceil(frames / float(max(1, max_buckets)))))
    peaks = WaveformPeaks(bucket_frames=bucket, channels=int(arr.shape[1]))
    peaks.add(arr)
    peaks.flush_partial()
    return peaks.extent()
