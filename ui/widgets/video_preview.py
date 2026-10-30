"""
VOX Studio — Video, Görsel ve Ses Önizleme / Oynatıcı Paneli (QtMultimedia)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, QUrl, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core import constants as C
from core.logger import get_logger
from ui import icons

logger = get_logger("ui.widgets.video_preview")


class VideoPreviewWidget(QWidget):
    """Gerçek zamanlı video, görsel ve ses önizleme + kare denetim penceresi."""

    positionChanged = Signal(float)  # saniye cinsinden konum
    durationChanged = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.fps = 30.0
        self.duration_sec = 0.0
        self.current_time_sec = 0.0
        self._is_seeking = False
        self.current_media_path: Optional[str] = None
        self.playback_controller: Optional[Any] = None
        self.is_timeline_mode: bool = False
        self.timeline_duration_sec: float = 0.0

        self._build_player()
        self._build_ui()

    def _build_player(self):
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.8)
        self.player.setAudioOutput(self.audio_output)

        self.player.positionChanged.connect(self._on_player_pos_changed)
        self.player.durationChanged.connect(self._on_player_dur_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # Video / Görsel Ekranı
        self.screen_frame = QFrame()
        self.screen_frame.setObjectName("Card")
        self.screen_frame.setStyleSheet("background-color: #0B0C0E; border-radius: 6px;")
        sf_layout = QVBoxLayout(self.screen_frame)
        sf_layout.setContentsMargins(0, 0, 0, 0)

        self.display_stack = QStackedWidget()

        self.video_widget = QVideoWidget()
        self.player.setVideoOutput(self.video_widget)
        self.display_stack.addWidget(self.video_widget)

        # Görsel / Ses Önizleme Etiketi (Görsel dosyaları veya ses önizlemesi için)
        self.image_preview_label = QLabel("Önizleme için medya seçin veya timeline'a ekleyin")
        self.image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_label.setStyleSheet("color: #6C7687; font-size: 12px; padding: 16px;")
        self.display_stack.addWidget(self.image_preview_label)

        sf_layout.addWidget(self.display_stack)

        # Canlı Altyazı Bindirmesi (Canlı Timeline Subtitle Overlay)
        self.subtitle_overlay_lbl = QLabel("")
        self.subtitle_overlay_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_overlay_lbl.setStyleSheet(
            "background-color: rgba(0, 0, 0, 180); color: #FFFFFF; "
            "font-size: 14px; font-weight: 600; padding: 6px 14px; border-radius: 4px; margin-bottom: 8px;"
        )
        self.subtitle_overlay_lbl.setVisible(False)
        sf_layout.addWidget(self.subtitle_overlay_lbl, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)

        root.addWidget(self.screen_frame, 1)

        # Oynatma ve Zaman Çubuğu
        ctrl_bar = QFrame()
        ctrl_bar.setFixedHeight(44)
        ctrl_bar.setStyleSheet("background-color: #17191F; border: 1px solid #2E3442; border-radius: 6px;")
        c_layout = QHBoxLayout(ctrl_bar)
        c_layout.setContentsMargins(10, 4, 10, 4)
        c_layout.setSpacing(6)

        # Kare Geri
        self.prev_frame_btn = QPushButton()
        self.prev_frame_btn.setProperty("compact", "true")
        self.prev_frame_btn.setIcon(icons.icon("skip-back", size=15))
        self.prev_frame_btn.setIconSize(QSize(15, 15))
        self.prev_frame_btn.setFixedSize(32, 30)
        self.prev_frame_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_frame_btn.setToolTip("1 Kare Geri (Sol Ok)")
        self.prev_frame_btn.clicked.connect(self.step_backward)
        c_layout.addWidget(self.prev_frame_btn)

        # Oynat / Durdur
        self.play_btn = QPushButton(" Oynat")
        self.play_btn.setProperty("variant", "primary")
        self.play_btn.setProperty("compact", "true")
        self.play_btn.setIcon(icons.icon("play", size=14, color="#FFFFFF"))
        self.play_btn.setIconSize(QSize(14, 14))
        self.play_btn.setMinimumSize(88, 30)
        self.play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_btn.setToolTip("Önizlemeyi Oynat / Duraklat (Boşluk)")
        self.play_btn.clicked.connect(self.toggle_play)
        c_layout.addWidget(self.play_btn)

        # Kare İleri
        self.next_frame_btn = QPushButton()
        self.next_frame_btn.setProperty("compact", "true")
        self.next_frame_btn.setIcon(icons.icon("skip-forward", size=15))
        self.next_frame_btn.setIconSize(QSize(15, 15))
        self.next_frame_btn.setFixedSize(32, 30)
        self.next_frame_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_frame_btn.setToolTip("1 Kare İleri (Sağ Ok)")
        self.next_frame_btn.clicked.connect(self.step_forward)
        c_layout.addWidget(self.next_frame_btn)

        # Zaman Kodu (HH:MM:SS:FF)
        self.timecode_label = QLabel("00:00:00:00 / 00:00:00:00")
        self.timecode_label.setStyleSheet(
            "color: #F2F4F8; font-family: Consolas, monospace; font-size: 12px; font-weight: bold;"
        )
        c_layout.addWidget(self.timecode_label)

        c_layout.addStretch(1)

        # En-Boy Oranı
        ratio_lbl = QLabel("Oran:")
        ratio_lbl.setStyleSheet("color: #A2ABB8; font-size: 11px;")
        c_layout.addWidget(ratio_lbl)
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(["16:9 Geniş", "9:16 Dikey", "1:1 Kare", "4:3 Klasik"])
        self.ratio_combo.setMinimumWidth(108)
        self.ratio_combo.currentIndexChanged.connect(self._change_ratio)
        c_layout.addWidget(self.ratio_combo)

        # Ses Seviyesi Slider
        vol_icon = QLabel()
        vol_icon.setPixmap(icons.pixmap("volume", 15, C.COLOR_TEXT_SECONDARY))
        c_layout.addWidget(vol_icon)
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.setFixedWidth(74)
        self.vol_slider.setToolTip("Önizleme ses seviyesi")
        self.vol_slider.valueChanged.connect(lambda v: self.audio_output.setVolume(v / 100.0))
        c_layout.addWidget(self.vol_slider)

        root.addWidget(ctrl_bar)

    def load_media(self, media_path: str | Path):
        p = Path(media_path)
        if not p.exists():
            return
        self.current_media_path = str(p)
        suffix = p.suffix.lower()

        if suffix in C.IMAGE_EXTENSIONS:
            self.player.stop()
            pm = QPixmap(str(p))
            if not pm.isNull():
                scaled = pm.scaled(
                    max(320, self.screen_frame.width() - 16),
                    max(180, self.screen_frame.height() - 16),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.image_preview_label.setPixmap(scaled)
            else:
                self.image_preview_label.setText(f"Görsel: {p.name}")
            self.display_stack.setCurrentWidget(self.image_preview_label)
            self.duration_sec = 5.0
            self._update_timecode(0.0)
            logger.info(f"Görsel önizlemesine yüklendi: {p.name}")
            return

        if suffix in C.SUBTITLE_EXTENSIONS:
            self.player.stop()
            try:
                from ai.media_extractor import parse_subtitles
                cues = parse_subtitles(p)
                preview_txt = "\n".join(
                    f"[{c.start_sec:.1f}s - {c.end_sec:.1f}s] {c.text}" for c in cues[:5]
                )
                self.image_preview_label.setPixmap(QPixmap())
                self.image_preview_label.setText(f"Altyazı: {p.name} ({len(cues)} replik)\n\n{preview_txt}")
            except Exception as exc:
                self.image_preview_label.setText(f"Altyazı okunamadı: {exc}")
            self.display_stack.setCurrentWidget(self.image_preview_label)
            return

        # Video veya Ses dosyası
        self.display_stack.setCurrentWidget(self.video_widget)
        url = QUrl.fromLocalFile(str(p))
        if self.player.source() != url:
            self.player.setSource(url)
        logger.info(f"Medya önizlemesine yüklendi: {p.name}")

    def set_playback_controller(self, controller: Optional[Any]) -> None:
        """Master PlaybackController nesnesini bağlar ve önizleme zaman çizelgesi modunu etkinleştirir."""
        self.playback_controller = controller
        if controller is not None:
            self.is_timeline_mode = True
            self.fps = controller.fps
            controller.timelinePositionChanged.connect(self._on_timeline_clock_pos_changed)
            controller.playbackStateChanged.connect(self._on_playback_state_bool_changed)
            controller.frameReady.connect(self.set_composition_frame)

    def _on_timeline_clock_pos_changed(self, seq_t: float) -> None:
        dur = self.playback_controller.duration if self.playback_controller else self.duration_sec
        self.set_timeline_time(seq_t, dur)

    def _on_playback_state_bool_changed(self, is_playing: bool) -> None:
        if is_playing:
            self.play_btn.setText(" Durdur")
            self.play_btn.setIcon(icons.icon("pause", size=14, color="#FFFFFF"))
        else:
            self.play_btn.setText(" Oynat")
            self.play_btn.setIcon(icons.icon("play", size=14, color="#FFFFFF"))

    def set_timeline_time(self, curr_sec: float, total_sec: Optional[float] = None) -> None:
        """Zaman çizelgesi oynatıcı süresini ve toplam süresini ayarlar."""
        self.is_timeline_mode = True
        self.current_time_sec = max(0.0, float(curr_sec))
        if total_sec is not None:
            self.timeline_duration_sec = max(0.0, float(total_sec))
        self._update_timecode(self.current_time_sec)

    def show_blank_frame(self) -> None:
        """Boşluklarda (gap) veya video klibi olmayan zamanlarda temiz siyah kare gösterir."""
        w = max(320, self.screen_frame.width() - 16)
        h = max(180, self.screen_frame.height() - 16)
        blank = QPixmap(w, h)
        blank.fill(Qt.GlobalColor.black)
        self.image_preview_label.setPixmap(blank)
        self.image_preview_label.setText("")
        self.display_stack.setCurrentWidget(self.image_preview_label)
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        self.current_media_path = None

    def set_composition_frame(self, frame: Any) -> None:
        """Kompozisyon karesini (QImage veya QPixmap) önizleme ekranında gösterir."""
        if frame is None:
            self.show_blank_frame()
            return
        if isinstance(frame, QImage):
            pix = QPixmap.fromImage(frame)
        elif isinstance(frame, QPixmap):
            pix = frame
        else:
            return
        if not pix.isNull():
            scaled = pix.scaled(
                max(320, self.screen_frame.width() - 16),
                max(180, self.screen_frame.height() - 16),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_preview_label.setPixmap(scaled)
            self.image_preview_label.setText("")
            self.display_stack.setCurrentWidget(self.image_preview_label)

    def toggle_play(self):
        if self.playback_controller is not None:
            self.playback_controller.toggle_play()
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState):
        if self.is_timeline_mode and self.playback_controller is not None:
            return
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_btn.setText(" Durdur")
            self.play_btn.setIcon(icons.icon("pause", size=14, color="#FFFFFF"))
        else:
            self.play_btn.setText(" Oynat")
            self.play_btn.setIcon(icons.icon("play", size=14, color="#FFFFFF"))

    def seek_seconds(self, sec: float):
        self._is_seeking = True
        self.current_time_sec = max(0.0, sec)
        ms = int(self.current_time_sec * 1000)
        self.player.setPosition(ms)
        self._update_timecode(self.current_time_sec)
        self._is_seeking = False

    def seek_source_seconds(self, source_sec: float):
        """Alt medya oynatıcısını zaman kodu gösterimini bozmadan kaynak zamana konumlandırır."""
        self._is_seeking = True
        ms = int(max(0.0, float(source_sec)) * 1000)
        self.player.setPosition(ms)
        self._is_seeking = False

    def step_forward(self):
        if self.playback_controller is not None:
            self.playback_controller.step_forward()
            return
        frame_dur = 1.0 / max(1.0, self.fps)
        self.seek_seconds(self.current_time_sec + frame_dur)
        self.positionChanged.emit(self.current_time_sec)

    def step_backward(self):
        if self.playback_controller is not None:
            self.playback_controller.step_backward()
            return
        frame_dur = 1.0 / max(1.0, self.fps)
        self.seek_seconds(max(0.0, self.current_time_sec - frame_dur))
        self.positionChanged.emit(self.current_time_sec)

    def _on_player_pos_changed(self, pos_ms: int):
        if self._is_seeking:
            return
        # Zaman çizelgesi modunda playhead master clock tarafından yönetilir;
        # klip içi süre timeline playhead'ini sıfırlamamalıdır!
        if self.is_timeline_mode:
            return
        sec = pos_ms / 1000.0
        self.current_time_sec = sec
        self._update_timecode(sec)
        self.positionChanged.emit(sec)

    def _on_player_dur_changed(self, dur_ms: int):
        self.duration_sec = dur_ms / 1000.0
        self.durationChanged.emit(self.duration_sec)
        self._update_timecode(self.current_time_sec)

    def set_subtitle_overlay(self, text: str) -> None:
        t = (text or "").strip()
        if t:
            self.subtitle_overlay_lbl.setText(t)
            self.subtitle_overlay_lbl.setVisible(True)
        else:
            self.subtitle_overlay_lbl.setVisible(False)

    def _update_timecode(self, curr_sec: float):
        def _fmt(s: float) -> str:
            s = max(0.0, float(s))
            total_sec = int(s)
            hours = total_sec // 3600
            mins = (total_sec % 3600) // 60
            secs = total_sec % 60
            frames = int((s % 1.0) * self.fps)
            return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"

        total_dur = (
            self.timeline_duration_sec
            if self.timeline_duration_sec > 0
            else self.duration_sec
        )
        txt = f"{_fmt(curr_sec)} / {_fmt(total_dur)}"
        self.timecode_label.setText(txt)

    def _change_ratio(self, idx: int):
        self.update()

