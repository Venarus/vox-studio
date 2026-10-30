"""
VOX Studio — Açılış Splash Ekranı (Animasyonlu Karşılama)

Özellikler:
- Siyah arka plan (#090A0C) ve kontrollü koyu kırmızı vurgu (#E53945).
- 0.8 saniyelik pürüzsüz animasyonlu yükleme göstergesi.
- Windows erişilebilirlik "Azaltılmış Hareket" (Reduced Motion) ayarına saygı gösterir.
- Ana pencere hazır olduğunda pencereyi 'showMaximized()' olarak açar.
"""

from __future__ import annotations

import sys
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSplashScreen, QWidget, QApplication

from core import constants as C
from core.logger import get_logger

logger = get_logger("ui.splash")


class VoxSplashScreen(QSplashScreen):
    """
    VOX Studio açılışında gösterilen hafif, şık ve modern splash ekranı.
    """

    def __init__(self, duration_ms: int = 800, reduced_motion: bool = False):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        self.duration_ms = 100 if reduced_motion else max(400, duration_ms)
        self.progress: float = 0.0  # 0.0 .. 1.0
        self.status_text: str = "Stüdyo bileşenleri yükleniyor..."

        # Splash pencere boyutu (DPI duyarlı)
        self.resize(520, 320)

        # Ekranın ortasına konumlandır
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(
                geo.x() + (geo.width() - self.width()) // 2,
                geo.y() + (geo.height() - self.height()) // 2,
            )

        # Animasyon zamanlayıcısı
        self._step_timer = QTimer(self)
        self._step_timer.setInterval(25)  # 40 fps
        self._step_timer.timeout.connect(self._on_tick)
        self._elapsed = 0

    def start_animation(self) -> None:
        """Animasyonu başlatır."""
        self.show()
        QApplication.processEvents()
        self._step_timer.start()

    def set_status(self, text: str) -> None:
        self.status_text = text
        self.update()
        QApplication.processEvents()

    def _on_tick(self) -> None:
        self._elapsed += 25
        self.progress = min(1.0, self._elapsed / float(self.duration_ms))
        if self.progress < 0.35:
            self.status_text = "Ses ve video motorları başlatılıyor..."
        elif self.progress < 0.70:
            self.status_text = "Çok dilli stüdyo modülleri hazırlanıyor..."
        elif self.progress < 0.95:
            self.status_text = "Çalışma alanı yapılandırılıyor..."
        else:
            self.status_text = "VOX Studio Hazır."

        self.update()

        if self.progress >= 1.0:
            self._step_timer.stop()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w, h = float(self.width()), float(self.height())

        # 1. Arka plan (#090A0C) ve çok ince koyu kenarlık (#28292E)
        bg_rect = QRectF(0, 0, w, h)
        painter.fillRect(bg_rect, QColor("#090A0C"))

        pen_border = QPen(QColor("#28292E"), 1.0)
        painter.setPen(pen_border)
        painter.drawRect(bg_rect.adjusted(0.5, 0.5, -0.5, -0.5))

        # 2. Üst/Arka plan koyu kırmızı şık parlama gradyanı (Hafif ve profesyonel)
        glow_grad = QLinearGradient(0, 0, w, h * 0.7)
        glow_grad.setColorAt(0.0, QColor(229, 57, 69, 35))   # #E53945 %14 alfa
        glow_grad.setColorAt(1.0, QColor(9, 10, 12, 0))
        painter.fillRect(QRectF(0, 0, w, h * 0.7), glow_grad)

        # 3. İkon / Rozet Çizimi (Ortada modern stilize 'V')
        cx, cy = w / 2.0, 95.0
        badge_size = 54.0
        badge_rect = QRectF(cx - badge_size / 2.0, cy - badge_size / 2.0, badge_size, badge_size)

        badge_path = QPainterPath()
        badge_path.addRoundedRect(badge_rect, 14.0, 14.0)
        painter.fillPath(badge_path, QColor("#151619"))
        painter.strokePath(badge_path, QPen(QColor("#E53945"), 1.5))

        # V Harfi çizimi
        v_path = QPainterPath()
        v_path.moveTo(cx - 14.0, cy - 12.0)
        v_path.lineTo(cx, cy + 14.0)
        v_path.lineTo(cx + 14.0, cy - 12.0)
        v_pen = QPen(QColor("#E53945"), 4.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(v_path, v_pen)

        # 4. Başlık: VOX Studio (#F5F5F6)
        title_font = QFont("Segoe UI", 22, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#F5F5F6"))
        painter.drawText(QRectF(0, 140, w, 36), Qt.AlignmentFlag.AlignCenter, "VOX Studio")

        # 5. Alt Başlık: Profesyonel Video Düzenleme, Dublaj ve AI Ses Laboratuvarı (#A1A1AA)
        sub_font = QFont("Segoe UI", 10, QFont.Weight.Normal)
        painter.setFont(sub_font)
        painter.setPen(QColor("#A1A1AA"))
        painter.drawText(
            QRectF(20, 178, w - 40, 22),
            Qt.AlignmentFlag.AlignCenter,
            "Video Düzenleme · AI Ses Ayrıştırma · Dublaj Stüdyosu",
        )

        # 6. Yükleme Barı (Koyu yüzey üzerinde ince #E53945 ilerleme çizgisi)
        bar_w = w - 120.0
        bar_h = 4.0
        bar_x = (w - bar_w) / 2.0
        bar_y = 230.0

        bar_bg_rect = QRectF(bar_x, bar_y, bar_w, bar_h)
        bar_bg_path = QPainterPath()
        bar_bg_path.addRoundedRect(bar_bg_rect, 2.0, 2.0)
        painter.fillPath(bar_bg_path, QColor("#1C1D21"))

        # İlerleme dolgusu
        fill_w = max(4.0, bar_w * self.progress)
        fill_rect = QRectF(bar_x, bar_y, fill_w, bar_h)
        fill_path = QPainterPath()
        fill_path.addRoundedRect(fill_rect, 2.0, 2.0)
        painter.fillPath(fill_path, QColor("#E53945"))

        # 7. Durum Metni (#71717A) ve Sürüm
        status_font = QFont("Segoe UI", 9)
        painter.setFont(status_font)
        painter.setPen(QColor("#71717A"))
        painter.drawText(QRectF(30, 246, w - 60, 20), Qt.AlignmentFlag.AlignCenter, self.status_text)

        ver_font = QFont("Segoe UI", 8)
        painter.setFont(ver_font)
        painter.setPen(QColor("#4B5563"))
        painter.drawText(QRectF(20, h - 28, w - 40, 18), Qt.AlignmentFlag.AlignRight, f"v{C.APP_VERSION}")

        painter.end()
