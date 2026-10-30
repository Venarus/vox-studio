"""
VOX Studio — Akıcı ve Göze Batmayan Toast Bildirim Bileşeni.

İşlem tamamlandığında (proje kaydedildi, kısayol güncellendi, altyazı aktarıldı vb.)
kullanıcıyı rahatsız etmeden ekranın sağ alt köşesinde yumuşak animasyonla belirip kaybolur.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from ui import icons


class ToastNotification(QFrame):
    """Yumuşak animasyonlu toast bildirim penceresi."""

    def __init__(
        self,
        text: str,
        variant: str = "success",  # "success" | "info" | "warning" | "error"
        duration_ms: int = 2600,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.SubWindow | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        border_color = {
            "success": "#10B981",
            "info": "#38BDF8",
            "warning": "#F59E0B",
            "error": "#E53945",
        }.get(variant, "#E53945")

        icon_name = {
            "success": "check",
            "info": "info",
            "warning": "alert",
            "error": "alert-circle",
        }.get(variant, "info")

        self.setStyleSheet(
            f"ToastNotification {{ background-color: #151619; border: 1px solid {border_color}; "
            f"border-radius: 8px; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 16, 10)
        layout.setSpacing(10)

        ic = QLabel()
        ic.setPixmap(icons.pixmap(icon_name, 16, border_color))
        layout.addWidget(ic)

        msg = QLabel(text)
        msg.setStyleSheet("color: #F5F5F6; font-size: 12px; font-weight: 600;")
        layout.addWidget(msg)

        self.adjustSize()

        # Konumlandırma (Sağ alt köşe)
        if parent:
            pw = parent.width()
            ph = parent.height()
            self.move(pw - self.width() - 24, ph - self.height() - 32)

        # Opaklık Animasyonu
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(0.0)

        self._anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._anim.setDuration(240)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Otomatik kapanma
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out)

        self.show()
        self._anim.start()
        self._hide_timer.start(duration_ms)

    def _fade_out(self) -> None:
        self._anim.stop()
        self._anim.setDuration(280)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._anim.finished.connect(self.deleteLater)
        self._anim.start()

    @classmethod
    def show_toast(
        cls,
        parent: Optional[QWidget],
        text: str,
        variant: str = "success",
        duration_ms: int = 2600,
    ) -> None:
        if parent is None:
            parent = QApplication.activeWindow()
        cls(text, variant=variant, duration_ms=duration_ms, parent=parent)
