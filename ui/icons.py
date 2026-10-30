"""
VOX Studio — vektör simgeler.

Simgeler gömülü SVG olarak üretilir; renk temadan gelir ve yüksek DPI'da
keskin kalır. Harici dosya bağımlılığı yoktur.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from core import constants as C

_STROKE = (
    'fill="none" stroke="{color}" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round"'
)

_ICONS: Dict[str, str] = {
    # Navigasyon
    "folder": f'<path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h4l2 2.2h7A1.5 1.5 0 0 1 19 9.7v7.8A1.5 1.5 0 0 1 17.5 19h-13A1.5 1.5 0 0 1 3 17.5z" {_STROKE}/>',
    "mic": (
        f'<rect x="9" y="2.8" width="6" height="11" rx="3" {_STROKE}/>'
        f'<path d="M5.5 11.2a6.5 6.5 0 0 0 13 0" {_STROKE}/>'
        f'<path d="M12 17.7V21M8.6 21h6.8" {_STROKE}/>'
    ),
    "sliders": (
        f'<path d="M4 7h6M14 7h6M4 12h10M18 12h2M4 17h3M11 17h9" {_STROKE}/>'
        f'<circle cx="12" cy="7" r="2" {_STROKE}/>'
        f'<circle cx="16" cy="12" r="2" {_STROKE}/>'
        f'<circle cx="9" cy="17" r="2" {_STROKE}/>'
    ),
    "film": (
        f'<rect x="3" y="4.5" width="18" height="15" rx="2" {_STROKE}/>'
        f'<path d="M7.5 4.5v15M16.5 4.5v15M3 12h18M3 8.2h4.5M3 15.8h4.5M16.5 8.2H21M16.5 15.8H21" {_STROKE}/>'
    ),
    "text": (
        f'<path d="M5 5.5h14M5 10h14M5 14.5h9M5 19h6" {_STROKE}/>'
    ),
    "layers": (
        f'<path d="M12 3.5 21 8l-9 4.5L3 8z" {_STROKE}/>'
        f'<path d="M3 12.2l9 4.5 9-4.5M3 16.4l9 4.5 9-4.5" {_STROKE}/>'
    ),
    "library": (
        f'<rect x="3.5" y="4" width="17" height="16" rx="2" {_STROKE}/>'
        f'<path d="M3.5 9h17M9.5 9v11" {_STROKE}/>'
    ),
    "gear": (
        f'<circle cx="12" cy="12" r="3.2" {_STROKE}/>'
        f'<path d="M12 2.8v2.6M12 18.6v2.6M4.5 4.5l1.9 1.9M17.6 17.6l1.9 1.9M2.8 12h2.6M18.6 12h2.6M4.5 19.5l1.9-1.9M17.6 6.4l1.9-1.9" {_STROKE}/>'
    ),
    # Taşıma
    "play": '<path d="M8 4.8 19 12 8 19.2z" fill="{color}"/>',
    "pause": '<rect x="7.5" y="4.5" width="3.4" height="15" rx="1" fill="{color}"/><rect x="13.1" y="4.5" width="3.4" height="15" rx="1" fill="{color}"/>',
    "stop": '<rect x="6.5" y="6.5" width="11" height="11" rx="1.6" fill="{color}"/>',
    "record": '<circle cx="12" cy="12" r="6.2" fill="{color}"/>',
    "skip-back": f'<path d="M11 6v12M6 6v12" {_STROKE}/><path d="M19 6.5 12 12l7 5.5z" fill="{{color}}"/>',
    "skip-forward": f'<path d="M13 6v12M18 6v12" {_STROKE}/><path d="M5 6.5 12 12l-7 5.5z" fill="{{color}}"/>',
    # Eylemler
    "plus": f'<path d="M12 5v14M5 12h14" {_STROKE}/>',
    "trash": f'<path d="M4 7h16M9.5 7V4.8h5V7M6.5 7l.9 12.2h9.2L17.5 7M10 11v5M14 11v5" {_STROKE}/>',
    "refresh": f'<path d="M20 12a8 8 0 1 1-2.6-5.9M20 4v4.5h-4.5" {_STROKE}/>',
    "save": (
        f'<path d="M5 4h11l3.5 3.5V20H5z" {_STROKE}/>'
        f'<path d="M8.5 4v5h6V4M8 20v-5.5h8V20" {_STROKE}/>'
    ),
    "open-external": f'<path d="M14 4h6v6M20 4l-8 8M18 14v5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 19V8a1.5 1.5 0 0 1 1.5-1.5H10" {_STROKE}/>',
    "waveform": (
        f'<path d="M3 12h2.2M8 7.5v9M12 4.5v15M16 8.5v7M20.8 12H19" {_STROKE}/>'
    ),
    "analyze": (
        f'<path d="M4 19V9M8.5 19V5M13 19v-7M17.5 19v-4M21 19V11" {_STROKE}/>'
    ),
    "check": f'<path d="M5 12.5 9.5 17 19 7" {_STROKE}/>',
    "warning": (
        f'<path d="M12 3.8 21 19.5H3z" {_STROKE}/>'
        f'<path d="M12 9.5v4.5M12 17v.4" {_STROKE}/>'
    ),
    "meter": f'<path d="M4 14a8 8 0 0 1 16 0" {_STROKE}/><path d="M12 14l4.2-4.2" {_STROKE}/>',
    "headphones": (
        f'<path d="M4.5 15v-3a7.5 7.5 0 0 1 15 0v3" {_STROKE}/>'
        f'<rect x="3" y="14" width="3.6" height="6" rx="1.6" {_STROKE}/>'
        f'<rect x="17.4" y="14" width="3.6" height="6" rx="1.6" {_STROKE}/>'
    ),
    "clock": f'<circle cx="12" cy="12" r="8.2" {_STROKE}/><path d="M12 7.5V12l3 2" {_STROKE}/>',
    "info": f'<circle cx="12" cy="12" r="8.2" {_STROKE}/><path d="M12 11v5.2M12 8v.4" {_STROKE}/>',
    # Medya kategorileri ve düzenleme simgeleri (Lucide ISC uyumlu vektör çizimler)
    "video": (
        f'<rect x="2.5" y="5.5" width="13.5" height="13" rx="2" {_STROKE}/>'
        f'<path d="m16 10 5.5-3v10L16 14z" {_STROKE}/>'
    ),
    "music": (
        f'<path d="M9 18V5l11-2v13" {_STROKE}/>'
        f'<circle cx="6.5" cy="18" r="2.5" {_STROKE}/>'
        f'<circle cx="17.5" cy="16" r="2.5" {_STROKE}/>'
    ),
    "image": (
        f'<rect x="3.5" y="3.5" width="17" height="17" rx="2" {_STROKE}/>'
        f'<circle cx="8.8" cy="8.8" r="1.6" {_STROKE}/>'
        f'<path d="m20.5 15-4.5-4.5L5.5 20.5" {_STROKE}/>'
    ),
    "subtitle": (
        f'<rect x="2.5" y="5" width="19" height="14" rx="2" {_STROKE}/>'
        f'<path d="M6.5 12h4M13.5 12h4M6.5 15.5h6M15.5 15.5h2" {_STROKE}/>'
    ),
    "sparkles": (
        f'<path d="m12 3 1.9 4.6L18.5 9.5l-4.6 1.9L12 16l-1.9-4.6L5.5 9.5l4.6-1.9z" {_STROKE}/>'
        f'<path d="M19 15v4M17 17h4M5 16v3M3.5 17.5h3" {_STROKE}/>'
    ),
    "scissors": (
        f'<circle cx="6" cy="6" r="2.6" {_STROKE}/>'
        f'<circle cx="6" cy="18" r="2.6" {_STROKE}/>'
        f'<path d="M20 4 8.2 15.8M14.5 14.5 20 20M8.2 8.2 12 12" {_STROKE}/>'
    ),
    "copy": (
        f'<rect x="8.5" y="8.5" width="11.5" height="11.5" rx="2" {_STROKE}/>'
        f'<path d="M5 15.5H4a1.5 1.5 0 0 1-1.5-1.5V4A1.5 1.5 0 0 1 4 2.5h10A1.5 1.5 0 0 1 15.5 4v1" {_STROKE}/>'
    ),
    "undo": (
        f'<path d="M3.5 7.5v5h5" {_STROKE}/>'
        f'<path d="M3.5 12.5A8 8 0 1 1 6 18" {_STROKE}/>'
    ),
    "redo": (
        f'<path d="M20.5 7.5v5h-5" {_STROKE}/>'
        f'<path d="M20.5 12.5A8 8 0 1 0 18 18" {_STROKE}/>'
    ),
    "upload": (
        f'<path d="M12 15V3.5M7.5 8 12 3.5 16.5 8" {_STROKE}/>'
        f'<path d="M4 15.5v3A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5v-3" {_STROKE}/>'
    ),
    "download": (
        f'<path d="M12 3.5V15M7.5 10.5 12 15l4.5-4.5" {_STROKE}/>'
        f'<path d="M4 15.5v3A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5v-3" {_STROKE}/>'
    ),
    "shield-check": (
        f'<path d="M12 3 4.5 6.2v5.3c0 4.8 3.2 8.8 7.5 10 4.3-1.2 7.5-5.2 7.5-10V6.2z" {_STROKE}/>'
        f'<path d="m9 12 2.2 2.2 4.3-4.4" {_STROKE}/>'
    ),
    "volume": (
        f'<path d="M11 5 6 9H3v6h3l5 4z" {_STROKE}/>'
        f'<path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 6a8.5 8.5 0 0 1 0 12" {_STROKE}/>'
    ),
    "mute": (
        f'<path d="M11 5 6 9H3v6h3l5 4z" {_STROKE}/>'
        f'<path d="m21 9-5 6M16 9l5 6" {_STROKE}/>'
    ),
    "search": (
        f'<circle cx="11" cy="11" r="6.5" {_STROKE}/>'
        f'<path d="m20 20-3.8-3.8" {_STROKE}/>'
    ),
    "lock": (
        f'<rect x="5" y="11" width="14" height="10" rx="2" {_STROKE}/>'
        f'<path d="M8 11V7a4 4 0 0 1 8 0v4" {_STROKE}/>'
    ),
    "unlock": (
        f'<rect x="5" y="11" width="14" height="10" rx="2" {_STROKE}/>'
        f'<path d="M8 11V7a4 4 0 0 1 7.6-1.5" {_STROKE}/>'
    ),
    "magnet": (
        f'<path d="M6 3v7a6 6 0 0 0 12 0V3M6 7h4M14 7h4" {_STROKE}/>'
    ),
    "bookmark": (
        f'<path d="m19 21-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" {_STROKE}/>'
    ),
    "activity": (
        f'<path d="M22 12h-4l-3 9L9 3l-3 9H2" {_STROKE}/>'
    ),
    "home": (
        f'<path d="m3 9.5 9-7 9 7V20a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 20z" {_STROKE}/>'
        f'<path d="M9 21.5V12h6v9.5" {_STROKE}/>'
    ),
    "grid": (
        f'<rect x="3.5" y="3.5" width="7" height="7" rx="1.5" {_STROKE}/>'
        f'<rect x="13.5" y="3.5" width="7" height="7" rx="1.5" {_STROKE}/>'
        f'<rect x="3.5" y="13.5" width="7" height="7" rx="1.5" {_STROKE}/>'
        f'<rect x="13.5" y="13.5" width="7" height="7" rx="1.5" {_STROKE}/>'
    ),
    "list": (
        f'<path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01" {_STROKE}/>'
    ),
    "more-vertical": (
        '<circle cx="12" cy="5.5" r="1.5" fill="{color}"/>'
        '<circle cx="12" cy="12" r="1.5" fill="{color}"/>'
        '<circle cx="12" cy="18.5" r="1.5" fill="{color}"/>'
    ),
    "cpu": (
        f'<rect x="5" y="5" width="14" height="14" rx="2" {_STROKE}/>'
        f'<rect x="9" y="9" width="6" height="6" rx="1" {_STROKE}/>'
        f'<path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" {_STROKE}/>'
    ),
    "memory": (
        f'<rect x="2.5" y="6.5" width="19" height="9" rx="1.5" {_STROKE}/>'
        f'<path d="M6 10v2M10 10v2M14 10v2M18 10v2M6 15.5V18M12 15.5V18M18 15.5V18" {_STROKE}/>'
    ),
    "hard-drive": (
        f'<rect x="2.5" y="5.5" width="19" height="13" rx="2" {_STROKE}/>'
        f'<path d="M2.5 13h19" {_STROKE}/>'
        '<circle cx="6.5" cy="15.8" r="1.1" fill="{color}"/>'
        '<circle cx="10" cy="15.8" r="1.1" fill="{color}"/>'
    ),
    "chevron-right": (
        f'<path d="m9 6 6 6-6 6" {_STROKE}/>'
    ),
    "chevron-left": (
        f'<path d="m15 18-6-6 6-6" {_STROKE}/>'
    ),
    "edit": (
        f'<path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" {_STROKE}/>'
    ),
    "tasks": (
        f'<rect x="3.5" y="3.5" width="17" height="17" rx="2.2" {_STROKE}/>'
        f'<path d="m8 12 2.5 2.5L16 9" {_STROKE}/>'
    ),
}

_DISABLED_COLOR = C.COLOR_TEXT_MUTED


@lru_cache(maxsize=256)
def _render_svg(name: str, size: int, color: str, dpr: float) -> QPixmap:
    body = _ICONS.get(name)
    if body is None:
        body = _ICONS["info"]
    body = body.replace("{color}", color)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'width="{size}" height="{size}">{body}</svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return pm


def pixmap(name: str, size: int = 20, color: str = C.COLOR_TEXT_PRIMARY, dpr: float = 2.0) -> QPixmap:
    return _render_svg(name, size, color, dpr)


def icon(name: str, color: str = C.COLOR_TEXT_SECONDARY, size: int = 18) -> QIcon:
    ic = QIcon()
    ic.addPixmap(_render_svg(name, size, color, 2.0), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(_render_svg(name, size, _DISABLED_COLOR, 2.0), QIcon.Mode.Disabled, QIcon.State.Off)
    return ic


def app_icon() -> QIcon:
    """Uygulama simgesi: bordo zemin + mikrofon."""
    ico_path = Path(__file__).resolve().parent.parent / "assets" / "vox_studio.ico"
    if ico_path.exists():
        return QIcon(str(ico_path))
    ic = QIcon()
    for size in (16, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(QColor(C.COLOR_BG_DARKEST))
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QColor(C.COLOR_ACCENT))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, size, size, size * 0.22, size * 0.22)
        renderer = QSvgRenderer(
            QByteArray(
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
                f'<rect x="9" y="3.2" width="6" height="11" rx="3" fill="#F7F3F3"/>'
                f'<path d="M5.8 11a6.2 6.2 0 0 0 12.4 0" fill="none" stroke="#F7F3F3" '
                f'stroke-width="1.8" stroke-linecap="round"/>'
                f'<path d="M12 17.2V20.6M9 20.6h6" stroke="#F7F3F3" stroke-width="1.8" '
                f'stroke-linecap="round"/></svg>'.encode("utf-8")
            )
        )
        renderer.render(painter, QRectF(size * 0.14, size * 0.14, size * 0.72, size * 0.72))
        painter.end()
        ic.addPixmap(pm)
    return ic
