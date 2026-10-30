"""
Henüz uygulanmamış modüller için dürüst yer tutucu sayfalar.

Bu sayfalar çalışıyormuş gibi görünen düğmeler veya örnek veri içermez;
planlanan işlevleri ve bilinen kısıtları açıkça listeler.
"""

from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core import constants as C
from ui import icons


class PlaceholderView(QWidget):
    def __init__(
        self,
        title: str,
        description: str,
        icon_name: str,
        planned: Iterable[str],
        constraints: Iterable[str] = (),
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._build(title, description, icon_name, list(planned), list(constraints))

    def _build(
        self,
        title: str,
        description: str,
        icon_name: str,
        planned: list[str],
        constraints: list[str],
    ) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(14)

        header = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(icons.pixmap(icon_name, 36, C.COLOR_ACCENT_HOVER))
        icon.setFixedSize(44, 44)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(icon)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("ViewTitle")
        subtitle = QLabel(description)
        subtitle.setObjectName("ViewSubtitle")
        subtitle.setWordWrap(True)
        title_box.addWidget(title_label)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)

        badge = QLabel("GELİŞTİRİLİYOR")
        badge.setObjectName("DevBadge")
        header.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll, 1)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 4, 8, 4)
        layout.setSpacing(12)

        notice = QFrame()
        notice.setObjectName("NoticePanel")
        notice_layout = QHBoxLayout(notice)
        notice_layout.setContentsMargins(14, 12, 14, 12)
        notice_layout.setSpacing(10)
        warn_icon = QLabel()
        warn_icon.setPixmap(icons.pixmap("warning", 18, C.COLOR_VU_YELLOW))
        notice_layout.addWidget(warn_icon, 0, Qt.AlignmentFlag.AlignTop)
        notice_text = QLabel(
            "Bu modül henüz uygulanmadı. Sayfa, çalışan bir özellik varmış izlenimi "
            "vermek için sahte veri veya işlevsiz düğme göstermez. Aşağıdaki liste "
            "planlanan işlevlerdir; tamamlanma sırası değişebilir."
        )
        notice_text.setWordWrap(True)
        notice_text.setObjectName("Muted")
        notice_layout.addWidget(notice_text, 1)
        layout.addWidget(notice)

        if planned:
            plan_card = QFrame()
            plan_card.setObjectName("Card")
            plan_layout = QVBoxLayout(plan_card)
            plan_layout.setContentsMargins(18, 14, 18, 16)
            plan_layout.setSpacing(6)
            plan_title = QLabel("PLANLANAN İŞLEVLER")
            plan_title.setObjectName("SectionTitle")
            plan_layout.addWidget(plan_title)
            for text in planned:
                row = QLabel(f"•  {text}")
                row.setObjectName("Muted")
                row.setWordWrap(True)
                plan_layout.addWidget(row)
            layout.addWidget(plan_card)

        if constraints:
            constraint_card = QFrame()
            constraint_card.setObjectName("Card")
            constraint_layout = QVBoxLayout(constraint_card)
            constraint_layout.setContentsMargins(18, 14, 18, 16)
            constraint_layout.setSpacing(6)
            constraint_title = QLabel("BİLİNEN KISITLAR VE GEREKSİNİMLER")
            constraint_title.setObjectName("SectionTitle")
            constraint_layout.addWidget(constraint_title)
            for text in constraints:
                row = QLabel(f"•  {text}")
                row.setObjectName("MutedSmall")
                row.setWordWrap(True)
                constraint_layout.addWidget(row)
            layout.addWidget(constraint_card)

        layout.addStretch(1)
        scroll.setWidget(content)
