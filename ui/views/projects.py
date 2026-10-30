"""
VOX Studio — Profesyonel Ana Sayfa (Home & Proje Merkezi) Görünümü.

Bölüm 1–10 kapsamında:
- Siyah-kırmızı tasarım sistemiyle uyumlu sade Hoş Geldin Banner'ı
- 4 Hızlı İşlem Kartı (Video Düzenle, Ses Ayrıştırma, Dublaj Stüdyosu, Otomatik Altyazı)
- Gerçek veriye bağlı Son Projeler alanı (Arama, Sıralama, Izgara/Liste görünümü,
  gerçek küçük resim veya vektör önizleme, ⋮ bağlam menüsü: Aç, Yeniden Adlandır,
  Kopyasını Oluştur, Proje Konumunu Aç, Sil)
- Sağ Panel: Gerçek TaskManager görevlerine bağlı Devam Eden İşlemler ve gerçek
  donanım/işletim sistemi sorgularına bağlı Sistem Durumu (GPU, RAM, Depolama)
- Alt Hızlı Erişim Şeridi ve Yeni Proje Oluşturma Sihirbazı (NewProjectDialog)
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPoint, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core import constants as C
from core.diagnostics import SystemDiagnostics, TaskEntry
from core.format import format_duration, format_file_size
from ui import icons
from ui.context import AppContext


class HeroBannerWidget(QFrame):
    """Bölüm 3.C: Koyu kırmızı-siyah vurgulu, sade ve modern Hoş Geldin Banner'ı."""

    newProjectClicked = Signal()
    openProjectClicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("HeroBanner")
        self.setMinimumHeight(168)
        self.setMaximumHeight(188)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(20)

        left = QVBoxLayout()
        left.setSpacing(6)

        title_lbl = QLabel(
            f'<span style="color:{C.COLOR_TEXT_PRIMARY};">Projeni oluşturmaya </span>'
            f'<span style="color:{C.COLOR_ACCENT};">başla.</span>'
        )
        title_lbl.setTextFormat(Qt.TextFormat.RichText)
        title_lbl.setStyleSheet("font-size: 26px; font-weight: 800; background: transparent;")
        left.addWidget(title_lbl)

        sub_lbl = QLabel(
            "Video düzenle, sesleri ayrıştır ve dublaj çalışmalarını tek bir yerde yönet."
        )
        sub_lbl.setWordWrap(True)
        sub_lbl.setStyleSheet(
            f"color: {C.COLOR_TEXT_SECONDARY}; font-size: 13px; background: transparent;"
        )
        left.addWidget(sub_lbl)
        left.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.new_project_btn = QPushButton("  + Yeni Proje")
        self.new_project_btn.setProperty("variant", "primary")
        self.new_project_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_project_btn.setMinimumHeight(38)
        self.new_project_btn.setStyleSheet(
            f"QPushButton {{ background: {C.COLOR_ACCENT}; border: 1px solid {C.COLOR_ACCENT_HOVER}; "
            "border-radius: 9px; padding: 7px 18px; font-size: 13px; font-weight: 700; color: #FFFFFF; }}"
            f"QPushButton:hover {{ background: {C.COLOR_ACCENT_HOVER}; }}"
        )
        self.new_project_btn.clicked.connect(self.newProjectClicked.emit)
        btn_row.addWidget(self.new_project_btn)

        self.open_project_btn = QPushButton("  Mevcut Projeyi Aç")
        self.open_project_btn.setIcon(icons.icon("folder", C.COLOR_TEXT_PRIMARY, size=15))
        self.open_project_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_project_btn.setMinimumHeight(38)
        self.open_project_btn.setStyleSheet(
            f"QPushButton {{ background: rgba(21, 22, 25, 210); border: 1px solid {C.COLOR_BORDER_LIGHT}; "
            f"border-radius: 9px; padding: 7px 16px; font-size: 12px; font-weight: 600; color: {C.COLOR_TEXT_PRIMARY}; }}"
            f"QPushButton:hover {{ background: {C.COLOR_BG_HOVER}; border-color: {C.COLOR_ACCENT}; }}"
        )
        self.open_project_btn.clicked.connect(self.openProjectClicked.emit)
        btn_row.addWidget(self.open_project_btn)

        btn_row.addStretch(1)
        left.addLayout(btn_row)
        layout.addLayout(left, 1)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())

        # Sağ tarafta zarif akustik dalga / kırmızı ışık çizgileri (stok fotoğraf kullanmadan modern görünüm)
        pen = QPen(QColor(229, 57, 69, 42), 1.6)
        painter.setPen(pen)
        for idx in range(5):
            offset = idx * 14.0
            path = QPainterPath()
            path.moveTo(w * 0.45, h + 10.0)
            path.cubicTo(
                w * 0.62,
                h * (0.65 - idx * 0.06),
                w * 0.78,
                h * (0.88 - idx * 0.08),
                w + 10.0,
                h * (0.18 + offset / max(1.0, h)),
            )
            painter.drawPath(path)

        glow_grad = QLinearGradient(w * 0.65, 0.0, w, h)
        glow_grad.setColorAt(0.0, QColor(229, 57, 69, 0))
        glow_grad.setColorAt(0.65, QColor(229, 57, 69, 24))
        glow_grad.setColorAt(1.0, QColor(255, 70, 82, 38))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow_grad)
        painter.drawRoundedRect(QRectF(w * 0.55, 1.0, w * 0.45 - 1.0, h - 2.0), 14.0, 14.0)
        painter.end()


class ActionCardWidget(QFrame):
    """Bölüm 3.D: 4 Hızlı İşlem Kartından her biri."""

    clicked = Signal()

    def __init__(
        self,
        title: str,
        subtitle: str,
        icon_name: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("ActionCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(104)
        self.setMaximumHeight(116)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(8)

        badge = QLabel()
        badge.setObjectName("IconBadge")
        badge.setFixedSize(34, 34)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setPixmap(icons.pixmap(icon_name, 18, C.COLOR_ACCENT_HOVER))
        top.addWidget(badge)
        top.addStretch(1)

        arrow = QLabel()
        arrow.setPixmap(icons.pixmap("chevron-right", 16, C.COLOR_TEXT_MUTED))
        arrow.setStyleSheet("background: transparent;")
        top.addWidget(arrow)
        lay.addLayout(top)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 14px; font-weight: 700; background: transparent;"
        )
        lay.addWidget(title_lbl)

        sub_lbl = QLabel(subtitle)
        sub_lbl.setStyleSheet(
            f"color: {C.COLOR_TEXT_MUTED}; font-size: 11px; background: transparent;"
        )
        sub_lbl.setWordWrap(True)
        lay.addWidget(sub_lbl)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class ProjectPreviewBanner(QWidget):
    """Proje kartının üst kısmındaki gerçek küçük resim veya koyu stüdyo önizleme alanı."""

    def __init__(
        self,
        project_data: dict,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.project_data = project_data
        self.setFixedHeight(106)
        self._pixmap: Optional[QPixmap] = None
        thumb = project_data.get("thumbnail_path")
        if thumb and Path(thumb).exists():
            pm = QPixmap(str(thumb))
            if not pm.isNull():
                self._pixmap = pm

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())

        clip_path = QPainterPath()
        clip_path.addRoundedRect(QRectF(0, 0, w, h + 10), 11.0, 11.0)
        painter.setClipPath(clip_path)

        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                int(w),
                int(h),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            sx = max(0, (scaled.width() - int(w)) // 2)
            sy = max(0, (scaled.height() - int(h)) // 2)
            painter.drawPixmap(0, 0, scaled, sx, sy, int(w), int(h))
            # Alt karartma gradyanı
            grad = QLinearGradient(0, h * 0.45, 0, h)
            grad.setColorAt(0.0, QColor(9, 10, 12, 0))
            grad.setColorAt(1.0, QColor(9, 10, 12, 210))
            painter.fillRect(QRectF(0, 0, w, h), grad)
        else:
            # Gerçek küçük resim yoksa temiz koyu-kırmızı stüdyo dalga önizlemesi
            bg = QLinearGradient(0, 0, w, h)
            bg.setColorAt(0.0, QColor("#121318"))
            bg.setColorAt(0.6, QColor("#1B1216"))
            bg.setColorAt(1.0, QColor("#0E0F13"))
            painter.fillRect(QRectF(0, 0, w, h), bg)

            # Zarif ses dalgası çubukları
            bars = 28
            bar_w = 3.0
            gap = (w - 36.0) / max(1, bars)
            name_seed = sum(ord(c) for c in str(self.project_data.get("name", "VOX")))
            for i in range(bars):
                x = 18.0 + i * gap
                norm = ((i * 7 + name_seed) % 19) / 19.0
                bh = 10.0 + norm * 38.0
                y = (h - bh) / 2.0
                col = QColor(C.COLOR_ACCENT) if i % 4 == 0 else QColor(229, 57, 69, 95)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(col)
                painter.drawRoundedRect(QRectF(x, y, bar_w, bh), 1.5, 1.5)

        # Eksik medya uyarısı (varsa sol üstte)
        missing_cnt = int(self.project_data.get("missing_media_count") or 0)
        if missing_cnt > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(241, 196, 15, 220))
            painter.drawRoundedRect(QRectF(8, 8, 92, 20), 5.0, 5.0)
            painter.setPen(QColor("#111113"))
            f = QFont(self.font())
            f.setPointSize(8)
            f.setBold(True)
            painter.setFont(f)
            painter.drawText(
                QRectF(8, 8, 92, 20),
                Qt.AlignmentFlag.AlignCenter,
                f"Eksik Medya ({missing_cnt})",
            )

        # Sağ altta süre rozeti
        dur = float(self.project_data.get("duration_sec") or 0.0)
        dur_txt = format_duration(dur) if dur > 0 else (self.project_data.get("resolution") or "1920x1080")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(9, 10, 12, 205))
        pill_w = max(46.0, len(dur_txt) * 6.5 + 14.0)
        pill_rect = QRectF(w - pill_w - 8.0, h - 26.0, pill_w, 19.0)
        painter.drawRoundedRect(pill_rect, 5.0, 5.0)
        painter.setPen(QColor(C.COLOR_TEXT_PRIMARY))
        f2 = QFont("Consolas", 8)
        f2.setBold(True)
        painter.setFont(f2)
        painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, dur_txt)
        painter.end()


class ProjectCardWidget(QFrame):
    """Bölüm 4: Gerçek proje verisini gösteren ızgara veya liste kartı."""

    openRequested = Signal(str)
    renameRequested = Signal(str)
    duplicateRequested = Signal(str)
    openFolderRequested = Signal(str)
    deleteRequested = Signal(str)

    def __init__(
        self,
        project_data: dict,
        list_mode: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.project_data = project_data
        self.project_path = str(project_data.get("path", ""))
        self.setObjectName("ProjectCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        ptype = str(project_data.get("project_type") or "Video Projesi")
        if "dublaj" in ptype.lower():
            icon_name = "mic"
        elif "ayrıştırma" in ptype.lower() or "stem" in ptype.lower():
            icon_name = "waveform"
        elif "ses" in ptype.lower():
            icon_name = "music"
        else:
            icon_name = "video"

        ts = float(project_data.get("updated_at") or 0.0)
        dt_str = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M") if ts > 0 else "—"

        if list_mode:
            self.setFixedHeight(60)
            row = QHBoxLayout(self)
            row.setContentsMargins(14, 8, 14, 8)
            row.setSpacing(12)

            badge = QLabel()
            badge.setObjectName("IconBadge")
            badge.setFixedSize(34, 34)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setPixmap(icons.pixmap(icon_name, 16, C.COLOR_ACCENT_HOVER))
            row.addWidget(badge)

            name_col = QVBoxLayout()
            name_col.setSpacing(2)
            name_lbl = QLabel(str(project_data.get("name", "Adsız Proje")))
            name_lbl.setStyleSheet(
                f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 13px; font-weight: 700; background: transparent;"
            )
            sub_lbl = QLabel(f"{ptype}  ·  {project_data.get('resolution', '1920x1080')}")
            sub_lbl.setStyleSheet(
                f"color: {C.COLOR_TEXT_MUTED}; font-size: 11px; background: transparent;"
            )
            name_col.addWidget(name_lbl)
            name_col.addWidget(sub_lbl)
            row.addLayout(name_col, 1)

            date_lbl = QLabel(dt_str)
            date_lbl.setStyleSheet(
                f"color: {C.COLOR_TEXT_SECONDARY}; font-size: 11px; background: transparent;"
            )
            row.addWidget(date_lbl)

            menu_btn = self._build_menu_button()
            row.addWidget(menu_btn)
        else:
            self.setFixedHeight(196)
            col = QVBoxLayout(self)
            col.setContentsMargins(0, 0, 0, 10)
            col.setSpacing(8)

            preview = ProjectPreviewBanner(project_data, self)
            col.addWidget(preview)

            body = QVBoxLayout()
            body.setContentsMargins(12, 2, 10, 0)
            body.setSpacing(6)

            mid = QHBoxLayout()
            mid.setSpacing(8)

            badge = QLabel()
            badge.setObjectName("IconBadge")
            badge.setFixedSize(28, 28)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setPixmap(icons.pixmap(icon_name, 14, C.COLOR_ACCENT_HOVER))
            mid.addWidget(badge)

            title_col = QVBoxLayout()
            title_col.setSpacing(1)
            raw_name = str(project_data.get("name", "Adsız Proje"))
            name_lbl = QLabel(raw_name)
            metrics = QFontMetrics(name_lbl.font())
            name_lbl.setText(metrics.elidedText(raw_name, Qt.TextElideMode.ElideRight, 150))
            name_lbl.setToolTip(f"{raw_name}\n{self.project_path}")
            name_lbl.setStyleSheet(
                f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 13px; font-weight: 700; background: transparent;"
            )
            type_lbl = QLabel(ptype)
            type_lbl.setStyleSheet(
                f"color: {C.COLOR_TEXT_MUTED}; font-size: 11px; background: transparent;"
            )
            title_col.addWidget(name_lbl)
            title_col.addWidget(type_lbl)
            mid.addLayout(title_col, 1)

            menu_btn = self._build_menu_button()
            mid.addWidget(menu_btn)
            body.addLayout(mid)

            foot = QHBoxLayout()
            foot.setSpacing(5)
            clk = QLabel()
            clk.setPixmap(icons.pixmap("clock", 12, C.COLOR_TEXT_MUTED))
            clk.setStyleSheet("background: transparent;")
            foot.addWidget(clk)
            date_lbl = QLabel(dt_str)
            date_lbl.setStyleSheet(
                f"color: {C.COLOR_TEXT_MUTED}; font-size: 11px; background: transparent;"
            )
            foot.addWidget(date_lbl)
            foot.addStretch(1)
            body.addLayout(foot)

            col.addLayout(body)

    def _build_menu_button(self) -> QPushButton:
        btn = QPushButton()
        btn.setProperty("variant", "ghost")
        btn.setFixedSize(26, 26)
        btn.setIcon(icons.icon("more-vertical", C.COLOR_TEXT_SECONDARY, size=15))
        btn.setToolTip("Proje işlemleri")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self._show_context_menu)
        return btn

    def _show_context_menu(self) -> None:
        menu = QMenu(self)
        act_open = menu.addAction(icons.icon("folder", C.COLOR_TEXT_PRIMARY, 14), "Projeyi Aç")
        act_rename = menu.addAction(icons.icon("edit", C.COLOR_TEXT_PRIMARY, 14), "Yeniden Adlandır")
        act_dup = menu.addAction(icons.icon("copy", C.COLOR_TEXT_PRIMARY, 14), "Kopyasını Oluştur")
        act_loc = menu.addAction(
            icons.icon("open-external", C.COLOR_TEXT_PRIMARY, 14), "Proje Konumunu Aç"
        )
        menu.addSeparator()
        act_del = menu.addAction(icons.icon("trash", C.COLOR_ACCENT_HOVER, 14), "Projeyi Sil")

        chosen = menu.exec(self.mapToGlobal(QPoint(self.width() - 28, 36)))
        if chosen == act_open:
            self.openRequested.emit(self.project_path)
        elif chosen == act_rename:
            self.renameRequested.emit(self.project_path)
        elif chosen == act_dup:
            self.duplicateRequested.emit(self.project_path)
        elif chosen == act_loc:
            self.openFolderRequested.emit(self.project_path)
        elif chosen == act_del:
            self.deleteRequested.emit(self.project_path)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.openRequested.emit(self.project_path)
        elif event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu()
        super().mouseReleaseEvent(event)


class QuickStripButton(QFrame):
    """Bölüm 6: Alt Hızlı Erişim Şeridi butonu."""

    clicked = Signal()

    def __init__(self, label: str, icon_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("QuickStripCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(48)

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 6, 12, 6)
        row.setSpacing(10)

        badge = QLabel()
        badge.setObjectName("IconBadge")
        badge.setFixedSize(28, 28)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setPixmap(icons.pixmap(icon_name, 15, C.COLOR_ACCENT_HOVER))
        row.addWidget(badge)

        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        row.addWidget(lbl, 1)

        arr = QLabel()
        arr.setPixmap(icons.pixmap("chevron-right", 14, C.COLOR_TEXT_MUTED))
        arr.setStyleSheet("background: transparent;")
        row.addWidget(arr)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class NewProjectDialog(QDialog):
    """Bölüm 7: Yeni Proje Oluşturma Penceresi."""

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Yeni Proje Oluştur")
        self.setMinimumWidth(520)
        self.selected_mode = "blank_video"
        self.existing_voxproj_path: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 20)
        root.setSpacing(14)

        head = QLabel("Yeni Proje Oluştur")
        head.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {C.COLOR_TEXT_PRIMARY};")
        root.addWidget(head)

        sub = QLabel(
            "Proje çalışma modunu, kayıt konumunu ve başlangıç ayarlarını belirleyin."
        )
        sub.setStyleSheet(f"color: {C.COLOR_TEXT_SECONDARY}; font-size: 12px;")
        root.addWidget(sub)

        # Başlangıç türü seçimi
        mode_card = QFrame()
        mode_card.setObjectName("Card")
        mode_lay = QVBoxLayout(mode_card)
        mode_lay.setContentsMargins(14, 12, 14, 12)
        mode_lay.setSpacing(8)

        mode_title = QLabel("BAŞLANGIÇ TÜRÜ")
        mode_title.setObjectName("SectionTitle")
        mode_lay.addWidget(mode_title)

        self.mode_group = QButtonGroup(self)
        modes = [
            ("blank_video", "Boş video projesi", "Çok katmanlı video ve ses timeline'ı ile başla"),
            ("from_media", "Mevcut medyadan proje oluştur", "Bir video veya ses dosyası seçerek başla"),
            ("stem_work", "Ses ayrıştırma çalışması başlat", "AI ile vokal, diyalog ve müzik ayrıştırmaya geç"),
            ("dubbing_work", "Dublaj projesi başlat", "Replik listesi, karakter ve çoklu deneme stüdyosu"),
            ("open_existing", "Mevcut projeyi aç (.voxproj)", "Diskteki kayıtlı bir .voxproj dosyasını yükle"),
        ]
        self._radio_buttons: Dict[str, QRadioButton] = {}
        for idx, (m_key, m_label, m_desc) in enumerate(modes):
            rb = QRadioButton(f"{m_label}  —  {m_desc}")
            rb.setProperty("mode_key", m_key)
            if idx == 0:
                rb.setChecked(True)
            rb.toggled.connect(self._on_mode_toggled)
            self.mode_group.addButton(rb)
            self._radio_buttons[m_key] = rb
            mode_lay.addWidget(rb)

        root.addWidget(mode_card)

        # Temel proje bilgileri
        self.details_card = QFrame()
        self.details_card.setObjectName("Card")
        d_lay = QVBoxLayout(self.details_card)
        d_lay.setContentsMargins(14, 12, 14, 12)
        d_lay.setSpacing(10)

        name_row = QHBoxLayout()
        name_lbl = QLabel("Proje Adı:")
        name_lbl.setFixedWidth(115)
        self.name_edit = QLineEdit("Yeni Video Projesi")
        name_row.addWidget(name_lbl)
        name_row.addWidget(self.name_edit, 1)
        d_lay.addLayout(name_row)

        loc_row = QHBoxLayout()
        loc_lbl = QLabel("Kayıt Konumu:")
        loc_lbl.setFixedWidth(115)
        self.loc_edit = QLineEdit(str(C.PROJECTS_DIR))
        loc_btn = QPushButton("Gözat...")
        loc_btn.clicked.connect(self._browse_folder)
        loc_row.addWidget(loc_lbl)
        loc_row.addWidget(self.loc_edit, 1)
        loc_row.addWidget(loc_btn)
        d_lay.addLayout(loc_row)

        media_row = QHBoxLayout()
        media_lbl = QLabel("Başlangıç Medyası:")
        media_lbl.setFixedWidth(115)
        self.media_edit = QLineEdit()
        self.media_edit.setPlaceholderText("İsteğe bağlı video veya ses dosyası seçin...")
        media_btn = QPushButton("Dosya Seç...")
        media_btn.clicked.connect(self._browse_media)
        media_row.addWidget(media_lbl)
        media_row.addWidget(self.media_edit, 1)
        media_row.addWidget(media_btn)
        d_lay.addLayout(media_row)

        # Katlanabilir Gelişmiş Teknik Ayarlar
        self.adv_toggle_btn = QPushButton("Gelişmiş Teknik Ayarlar (Çözünürlük, Kare Hızı, Örnekleme Hızı) ▾")
        self.adv_toggle_btn.setProperty("variant", "ghost")
        self.adv_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.adv_toggle_btn.clicked.connect(self._toggle_advanced)
        d_lay.addWidget(self.adv_toggle_btn)

        self.adv_widget = QWidget()
        adv_lay = QGridLayout(self.adv_widget)
        adv_lay.setContentsMargins(0, 4, 0, 0)
        adv_lay.setHorizontalSpacing(10)
        adv_lay.setVerticalSpacing(8)

        adv_lay.addWidget(QLabel("Çözünürlük:"), 0, 0)
        self.res_combo = QComboBox()
        self.res_combo.addItem("1920x1080 (Full HD 16:9)", (1920, 1080))
        self.res_combo.addItem("1280x720 (HD 16:9)", (1280, 720))
        self.res_combo.addItem("1080x1920 (Dikey 9:16)", (1080, 1920))
        self.res_combo.addItem("3840x2160 (4K UHD 16:9)", (3840, 2160))
        adv_lay.addWidget(self.res_combo, 0, 1)

        adv_lay.addWidget(QLabel("Kare Hızı (FPS):"), 1, 0)
        self.fps_combo = QComboBox()
        self.fps_combo.addItem("30 fps (Standart)", 30.0)
        self.fps_combo.addItem("24 fps (Sinema)", 24.0)
        self.fps_combo.addItem("25 fps (PAL / Yayın)", 25.0)
        self.fps_combo.addItem("60 fps (Akıcı)", 60.0)
        adv_lay.addWidget(self.fps_combo, 1, 1)

        adv_lay.addWidget(QLabel("Ses Örnekleme Hızı:"), 2, 0)
        self.sr_combo = QComboBox()
        self.sr_combo.addItem("48000 Hz (Profesyonel Video/Dublaj)", 48000)
        self.sr_combo.addItem("44100 Hz (Müzik Standardı)", 44100)
        adv_lay.addWidget(self.sr_combo, 2, 1)

        self.adv_widget.setVisible(False)
        d_lay.addWidget(self.adv_widget)
        root.addWidget(self.details_card)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("İptal")
        cancel_btn.clicked.connect(self.reject)
        self.create_btn = QPushButton("Projeyi Oluştur")
        self.create_btn.setProperty("variant", "primary")
        self.create_btn.setMinimumHeight(36)
        self.create_btn.clicked.connect(self._on_confirm)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self.create_btn)
        root.addLayout(btn_row)

    def _toggle_advanced(self) -> None:
        vis = not self.adv_widget.isVisible()
        self.adv_widget.setVisible(vis)
        self.adv_toggle_btn.setText(
            "Gelişmiş Teknik Ayarlar (Çözürlük, Kare Hızı, Örnekleme Hızı) ▴"
            if vis
            else "Gelişmiş Teknik Ayarlar (Çözünürlük, Kare Hızı, Örnekleme Hızı) ▾"
        )

    def _on_mode_toggled(self, checked: bool) -> None:
        if not checked:
            return
        btn = self.mode_group.checkedButton()
        if not btn:
            return
        mode = str(btn.property("mode_key") or "blank_video")
        self.selected_mode = mode
        if mode == "open_existing":
            self.details_card.setEnabled(False)
            self.create_btn.setText("Mevcut Projeyi Seç...")
        else:
            self.details_card.setEnabled(True)
            self.create_btn.setText("Projeyi Oluştur")
            default_names = {
                "blank_video": "Yeni Video Projesi",
                "from_media": "Medya Düzenleme Projesi",
                "stem_work": "Ses Ayrıştırma Projesi",
                "dubbing_work": "Dublaj Projesi",
            }
            if self.name_edit.text().strip() in default_names.values() or not self.name_edit.text().strip():
                self.name_edit.setText(default_names.get(mode, "Yeni Video Projesi"))

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Proje Kayıt Klasörü Seç", self.loc_edit.text() or str(C.PROJECTS_DIR)
        )
        if folder:
            self.loc_edit.setText(folder)

    def _browse_media(self) -> None:
        exts = " ".join(f"*{e}" for e in C.VIDEO_EXTENSIONS + C.AUDIO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Başlangıç Medyası Seç",
            str(self.ctx.recordings_dir()),
            f"Medya Dosyaları ({exts});;Tüm Dosyalar (*)",
        )
        if path:
            self.media_edit.setText(path)
            if self.name_edit.text().strip() in ("Yeni Video Projesi", "Medya Düzenleme Projesi"):
                self.name_edit.setText(Path(path).stem)

    def _on_confirm(self) -> None:
        if self.selected_mode == "open_existing":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "VOX Studio Projesi Aç",
                str(C.PROJECTS_DIR),
                "VOX Studio Projesi (*.voxproj);;Tüm Dosyalar (*)",
            )
            if path:
                self.existing_voxproj_path = path
                self.accept()
            return

        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Eksik Bilgi", "Lütfen geçerli bir proje adı girin.")
            return
        self.accept()

    def project_parameters(self) -> dict:
        type_map = {
            "blank_video": "Video Projesi",
            "from_media": "Video Projesi",
            "stem_work": "Ses Ayrıştırma",
            "dubbing_work": "Dublaj Projesi",
        }
        res = self.res_combo.currentData() or (1920, 1080)
        fps = float(self.fps_combo.currentData() or 30.0)
        sr = int(self.sr_combo.currentData() or 48000)
        media_p = self.media_edit.text().strip() or None
        return {
            "mode": self.selected_mode,
            "name": self.name_edit.text().strip() or "Yeni Video Projesi",
            "folder_path": self.loc_edit.text().strip() or str(C.PROJECTS_DIR),
            "project_type": type_map.get(self.selected_mode, "Video Projesi"),
            "resolution": res,
            "fps": fps,
            "sample_rate": sr,
            "media_path": media_p if (media_p and Path(media_p).exists()) else None,
        }


class ProjectsView(QWidget):
    """VOX Studio Profesyonel Ana Sayfa ve Proje Merkezi."""

    recordRequested = Signal()
    openFileRequested = Signal(str)
    libraryRequested = Signal()
    editorRequested = Signal()
    stemsRequested = Signal()
    dubbingRequested = Signal()
    transcriptionRequested = Signal()
    tasksRequested = Signal()
    projectOpened = Signal(str)

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctx = ctx
        self._view_mode = "grid"  # 'grid' veya 'list'
        self._grid_columns = 3
        self._cached_projects: List[dict] = []

        # Geriye dönük uyumluluk için gizli liste referansları
        self.recent_list = QListWidget()
        self.projects_list = QListWidget()

        self._build_ui()

        self._hw_timer = QTimer(self)
        self._hw_timer.setInterval(5000)
        self._hw_timer.timeout.connect(self.refresh_system_status)
        self._hw_timer.start()

        self.refresh_recent()

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Sol / Orta Ana Kaydırılabilir İçerik Alanı
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        center_container = QWidget()
        self.center_layout = QVBoxLayout(center_container)
        self.center_layout.setContentsMargins(24, 18, 20, 20)
        self.center_layout.setSpacing(18)

        # Gizli uyumluluk rozeti (test_ui uyumu için)
        self.active_badge = QLabel("YEREL STÜDYO · HAZIR")
        self.active_badge.setObjectName("RecordBadge")
        self.active_badge.setVisible(False)

        # 1. Hoş Geldin Banner'ı (Bölüm 3.C)
        self.hero_banner = HeroBannerWidget(self)
        self.hero_banner.newProjectClicked.connect(self._create_new_project)
        self.hero_banner.openProjectClicked.connect(self._choose_project)
        self.new_project_btn = self.hero_banner.new_project_btn
        self.open_project_btn = self.hero_banner.open_project_btn
        self.center_layout.addWidget(self.hero_banner)

        # 2. Dört Hızlı İşlem Kartı (Bölüm 3.D)
        actions_row = QHBoxLayout()
        actions_row.setSpacing(12)

        self.card_video = ActionCardWidget(
            "Video Düzenle",
            "Hızlı ve kolay düzenleme",
            "scissors",
            self,
        )
        self.card_video.clicked.connect(self.editorRequested.emit)
        actions_row.addWidget(self.card_video, 1)

        self.card_stems = ActionCardWidget(
            "Ses Ayrıştırma",
            "AI ile temiz stem'ler",
            "waveform",
            self,
        )
        self.card_stems.clicked.connect(self.stemsRequested.emit)
        actions_row.addWidget(self.card_stems, 1)

        self.card_dubbing = ActionCardWidget(
            "Dublaj Stüdyosu",
            "Profesyonel seslendirme",
            "mic",
            self,
        )
        self.card_dubbing.clicked.connect(self.dubbingRequested.emit)
        actions_row.addWidget(self.card_dubbing, 1)

        self.card_subtitles = ActionCardWidget(
            "Otomatik Altyazı",
            "Konuşmayı metne dönüştür",
            "text",
            self,
        )
        self.card_subtitles.clicked.connect(self.transcriptionRequested.emit)
        actions_row.addWidget(self.card_subtitles, 1)

        self.center_layout.addLayout(actions_row)

        # 3. Son Projeler Başlık ve Kontrolleri (Bölüm 4)
        proj_header = QHBoxLayout()
        proj_header.setSpacing(10)

        proj_title = QLabel("Son Projeler")
        proj_title.setStyleSheet(
            f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 16px; font-weight: 700;"
        )
        proj_header.addWidget(proj_title)

        self.proj_count_lbl = QLabel("0")
        self.proj_count_lbl.setStyleSheet(
            f"background: {C.COLOR_BG_SURFACE}; color: {C.COLOR_TEXT_SECONDARY}; "
            f"border: 1px solid {C.COLOR_BORDER}; border-radius: 9px; padding: 1px 8px; font-size: 11px;"
        )
        proj_header.addWidget(self.proj_count_lbl)
        proj_header.addStretch(1)

        # Proje arama kutusu
        search_frame = QFrame()
        search_frame.setObjectName("TopSearchEdit")
        search_frame.setFixedWidth(190)
        search_frame.setFixedHeight(32)
        sf_lay = QHBoxLayout(search_frame)
        sf_lay.setContentsMargins(8, 0, 8, 0)
        sf_lay.setSpacing(6)
        s_ico = QLabel()
        s_ico.setPixmap(icons.pixmap("search", 13, C.COLOR_TEXT_MUTED))
        sf_lay.addWidget(s_ico)
        self.project_search_edit = QLineEdit()
        self.project_search_edit.setPlaceholderText("Proje ara...")
        self.project_search_edit.setStyleSheet(
            "background: transparent; border: none; padding: 0; font-size: 12px;"
        )
        self.project_search_edit.textChanged.connect(self._render_projects_grid)
        sf_lay.addWidget(self.project_search_edit, 1)
        proj_header.addWidget(search_frame)

        # Sıralama seçici
        self.sort_combo = QComboBox()
        self.sort_combo.setFixedHeight(32)
        self.sort_combo.addItem("Son düzenlenen", "updated_desc")
        self.sort_combo.addItem("Ada göre (A-Z)", "name_asc")
        self.sort_combo.addItem("Oluşturulma tarihi", "created_desc")
        self.sort_combo.currentIndexChanged.connect(self._render_projects_grid)
        proj_header.addWidget(self.sort_combo)

        # Izgara / Liste görünüm değiştirici
        self.grid_mode_btn = QPushButton()
        self.grid_mode_btn.setCheckable(True)
        self.grid_mode_btn.setChecked(True)
        self.grid_mode_btn.setFixedSize(32, 32)
        self.grid_mode_btn.setIcon(icons.icon("grid", C.COLOR_TEXT_PRIMARY, 15))
        self.grid_mode_btn.setToolTip("Izgara görünümü")
        self.grid_mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.grid_mode_btn.clicked.connect(lambda: self._set_view_mode("grid"))
        proj_header.addWidget(self.grid_mode_btn)

        self.list_mode_btn = QPushButton()
        self.list_mode_btn.setCheckable(True)
        self.list_mode_btn.setFixedSize(32, 32)
        self.list_mode_btn.setIcon(icons.icon("list", C.COLOR_TEXT_SECONDARY, 15))
        self.list_mode_btn.setToolTip("Liste görünümü")
        self.list_mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.list_mode_btn.clicked.connect(lambda: self._set_view_mode("list"))
        proj_header.addWidget(self.list_mode_btn)

        self.center_layout.addLayout(proj_header)

        # Projeler Izgara / Liste Konteyneri
        self.projects_container = QWidget()
        self.projects_grid = QGridLayout(self.projects_container)
        self.projects_grid.setContentsMargins(0, 0, 0, 0)
        self.projects_grid.setHorizontalSpacing(12)
        self.projects_grid.setVerticalSpacing(12)
        self.center_layout.addWidget(self.projects_container)

        self.center_layout.addStretch(1)

        # 4. Alt Hızlı Erişim Şeridi (Bölüm 6)
        quick_title = QLabel("Hızlı Erişim")
        quick_title.setStyleSheet(
            f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 14px; font-weight: 700;"
        )
        self.center_layout.addWidget(quick_title)

        quick_row = QHBoxLayout()
        quick_row.setSpacing(10)

        q_stem = QuickStripButton("Ses Ayrıştırma", "waveform", self)
        q_stem.clicked.connect(self.stemsRequested.emit)
        quick_row.addWidget(q_stem, 1)

        q_dub = QuickStripButton("Dublaj Stüdyosu", "mic", self)
        q_dub.clicked.connect(self.dubbingRequested.emit)
        quick_row.addWidget(q_dub, 1)

        q_media = QuickStripButton("Medya Yükle", "folder", self)
        q_media.clicked.connect(self._choose_file)
        quick_row.addWidget(q_media, 1)

        q_new = QuickStripButton("Yeni Proje", "plus", self)
        q_new.clicked.connect(self._create_new_project)
        quick_row.addWidget(q_new, 1)

        self.center_layout.addLayout(quick_row)

        scroll.setWidget(center_container)
        outer.addWidget(scroll, 1)

        # 5. Sağ Panel: Devam Eden İşlemler ve Sistem Durumu (Bölüm 5)
        self.right_panel = self._build_right_panel()
        outer.addWidget(self.right_panel)

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("RightPanel")
        panel.setFixedWidth(C.RIGHT_PANEL_WIDTH)

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 18, 16, 18)
        lay.setSpacing(16)

        # A. Devam Eden İşlemler Kartı
        tasks_card = QFrame()
        tasks_card.setObjectName("RightStatusCard")
        t_lay = QVBoxLayout(tasks_card)
        t_lay.setContentsMargins(16, 14, 16, 14)
        t_lay.setSpacing(10)

        t_head = QHBoxLayout()
        t_head.setSpacing(8)
        t_ico = QLabel()
        t_ico.setPixmap(icons.pixmap("activity", 16, C.COLOR_ACCENT_HOVER))
        t_head.addWidget(t_ico)
        t_title = QLabel("Devam Eden İşlemler")
        t_title.setStyleSheet(
            f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 13px; font-weight: 700;"
        )
        t_head.addWidget(t_title, 1)

        all_tasks_btn = QPushButton("Tümü")
        all_tasks_btn.setProperty("variant", "link")
        all_tasks_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        all_tasks_btn.clicked.connect(self.tasksRequested.emit)
        t_head.addWidget(all_tasks_btn)
        t_lay.addLayout(t_head)

        self.tasks_list_box = QVBoxLayout()
        self.tasks_list_box.setSpacing(8)
        t_lay.addLayout(self.tasks_list_box)
        lay.addWidget(tasks_card)

        # B. Sistem Durumu Kartı (Gerçek Donanım Verileri)
        sys_card = QFrame()
        sys_card.setObjectName("RightStatusCard")
        s_lay = QVBoxLayout(sys_card)
        s_lay.setContentsMargins(16, 14, 16, 16)
        s_lay.setSpacing(12)

        s_head = QHBoxLayout()
        s_head.setSpacing(8)
        s_ico = QLabel()
        s_ico.setPixmap(icons.pixmap("info", 16, C.COLOR_TEXT_SECONDARY))
        s_head.addWidget(s_ico)
        s_title = QLabel("Sistem Durumu")
        s_title.setStyleSheet(
            f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 13px; font-weight: 700;"
        )
        s_head.addWidget(s_title, 1)
        s_lay.addLayout(s_head)

        # Genel sağlık özeti kutusu
        health_box = QFrame()
        health_box.setObjectName("TaskItemCard")
        hb_lay = QHBoxLayout(health_box)
        hb_lay.setContentsMargins(12, 10, 12, 10)
        hb_lay.setSpacing(10)

        self.health_icon_lbl = QLabel()
        self.health_icon_lbl.setFixedSize(28, 28)
        self.health_icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.health_icon_lbl.setStyleSheet(
            "background: rgba(46, 204, 113, 28); border-radius: 14px;"
        )
        self.health_icon_lbl.setPixmap(icons.pixmap("check", 16, C.COLOR_VU_GREEN))
        hb_lay.addWidget(self.health_icon_lbl)

        h_txt_col = QVBoxLayout()
        h_txt_col.setSpacing(1)
        self.health_title_lbl = QLabel("Sistem hazır")
        self.health_title_lbl.setStyleSheet(
            f"color: {C.COLOR_VU_GREEN}; font-size: 12px; font-weight: 700; background: transparent;"
        )
        self.health_sub_lbl = QLabel("Tüm ses ve video motorları sorunsuz çalışıyor.")
        self.health_sub_lbl.setWordWrap(True)
        self.health_sub_lbl.setStyleSheet(
            f"color: {C.COLOR_TEXT_SECONDARY}; font-size: 11px; background: transparent;"
        )
        h_txt_col.addWidget(self.health_title_lbl)
        h_txt_col.addWidget(self.health_sub_lbl)
        hb_lay.addLayout(h_txt_col, 1)
        s_lay.addWidget(health_box)

        # 1. GPU Satırı
        self.gpu_title_lbl, self.gpu_val_lbl, self.gpu_sub_lbl, self.gpu_bar = self._create_metric_row(
            s_lay, "cpu", "GPU"
        )
        # 2. RAM Satırı
        self.ram_title_lbl, self.ram_val_lbl, self.ram_sub_lbl, self.ram_bar = self._create_metric_row(
            s_lay, "memory", "RAM"
        )
        # 3. Depolama Satırı
        self.disk_title_lbl, self.disk_val_lbl, self.disk_sub_lbl, self.disk_bar = self._create_metric_row(
            s_lay, "hard-drive", "Depolama"
        )

        lay.addWidget(sys_card)
        lay.addStretch(1)
        return panel

    def _create_metric_row(
        self, parent_layout: QVBoxLayout, icon_name: str, label_text: str
    ) -> Tuple[QLabel, QLabel, QLabel, QProgressBar]:
        box = QWidget()
        b_lay = QVBoxLayout(box)
        b_lay.setContentsMargins(0, 2, 0, 2)
        b_lay.setSpacing(5)

        top = QHBoxLayout()
        top.setSpacing(8)
        ico = QLabel()
        ico.setFixedSize(26, 26)
        ico.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ico.setStyleSheet(
            f"background: {C.COLOR_BG_SURFACE}; border: 1px solid {C.COLOR_BORDER}; border-radius: 6px;"
        )
        ico.setPixmap(icons.pixmap(icon_name, 14, C.COLOR_TEXT_SECONDARY))
        top.addWidget(ico)

        mid_col = QVBoxLayout()
        mid_col.setSpacing(1)
        title_row = QHBoxLayout()
        t_lbl = QLabel(label_text)
        t_lbl.setStyleSheet(f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 12px; font-weight: 700;")
        v_lbl = QLabel("—")
        v_lbl.setStyleSheet(f"color: {C.COLOR_TEXT_SECONDARY}; font-size: 11px;")
        title_row.addWidget(t_lbl)
        title_row.addStretch(1)
        title_row.addWidget(v_lbl)
        mid_col.addLayout(title_row)

        sub_lbl = QLabel("—")
        sub_lbl.setStyleSheet(f"color: {C.COLOR_TEXT_MUTED}; font-size: 11px;")
        mid_col.addWidget(sub_lbl)
        top.addLayout(mid_col, 1)
        b_lay.addLayout(top)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(5)
        b_lay.addWidget(bar)

        parent_layout.addWidget(box)
        return t_lbl, v_lbl, sub_lbl, bar

    # ================================================================ dinamik paneller
    def refresh_tasks_panel(self) -> None:
        while self.tasks_list_box.count():
            item = self.tasks_list_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

        tasks: List[TaskEntry] = []
        if hasattr(self.ctx, "task_manager"):
            tasks = self.ctx.task_manager.list_tasks()[:3]

        if not tasks:
            empty = QFrame()
            empty.setObjectName("TaskItemCard")
            e_lay = QVBoxLayout(empty)
            e_lay.setContentsMargins(12, 12, 12, 12)
            e_lay.setSpacing(3)
            lbl1 = QLabel("Şu anda devam eden işlem yok.")
            lbl1.setStyleSheet(
                f"color: {C.COLOR_TEXT_SECONDARY}; font-size: 12px; font-weight: 600; background: transparent;"
            )
            lbl2 = QLabel("Ses ayrıştırma, dışa aktarma veya altyazı işlemleri başladığında burada görünür.")
            lbl2.setWordWrap(True)
            lbl2.setStyleSheet(
                f"color: {C.COLOR_TEXT_MUTED}; font-size: 10px; background: transparent;"
            )
            e_lay.addWidget(lbl1)
            e_lay.addWidget(lbl2)
            self.tasks_list_box.addWidget(empty)
            return

        cat_icon = {
            "stem": "waveform",
            "export": "video",
            "transcribe": "subtitle",
            "media": "folder",
        }
        for t in tasks:
            card = QFrame()
            card.setObjectName("TaskItemCard")
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            c_lay = QVBoxLayout(card)
            c_lay.setContentsMargins(12, 10, 12, 10)
            c_lay.setSpacing(6)

            row = QHBoxLayout()
            row.setSpacing(8)
            badge = QLabel()
            badge.setObjectName("IconBadge")
            badge.setFixedSize(28, 28)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setPixmap(
                icons.pixmap(cat_icon.get(t.category, "activity"), 15, C.COLOR_ACCENT_HOVER)
            )
            row.addWidget(badge)

            txt_col = QVBoxLayout()
            txt_col.setSpacing(1)
            t_lbl = QLabel(t.title)
            t_lbl.setStyleSheet(
                f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 12px; font-weight: 700; background: transparent;"
            )
            if t.status == "running":
                st_str = f"%{t.progress} tamamlandı · {t.message}"
            elif t.status == "completed":
                st_str = f"Tamamlandı ({t.elapsed_sec:.1f}s)"
            elif t.status == "failed":
                st_str = f"Hata: {t.error_detail or t.message}"
            else:
                st_str = "İptal edildi"
            s_lbl = QLabel(st_str)
            s_lbl.setStyleSheet(
                f"color: {C.COLOR_TEXT_MUTED}; font-size: 10px; background: transparent;"
            )
            txt_col.addWidget(t_lbl)
            txt_col.addWidget(s_lbl)
            row.addLayout(txt_col, 1)

            arr = QLabel()
            arr.setPixmap(icons.pixmap("chevron-right", 14, C.COLOR_TEXT_MUTED))
            arr.setStyleSheet("background: transparent;")
            row.addWidget(arr)
            c_lay.addLayout(row)

            p_bar = QProgressBar()
            p_bar.setRange(0, 100)
            p_bar.setValue(int(t.progress))
            p_bar.setTextVisible(False)
            p_bar.setFixedHeight(4)
            c_lay.addWidget(p_bar)

            self.tasks_list_box.addWidget(card)

    def refresh_system_status(self) -> None:
        metrics = SystemDiagnostics.live_hardware_metrics()

        st = str(metrics.get("status", "ok"))
        title = str(metrics.get("title", "Sistem hazır"))
        sub = str(metrics.get("subtitle", ""))
        self.health_title_lbl.setText(title)
        self.health_sub_lbl.setText(sub)
        if st == "ok":
            self.health_title_lbl.setStyleSheet(
                f"color: {C.COLOR_VU_GREEN}; font-size: 12px; font-weight: 700; background: transparent;"
            )
            self.health_icon_lbl.setPixmap(icons.pixmap("check", 16, C.COLOR_VU_GREEN))
        else:
            self.health_title_lbl.setStyleSheet(
                f"color: {C.COLOR_VU_YELLOW}; font-size: 12px; font-weight: 700; background: transparent;"
            )
            self.health_icon_lbl.setPixmap(icons.pixmap("warning", 16, C.COLOR_VU_YELLOW))

        gpu = metrics.get("gpu", {})
        if isinstance(gpu, dict):
            self.gpu_val_lbl.setText(str(gpu.get("detail", "—")))
            self.gpu_sub_lbl.setText(str(gpu.get("name", "—")))
            self.gpu_bar.setValue(int(gpu.get("percent", 0)))

        ram = metrics.get("ram", {})
        if isinstance(ram, dict):
            if ram.get("available"):
                self.ram_val_lbl.setText(f"%{ram.get('percent', 0)}")
                self.ram_sub_lbl.setText(
                    f"{ram.get('used_gb', 0.0):.1f} GB / {ram.get('total_gb', 0.0):.1f} GB"
                )
                self.ram_bar.setValue(int(ram.get("percent", 0)))
            else:
                self.ram_val_lbl.setText("—")
                self.ram_sub_lbl.setText("RAM bilgisi alınamadı")
                self.ram_bar.setValue(0)

        disk = metrics.get("disk", {})
        if isinstance(disk, dict):
            if disk.get("available"):
                self.disk_val_lbl.setText(f"%{disk.get('percent', 0)}")
                self.disk_sub_lbl.setText(str(disk.get("detail", "—")))
                self.disk_bar.setValue(int(disk.get("percent", 0)))
            else:
                self.disk_val_lbl.setText("—")
                self.disk_sub_lbl.setText("Depolama bilgisi alınamadı")
                self.disk_bar.setValue(0)

    # ================================================================ proje listesi ve görünüm
    def set_project_search_query(self, query: str) -> None:
        if self.project_search_edit.text() != query:
            self.project_search_edit.setText(query)

    def _set_view_mode(self, mode: str) -> None:
        self._view_mode = mode
        is_grid = mode == "grid"
        self.grid_mode_btn.setChecked(is_grid)
        self.list_mode_btn.setChecked(not is_grid)
        self.grid_mode_btn.setIcon(
            icons.icon("grid", C.COLOR_TEXT_PRIMARY if is_grid else C.COLOR_TEXT_SECONDARY, 15)
        )
        self.list_mode_btn.setIcon(
            icons.icon("list", C.COLOR_TEXT_PRIMARY if not is_grid else C.COLOR_TEXT_SECONDARY, 15)
        )
        self._render_projects_grid()

    def _render_projects_grid(self) -> None:
        while self.projects_grid.count():
            item = self.projects_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

        query = self.project_search_edit.text().strip().lower()
        sort_key = self.sort_combo.currentData() or "updated_desc"

        filtered = [
            p
            for p in self._cached_projects
            if not query
            or query in str(p.get("name", "")).lower()
            or query in str(p.get("project_type", "")).lower()
        ]

        if sort_key == "name_asc":
            filtered.sort(key=lambda x: str(x.get("name", "")).lower())
        elif sort_key == "created_desc":
            filtered.sort(key=lambda x: float(x.get("created_at") or 0.0), reverse=True)
        else:
            filtered.sort(key=lambda x: float(x.get("updated_at") or 0.0), reverse=True)

        self.proj_count_lbl.setText(str(len(filtered)))

        if not filtered:
            empty_frame = QFrame()
            empty_frame.setObjectName("EmptyPanel")
            empty_frame.setMinimumHeight(204)
            e_lay = QVBoxLayout(empty_frame)
            e_lay.setContentsMargins(28, 26, 28, 26)
            e_lay.setSpacing(10)
            e_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

            ico = QLabel()
            ico.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ico.setPixmap(icons.pixmap("folder", 28, C.COLOR_TEXT_MUTED))
            e_lay.addWidget(ico)

            msg_title = (
                "Aramayla eşleşen proje bulunamadı"
                if query
                else "Henüz kayıtlı proje bulunmuyor"
            )
            t_lbl = QLabel(msg_title)
            t_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t_lbl.setStyleSheet(
                f"color: {C.COLOR_TEXT_PRIMARY}; font-size: 14px; font-weight: 700;"
            )
            e_lay.addWidget(t_lbl)

            s_lbl = QLabel(
                "Yeni bir video düzenleme, ses ayrıştırma veya dublaj projesi oluşturarak başlayın."
            )
            s_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            s_lbl.setWordWrap(True)
            s_lbl.setMinimumHeight(22)
            s_lbl.setStyleSheet(f"color: {C.COLOR_TEXT_MUTED}; font-size: 12px;")
            e_lay.addWidget(s_lbl)

            if not query:
                create_btn = QPushButton("  + Yeni Proje Oluştur")
                create_btn.setProperty("variant", "primary")
                create_btn.setMinimumHeight(34)
                create_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                create_btn.clicked.connect(self._create_new_project)
                e_lay.addWidget(create_btn, 0, Qt.AlignmentFlag.AlignCenter)

            self.projects_grid.addWidget(empty_frame, 0, 0)
            return

        if self._view_mode == "list":
            for r_idx, proj_data in enumerate(filtered[:16]):
                card = self._build_project_card(proj_data, list_mode=True)
                self.projects_grid.addWidget(card, r_idx, 0)
        else:
            cols = max(1, self._grid_columns)
            for idx, proj_data in enumerate(filtered[:16]):
                r = idx // cols
                c = idx % cols
                card = self._build_project_card(proj_data, list_mode=False)
                self.projects_grid.addWidget(card, r, c)
            for c_idx in range(cols):
                self.projects_grid.setColumnStretch(c_idx, 1)

    def _build_project_card(self, proj_data: dict, list_mode: bool) -> ProjectCardWidget:
        card = ProjectCardWidget(proj_data, list_mode=list_mode, parent=self.projects_container)
        card.openRequested.connect(self._open_project_path)
        card.renameRequested.connect(self._on_rename_project)
        card.duplicateRequested.connect(self._on_duplicate_project)
        card.openFolderRequested.connect(self._on_open_project_folder)
        card.deleteRequested.connect(self._on_delete_project)
        return card

    # ================================================================ duyarlı yerleşim (Bölüm 8)
    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        w = self.width()
        # Dar ekranlarda sağ paneli gizleyerek orta alana tam nefes alanı bırak
        show_right = w >= 980
        if self.right_panel.isVisible() != show_right:
            self.right_panel.setVisible(show_right)

        avail_center = w - (C.RIGHT_PANEL_WIDTH if show_right else 0)
        if avail_center >= 1060:
            new_cols = 4
        elif avail_center >= 760:
            new_cols = 3
        elif avail_center >= 500:
            new_cols = 2
        else:
            new_cols = 1

        if new_cols != self._grid_columns:
            self._grid_columns = new_cols
            if self._view_mode == "grid":
                self._render_projects_grid()

    # ================================================================ eylemler
    def _create_new_project(self) -> None:
        dlg = NewProjectDialog(self.ctx, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if dlg.selected_mode == "open_existing" and dlg.existing_voxproj_path:
            self._open_project_path(dlg.existing_voxproj_path)
            return

        params = dlg.project_parameters()
        try:
            self.ctx.project_manager.create_project(
                name=params["name"],
                folder_path=params["folder_path"],
                sample_rate=params["sample_rate"],
                video_path=params["media_path"],
                project_type=params["project_type"],
                resolution=params["resolution"],
                fps=params["fps"],
            )
            self.refresh_recent()
            if self.ctx.project_manager.active_db:
                self.projectOpened.emit(str(self.ctx.project_manager.active_db.db_path))

            mode = params["mode"]
            if mode == "stem_work":
                self.stemsRequested.emit()
            elif mode == "dubbing_work":
                self.dubbingRequested.emit()
            else:
                self.editorRequested.emit()
        except Exception as exc:
            QMessageBox.critical(self, "Proje Oluşturulamadı", f"Hata:\n{exc}")

    def _choose_project(self) -> None:
        start_dir = str(C.PROJECTS_DIR)
        Path(start_dir).mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "VOX Studio Projesi Aç",
            start_dir,
            "VOX Studio Projesi (*.voxproj);;Tüm dosyalar (*)",
        )
        if path:
            self._open_project_path(path)

    def _open_project_path(self, path: str) -> None:
        proj = self.ctx.project_manager.open_project(path)
        if proj is None:
            QMessageBox.critical(
                self,
                "Proje Açılamadı",
                f"Proje dosyası okunamadı veya bozuk:\n{path}",
            )
            return
        self.refresh_recent()
        self.projectOpened.emit(path)
        self.editorRequested.emit()

    def _on_rename_project(self, db_path: str) -> None:
        current_name = Path(db_path).parent.name
        for p in self._cached_projects:
            if p.get("path") == db_path:
                current_name = str(p.get("name", current_name))
                break
        new_name, ok = QInputDialog.getText(
            self,
            "Projeyi Yeniden Adlandır",
            "Yeni proje adı:",
            text=current_name,
        )
        if ok and new_name.strip():
            if self.ctx.project_manager.rename_project(db_path, new_name.strip()):
                self.refresh_recent()
            else:
                QMessageBox.warning(self, "Hata", "Proje yeniden adlandırılamadı.")

    def _on_duplicate_project(self, db_path: str) -> None:
        new_db = self.ctx.project_manager.duplicate_project(db_path)
        if new_db is not None:
            self.refresh_recent()
        else:
            QMessageBox.warning(self, "Hata", "Proje kopyası oluşturulamadı.")

    def _on_open_project_folder(self, db_path: str) -> None:
        folder = Path(db_path).parent
        if not folder.exists():
            QMessageBox.warning(self, "Klasör Bulunamadı", f"Klasör mevcut değil:\n{folder}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            try:
                os.startfile(str(folder))  # type: ignore[attr-defined]
            except Exception as exc:
                QMessageBox.warning(self, "Klasör Açılamadı", f"{folder}\n\n{exc}")

    def _on_delete_project(self, db_path: str) -> None:
        proj_name = Path(db_path).parent.name
        for p in self._cached_projects:
            if p.get("path") == db_path:
                proj_name = str(p.get("name", proj_name))
                break

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Projeyi Sil")
        box.setText(f"'{proj_name}' projesini silmek istediğinize emin misiniz?")
        box.setInformativeText(
            f"Proje dosyası ({db_path}) kalıcı olarak silinecek. Dışa aktardığınız medya dosyaları silinmez."
        )
        del_btn = box.addButton("Projeyi Sil", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton("Vazgeç", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        if box.clickedButton() is del_btn:
            self.ctx.project_manager.delete_project(db_path)
            self.refresh_recent()

    def _choose_file(self) -> None:
        start_dir = str(self.ctx.recordings_dir())
        a_exts = " ".join(f"*{e}" for e in C.AUDIO_EXTENSIONS + C.VIDEO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Medya dosyası aç",
            start_dir,
            f"Ses ve Video Dosyaları ({a_exts});;Tüm dosyalar (*)",
        )
        if path:
            self.openFileRequested.emit(path)

    def refresh_recent(self) -> None:
        # 1. Geriye dönük uyumluluk için son kayıtlar listesini güncelle
        self.recent_list.clear()
        rec_paths = [
            p for p in (self.ctx.config.get("recent_recordings", []) or []) if Path(p).exists()
        ]
        for raw in rec_paths[:14]:
            path = Path(raw)
            try:
                size = format_file_size(path.stat().st_size)
            except OSError:
                size = "—"
            item = QListWidgetItem(f"{path.name}\n{size}  ·  {path.parent}")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.recent_list.addItem(item)

        # 2. Gerçek projeleri yükle ve kartları oluştur
        self._cached_projects = self.ctx.project_manager.list_recent_projects()
        self.projects_list.clear()
        for rp in self._cached_projects:
            p_item = QListWidgetItem(rp.get("name", ""))
            p_item.setData(Qt.ItemDataRole.UserRole, rp.get("path", ""))
            self.projects_list.addItem(p_item)

        self._render_projects_grid()
        self.refresh_tasks_panel()
        self.refresh_system_status()

        active = self.ctx.project_manager.active_project
        if active:
            self.active_badge.setText(f"● AKTİF PROJE: {active.name.upper()}")
        else:
            self.active_badge.setText("YEREL STÜDYO · HAZIR")
