"""
VOX Studio — arayüz teması.

Renkler `core.constants` içindeki tasarım sisteminden gelir; tek kaynak
oradan beslenir. QSS dinamik olarak üretilir (tema ileride çoğaltılabilir).
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from core import constants as C

FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"


def _replace(stylesheet: str) -> str:
    return stylesheet


def build_stylesheet() -> str:
    return f"""
/* ---------------------------------------------------------------- temel */
QWidget {{
    color: {C.COLOR_TEXT_PRIMARY};
    font-family: "{FONT_FAMILY}";
    font-size: 13px;
}}
QMainWindow, #AppRoot {{ background: {C.COLOR_BG_MAIN}; }}
QDialog {{ background: {C.COLOR_BG_MAIN}; }}
QToolTip {{
    background: {C.COLOR_BG_SURFACE};
    color: {C.COLOR_TEXT_PRIMARY};
    border: 1px solid {C.COLOR_BORDER_LIGHT};
    padding: 6px 8px;
    border-radius: 4px;
}}

/* --------------------------------------------------------------- yerleşim */
#Sidebar {{
    background: {C.COLOR_BG_DARKEST};
    border-right: 1px solid {C.COLOR_BORDER};
}}
#TopBar {{
    background: {C.COLOR_BG_DARKEST};
    border-bottom: 1px solid {C.COLOR_BORDER};
}}
#TransportBar {{
    background: {C.COLOR_BG_PANEL};
    border-top: 1px solid {C.COLOR_BORDER};
}}
#Card {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 12px;
}}
#CardFlat {{
    background: {C.COLOR_BG_PANEL};
    border: 1px solid {C.COLOR_BORDER};
}}
#HeroBanner {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0.6,
        stop:0 #15161A, stop:0.55 #1B1115, stop:0.85 #2A1218, stop:1 #131114);
    border: 1px solid #331D22;
    border-radius: 14px;
}}
#ActionCard {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 12px;
}}
#ActionCard:hover {{
    background: {C.COLOR_BG_HOVER};
    border: 1px solid {C.COLOR_ACCENT};
}}
#ProjectCard {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 12px;
}}
#ProjectCard:hover {{
    background: {C.COLOR_BG_HOVER};
    border: 1px solid {C.COLOR_ACCENT};
}}
#QuickStripCard {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 10px;
}}
#QuickStripCard:hover {{
    background: {C.COLOR_BG_HOVER};
    border: 1px solid {C.COLOR_ACCENT};
}}
#RightStatusCard {{
    background: {C.COLOR_BG_PANEL};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 12px;
}}
#TaskItemCard {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 9px;
}}
#TaskItemCard:hover {{
    background: {C.COLOR_BG_HOVER};
    border-color: {C.COLOR_BORDER_LIGHT};
}}
#IconBadge {{
    background: {C.COLOR_ACCENT_MUTED};
    border: 1px solid #3D1A1F;
    border-radius: 9px;
}}
#TopSearchEdit {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 8px;
    padding: 5px 12px;
    color: {C.COLOR_TEXT_PRIMARY};
    font-size: 12px;
}}
#TopSearchEdit:focus {{
    border: 1px solid {C.COLOR_ACCENT};
    background: {C.COLOR_BG_HOVER};
}}
#RightPanel {{
    background: {C.COLOR_BG_DARKEST};
    border-left: 1px solid {C.COLOR_BORDER};
}}
#Divider {{ background: {C.COLOR_BORDER}; max-height: 1px; min-height: 1px; }}
#VDivider {{ background: {C.COLOR_BORDER}; max-width: 1px; min-width: 1px; }}

/* ---------------------------------------------------------------- tipografi */
#BrandTitle {{ font-size: 15px; font-weight: 700; letter-spacing: 1.6px; }}
#BrandSub {{ font-size: 9px; color: {C.COLOR_TEXT_MUTED}; letter-spacing: 0.6px; }}
#ViewTitle {{ font-size: 22px; font-weight: 700; }}
#ViewSubtitle {{ font-size: 12px; color: {C.COLOR_TEXT_SECONDARY}; }}
#SectionTitle {{
    font-size: 11px;
    font-weight: 700;
    color: {C.COLOR_TEXT_SECONDARY};
    letter-spacing: 1.2px;
}}
#Muted {{ color: {C.COLOR_TEXT_SECONDARY}; }}
#MutedSmall {{ color: {C.COLOR_TEXT_MUTED}; font-size: 11px; }}
#Mono, #TimeDisplay {{ font-family: "{FONT_MONO}"; }}
#TimeDisplay {{ font-size: 22px; font-weight: 600; letter-spacing: 0.5px; }}
#TimeSub {{ font-family: "{FONT_MONO}"; font-size: 11px; color: {C.COLOR_TEXT_MUTED}; }}
#MetricValue {{ font-family: "{FONT_MONO}"; font-size: 16px; font-weight: 600; }}
#MetricLabel {{ font-size: 10px; color: {C.COLOR_TEXT_MUTED}; letter-spacing: 0.8px; }}
#BigLine {{ font-size: 18px; }}
#EmptyTitle {{ font-size: 16px; font-weight: 600; }}
#StatusOk {{ color: {C.COLOR_VU_GREEN}; }}
#StatusWarn {{ color: {C.COLOR_VU_YELLOW}; }}
#StatusErr {{ color: {C.COLOR_VU_RED}; }}

/* ------------------------------------------------------------------- butonlar */
QPushButton {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER_LIGHT};
    border-radius: 6px;
    padding: 6px 11px;
    color: {C.COLOR_TEXT_PRIMARY};
}}
QPushButton:hover {{ background: {C.COLOR_BG_HOVER}; border-color: {C.COLOR_ACCENT_HOVER}; }}
QPushButton:pressed {{ background: {C.COLOR_BG_PANEL}; }}
QPushButton:focus {{ border-color: {C.COLOR_ACCENT_HOVER}; }}
QPushButton:disabled {{
    color: {C.COLOR_TEXT_MUTED};
    border-color: {C.COLOR_BORDER};
    background: {C.COLOR_BG_PANEL};
}}
QPushButton[compact="true"] {{
    padding: 2px 6px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 600;
}}
QPushButton[pill="true"] {{
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    background: {C.COLOR_BG_PANEL};
    border: 1px solid {C.COLOR_BORDER};
    color: {C.COLOR_TEXT_SECONDARY};
}}
QPushButton[pill="true"]:hover {{
    background: {C.COLOR_BG_HOVER};
    color: {C.COLOR_TEXT_PRIMARY};
}}
QPushButton[pill="true"]:checked {{
    background: {C.COLOR_ACCENT};
    border-color: {C.COLOR_ACCENT_HOVER};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton[variant="primary"] {{
    background: {C.COLOR_ACCENT};
    border-color: {C.COLOR_ACCENT};
    font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{ background: {C.COLOR_ACCENT_HOVER}; border-color: {C.COLOR_ACCENT_HOVER}; }}
QPushButton[variant="primary"]:disabled {{ background: {C.COLOR_ACCENT_MUTED}; border-color: {C.COLOR_ACCENT_MUTED}; }}
QPushButton[variant="danger"] {{
    background: transparent;
    border-color: {C.COLOR_ACCENT};
    color: {C.COLOR_ACCENT_GLOW};
}}
QPushButton[variant="danger"]:hover {{ background: {C.COLOR_ACCENT_MUTED}; }}
QPushButton[variant="ghost"] {{
    background: transparent;
    border-color: transparent;
    color: {C.COLOR_TEXT_SECONDARY};
}}
QPushButton[variant="ghost"]:hover {{ background: {C.COLOR_BG_SURFACE}; color: {C.COLOR_TEXT_PRIMARY}; }}
QPushButton[variant="link"] {{
    background: transparent; border: none; color: {C.COLOR_TEXT_SECONDARY}; padding: 2px 4px;
}}
QPushButton[variant="link"]:hover {{ color: {C.COLOR_ACCENT_HOVER}; }}
QPushButton[recording="true"] {{
    background: {C.COLOR_ACCENT};
    border-color: {C.COLOR_ACCENT_GLOW};
    font-weight: 700;
}}

/* --------------------------------------------------------------------- grup kutuları */
QGroupBox {{
    background: {C.COLOR_BG_PANEL};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    padding-bottom: 8px;
    padding-left: 10px;
    padding-right: 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: {C.COLOR_TEXT_SECONDARY};
    font-size: 11px;
    font-weight: 700;
}}

/* --------------------------------------------------------------------- girdi */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {C.COLOR_ACCENT};
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QTextEdit:hover {{
    border-color: {C.COLOR_BORDER_LIGHT};
}}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {C.COLOR_ACCENT_HOVER};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER_LIGHT};
    selection-background-color: {C.COLOR_ACCENT_MUTED};
    outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 16px; background: {C.COLOR_BG_HOVER}; border: none;
}}

/* -------------------------------------------------------------------- listeler */
QListWidget, QTreeWidget, QTableWidget {{
    background: {C.COLOR_BG_PANEL};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 8px;
    outline: none;
}}
QListWidget::item {{ padding: 8px 10px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {C.COLOR_ACCENT_MUTED}; }}
QListWidget::item:hover {{ background: {C.COLOR_BG_SURFACE}; }}
QHeaderView::section {{
    background: {C.COLOR_BG_SURFACE};
    color: {C.COLOR_TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {C.COLOR_BORDER};
    padding: 7px 8px;
    font-size: 11px;
    font-weight: 600;
}}
QTableWidget {{ gridline-color: {C.COLOR_BORDER}; }}
QTableWidget::item:selected {{ background: {C.COLOR_ACCENT_MUTED}; }}

/* ---------------------------------------------------------------- kaydırma */
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {C.COLOR_BORDER_LIGHT}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {C.COLOR_TEXT_MUTED}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none; border: none; height: 0; width: 0;
}}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {C.COLOR_BORDER_LIGHT}; border-radius: 5px; min-width: 30px; }}

/* -------------------------------------------------------------------- sekmeler */
QTabWidget::pane {{ border: 1px solid {C.COLOR_BORDER}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {C.COLOR_TEXT_SECONDARY};
    padding: 8px 14px; border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {C.COLOR_TEXT_PRIMARY}; border-bottom: 2px solid {C.COLOR_ACCENT}; }}
QTabBar::tab:hover {{ color: {C.COLOR_TEXT_PRIMARY}; }}

/* -------------------------------------------------------------------- menü */
QMenu {{ background: {C.COLOR_BG_SURFACE}; border: 1px solid {C.COLOR_BORDER_LIGHT}; padding: 6px; }}
QMenu::item {{ padding: 6px 22px 6px 14px; border-radius: 4px; }}
QMenu::item:selected {{ background: {C.COLOR_ACCENT_MUTED}; }}
QMenu::separator {{ height: 1px; background: {C.COLOR_BORDER}; margin: 4px 8px; }}

/* ------------------------------------------------------------------ ilerleme */
QProgressBar {{
    background: {C.COLOR_BG_HOVER}; border: none;
    border-radius: 3px; height: 6px; max-height: 6px; text-align: center; font-size: 9px;
}}
QProgressBar::chunk {{ background: {C.COLOR_ACCENT}; border-radius: 3px; }}

/* ------------------------------------------------------------------- sidebar */
QPushButton[nav="true"] {{
    background: transparent;
    border: 1px solid transparent;
    border-left: 3px solid transparent;
    border-radius: 8px;
    padding: 9px 12px;
    text-align: left;
    color: {C.COLOR_TEXT_SECONDARY};
    font-size: 13px;
}}
QPushButton[nav="true"]:hover {{
    background: {C.COLOR_BG_SURFACE};
    color: {C.COLOR_TEXT_PRIMARY};
}}
QPushButton[nav="true"]:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2B1317, stop:1 #151215);
    border: 1px solid #451B21;
    border-left: 3px solid {C.COLOR_ACCENT};
    color: {C.COLOR_TEXT_PRIMARY};
    font-weight: 600;
}}

QSplitter::handle {{ background: {C.COLOR_BORDER}; width: 1px; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid {C.COLOR_BORDER_LIGHT}; background: {C.COLOR_BG_SURFACE};
}}
QCheckBox::indicator:checked {{ background: {C.COLOR_ACCENT}; border-color: {C.COLOR_ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {C.COLOR_BG_SURFACE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {C.COLOR_ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 12px; height: 12px; margin: -4px 0; border-radius: 6px;
    background: {C.COLOR_TEXT_PRIMARY};
}}

/* --------------------------------------------------------------- ölçü kutuları */
#MetricBox {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 7px;
}}
#MetricValueSmall {{
    font-family: "{FONT_MONO}";
    font-size: 13px;
    font-weight: 600;
    color: {C.COLOR_TEXT_PRIMARY};
}}
#ScoreValue {{
    font-family: "{FONT_MONO}";
    font-size: 34px;
    font-weight: 700;
}}
#TopValue {{ font-size: 12px; color: {C.COLOR_TEXT_PRIMARY}; }}
#StatusText {{ font-size: 12px; color: {C.COLOR_TEXT_SECONDARY}; }}
#WarningsBox {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER};
    border-radius: 6px;
    font-size: 11px;
    color: {C.COLOR_TEXT_SECONDARY};
}}
#RecentList {{ background: transparent; border: none; }}
#RecentList::item {{ padding: 6px 8px; }}

/* ------------------------------------------------------------- durum rozetleri */
#RecordBadge {{
    font-family: "{FONT_MONO}";
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    padding: 4px 10px;
    border-radius: 4px;
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER};
    color: {C.COLOR_TEXT_SECONDARY};
}}
#RecordBadge[state="recording"] {{
    background: {C.COLOR_ACCENT};
    border-color: {C.COLOR_ACCENT_GLOW};
    color: #FFFFFF;
}}
#RecordBadge[state="paused"] {{
    border-color: {C.COLOR_VU_YELLOW};
    color: {C.COLOR_VU_YELLOW};
}}
#RecordBadge[state="busy"] {{
    border-color: {C.COLOR_BORDER_LIGHT};
    color: {C.COLOR_TEXT_SECONDARY};
}}
#RecordBadge[state="error"] {{
    background: {C.COLOR_ACCENT_MUTED};
    border-color: {C.COLOR_VU_RED};
    color: {C.COLOR_VU_RED};
}}
#DevBadge {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 5px 10px;
    border-radius: 4px;
    background: {C.COLOR_ACCENT_MUTED};
    border: 1px solid {C.COLOR_ACCENT};
    color: {C.COLOR_ACCENT_GLOW};
}}

/* ------------------------------------------------------------------- paneller */
#EmptyPanel {{
    background: {C.COLOR_BG_PANEL};
    border: 1px dashed {C.COLOR_BORDER_LIGHT};
    border-radius: 10px;
}}
#NoticePanel {{
    background: {C.COLOR_BG_PANEL};
    border: 1px solid {C.COLOR_BORDER};
    border-left: 3px solid {C.COLOR_VU_YELLOW};
    border-radius: 8px;
}}

/* -------------------------------------------------------------------- taşıma */
#RecordButton {{
    background: {C.COLOR_ACCENT};
    border: 1px solid {C.COLOR_ACCENT};
    border-radius: 6px;
    font-weight: 700;
    font-size: 13px;
    color: #FFFFFF;
}}
#RecordButton:hover {{ background: {C.COLOR_ACCENT_HOVER}; border-color: {C.COLOR_ACCENT_HOVER}; }}
#RecordButton:pressed {{ background: {C.COLOR_ACCENT_MUTED}; }}
#RecordButton:disabled {{
    background: {C.COLOR_ACCENT_MUTED};
    border-color: {C.COLOR_ACCENT_MUTED};
    color: {C.COLOR_TEXT_MUTED};
}}
#RecordButton[recording="true"] {{
    background: {C.COLOR_ACCENT_GLOW};
    border-color: #FF8080;
    color: #1A0A0A;
}}
#RecordButton[recording="true"]:hover {{ background: #FF6B6B; }}
#TransportButton {{
    background: {C.COLOR_BG_SURFACE};
    border: 1px solid {C.COLOR_BORDER_LIGHT};
    border-radius: 6px;
    padding: 0;
}}
#TransportButton:hover {{ background: {C.COLOR_BG_HOVER}; }}
#TransportButton:disabled {{
    background: {C.COLOR_BG_PANEL};
    border-color: {C.COLOR_BORDER};
}}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    font = QFont(FONT_FAMILY, 9)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(C.COLOR_BG_MAIN))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(C.COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(C.COLOR_BG_PANEL))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(C.COLOR_BG_SURFACE))
    palette.setColor(QPalette.ColorRole.Text, QColor(C.COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(C.COLOR_BG_SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(C.COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(C.COLOR_ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(C.COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(C.COLOR_BG_SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(C.COLOR_TEXT_PRIMARY))
    app.setPalette(palette)
    app.setStyleSheet(build_stylesheet())
