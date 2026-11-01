"""
VOX Studio — Professional Voice Acting & Dubbing Workstation
Core Constants and Configuration Definitions

Bu modül yalnızca sabit değerler içerir; import anında dosya sistemi
üzerinde yan etki oluşturmaz. Kullanıcı verisi yolları için `core.paths`.
"""

from . import paths as _paths

APP_NAME = "VOX Studio"
APP_SUBTITLE = "Audio Engine V1 — AI Speech Restoration Workstation"
APP_VERSION = "1.1.0"
APP_AUTHOR = "VOX Audio Engineering"

# Geriye dönük uyumluluk için başlangıç anındaki yol anlık görüntüsü.
# Yeni kod `core.paths` fonksiyonlarını kullanmalıdır.
APP_DATA_DIR = _paths.app_home()
PROJECTS_DIR = _paths.projects_dir()
MODELS_DIR = _paths.models_dir()
CACHE_DIR = _paths.cache_dir()
LOGS_DIR = _paths.logs_dir()

# ---------------------------------------------------------------------------
# Ses yapılandırması
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 48000
SUPPORTED_SAMPLE_RATES = [44100, 48000, 88200, 96000]
DEFAULT_CHANNELS = 1  # Seslendirme kaydı varsayılanı mono
DEFAULT_BLOCK_SIZE = 1024
DEFAULT_BIT_DEPTH = 24  # 16, 24, 32 (float)

# Ekranda gösterilen ad -> libsndfile subtype
RECORDING_SUBTYPES = {
    "24-bit PCM": "PCM_24",
    "16-bit PCM": "PCM_16",
    "32-bit Float": "FLOAT",
}
DEFAULT_RECORDING_SUBTYPE = "PCM_24"
SUBTYPE_LABELS = {v: k for k, v in RECORDING_SUBTYPES.items()}

AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif", ".wma"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
SUBTITLE_EXTENSIONS = [".srt", ".vtt"]
ALL_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS + AUDIO_EXTENSIONS + IMAGE_EXTENSIONS + SUBTITLE_EXTENSIONS

# Seviye / analiz eşikleri
CLIP_THRESHOLD = 0.9995        # -0.004 dBFS üzeri "clipped" sayılır
SILENCE_DBFS = -50.0           # bu eşiğin altı sessizlik
LOW_LEVEL_DBFS = -45.0         # giriş kontrolünde "çok düşük" uyarısı
HOT_LEVEL_DBFS = -3.0          # giriş kontrolünde "çok yüksek" uyarısı

# ---------------------------------------------------------------------------
# Tasarım sistemi (siyah-kırmızı profesyonel stüdyo teması)
# ---------------------------------------------------------------------------
COLOR_BG_DARKEST = "#090A0C"      # Ana arka plan
COLOR_BG_MAIN = "#090A0C"         # Ana pencere arka planı
COLOR_BG_PANEL = "#101114"        # İkincil arka plan / panel
COLOR_BG_DARK = COLOR_BG_PANEL      # Geriye dönük uyumluluk takma adı (#101114)
COLOR_BG_SECONDARY = COLOR_BG_PANEL # İkincil arka plan takma adı (#101114)
COLOR_BG_SURFACE = "#151619"      # Kart yüzeyleri / yükseltilmiş yüzey
COLOR_BG_CARD = COLOR_BG_SURFACE    # Kart yüzeyi takma adı (#151619)
COLOR_BG_ELEVATED = COLOR_BG_SURFACE# Yükseltilmiş kart takma adı (#151619)
COLOR_BG_HOVER = "#1C1D21"        # Hover yüzeyleri
COLOR_BORDER = "#28292E"          # Kenarlıklar
COLOR_BORDER_LIGHT = "#363840"    # Seçili / aktif hafif çerçeve

COLOR_ACCENT = "#E53945"          # Ana vurgu kırmızısı
COLOR_ACCENT_PRIMARY = COLOR_ACCENT
COLOR_ACCENT_HOVER = "#FF4652"    # Hover ve aktif kırmızı
COLOR_ACCENT_MUTED = "#241114"    # Koyu kırmızımsı aktif yüzey
COLOR_ACCENT_GLOW = "#FF4652"     # Kayıt ve aktif göstergesi

COLOR_TEXT_PRIMARY = "#F5F5F6"    # Birincil metin
COLOR_TEXT_SECONDARY = "#A1A1AA"  # İkincil metin
COLOR_TEXT_MUTED = "#71717A"      # Pasif metin

COLOR_VU_GREEN = "#2ECC71"
COLOR_VU_YELLOW = "#F1C40F"
COLOR_VU_RED = "#E53945"

# ---------------------------------------------------------------------------
# Arayüz ölçüleri (yüksek DPI'da Qt ölçekler)
# ---------------------------------------------------------------------------
SIDEBAR_WIDTH = 216
SIDEBAR_COLLAPSED_WIDTH = 64
TOPBAR_HEIGHT = 54
TRANSPORT_HEIGHT = 72
RIGHT_PANEL_WIDTH = 296
MIN_WINDOW_SIZE = (1160, 760)
DEFAULT_WINDOW_SIZE = (1440, 900)

# Waveform
WAVEFORM_PEAK_BUCKET = 256        # kayıt sırasında özetleme penceresi (frame)
WAVEFORM_MIN_POINTS = 200         # çizim için alt sınır
WAVEFORM_COLOR = "#E53945"
WAVEFORM_COLOR_PLAYED = "#FF4652"
WAVEFORM_COLOR_MUTED = "#28292E"

# ---------------------------------------------------------------------------
# Replik / take durumları
# ---------------------------------------------------------------------------
LINE_STATUS_PENDING = "pending"
LINE_STATUS_RECORDED = "recorded"
LINE_STATUS_EDITING = "editing"
LINE_STATUS_APPROVED = "approved"
LINE_STATUS_RETAKE = "retake"

LINE_STATUS_LABELS_TR = {
    LINE_STATUS_PENDING: "Bekliyor",
    LINE_STATUS_RECORDED: "Kayıt alındı",
    LINE_STATUS_EDITING: "Düzenleniyor",
    LINE_STATUS_APPROVED: "Onaylandı",
    LINE_STATUS_RETAKE: "Yeniden kayıt gerekli",
}

# ---------------------------------------------------------------------------
# Varsayılan dosya adı şablonları
# ---------------------------------------------------------------------------
RECORDING_FILENAME_TEMPLATE = "{timestamp}_{label}"
LINE_FILENAME_TEMPLATE = "{character}_Scene{scene:02d}_Line{line:03d}"
