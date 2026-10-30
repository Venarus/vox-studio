"""
VOX Studio — Merkezi Çok Dilli Yerelleştirme (i18n) ve 30 Dil Kataloğu

Özellikler:
- 30 dil için yerel ad (Native Name), Türkçe ad, İngilizce ad, ISO kodu, Whisper dil kodu,
  sağdan sola (RTL) yön bilgisi, yazı sistemi (script) ve Unicode doğrulama örnek metinleri.
- Arayüz Dili (ui_language) ve Transkripsiyon Dili (transcription_language) birbirinden bağımsızdır.
- Canlı dil değiştirme, eksik çeviri anahtarı denetimi (find_missing_keys), güvenli fallback
  (Seçili Dil -> Türkçe -> İngilizce) ve NFC Unicode normalizasyonu.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from core.logger import get_logger

logger = get_logger("core.i18n")


@dataclass(frozen=True)
class LanguageSpec:
    code: str               # Uygulama içi standart kod (örn. 'tr', 'pt-BR', 'zh-CN')
    whisper_code: str       # Whisper / faster-whisper 2 harfli dil kodu (örn. 'tr', 'pt', 'zh')
    turkish_name: str       # Türkçe açıklama adı
    native_name: str        # Dilin kendi yerel adı
    english_name: str       # İngilizce adı
    is_rtl: bool = False    # Sağdan sola (Arapça, İbranice, Farsça, Urduca)
    script: str = "latin"   # latin | cyrillic | greek | arabic | hebrew | devanagari | bengali | cjk
    sample_text: str = ""   # Unicode ve altyazı testi için örnek cümle

    @property
    def display_label(self) -> str:
        """Seçici menülerde gösterilecek standart etiket: 'Yerel Ad — Türkçe Ad (kod)'."""
        if self.native_name == self.turkish_name:
            return f"{self.native_name} ({self.code})"
        return f"{self.native_name} — {self.turkish_name} ({self.code})"

    def __getitem__(self, idx):
        if idx == 0:
            return self.code
        elif idx == 1:
            return self.display_label
        raise IndexError("LanguageSpec index out of range")


class SupportedLanguagesCatalog(dict):
    """
    Hem sözlük (dict.keys(), dict.values(), catalog['tr'])
    hem de sıralı liste/tuple iterasyonu (for spec in catalog)
    ve indeks erişimi (catalog[0]) sağlayan 30 dil koleksiyonu.
    """
    def __init__(self, specs: Tuple[LanguageSpec, ...]):
        self._specs = list(specs)
        super().__init__({s.code: s for s in specs})

    def __iter__(self):
        return iter(self._specs)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._specs[item]
        return super().__getitem__(item)

    def values(self):
        return self._specs


# Ek Gereksinim 1.1: Desteklenmesi Zorunlu 30 Dil Kataloğu
_SPECS_LIST: Tuple[LanguageSpec, ...] = (
    LanguageSpec('tr', 'tr', 'Türkçe', 'Türkçe', 'Turkish', False, 'latin', 'Merhaba, ses kaydı ve dublaj stüdyosu hazır: ğüşıöçĞÜŞİÖÇ.'),
    LanguageSpec('en', 'en', 'İngilizce', 'English', 'English', False, 'latin', 'Hello, the voice acting and dubbing studio is ready.'),
    LanguageSpec('es', 'es', 'İspanyolca', 'Español', 'Spanish', False, 'latin', '¡Hola! El estudio de doblaje y grabación está listo: ñáéíóúü.'),
    LanguageSpec('de', 'de', 'Almanca', 'Deutsch', 'German', False, 'latin', 'Hallo, das Synchron- und Tonstudio ist bereit: äöüßÄÖÜ.'),
    LanguageSpec('fr', 'fr', 'Fransızca', 'Français', 'French', False, 'latin', "Bonjour, le studio de doublage et d'enregistrement est prêt : éèêàçœ."),
    LanguageSpec('it', 'it', 'İtalyanca', 'Italiano', 'Italian', False, 'latin', 'Ciao, lo studio di doppiaggio e registrazione è pronto: àèéìòù.'),
    LanguageSpec('pt', 'pt', 'Portekizce', 'Português', 'Portuguese', False, 'latin', 'Olá, o estúdio de dobragem e gravação está pronto: ãõçáéíóú.'),
    LanguageSpec('ru', 'ru', 'Rusça', 'Русский', 'Russian', False, 'cyrillic', 'Здравствуйте, студия дубляжа и звукозаписи готова к работе.'),
    LanguageSpec('ar', 'ar', 'Arapça', 'العربية', 'Arabic', True, 'arabic', 'مرحباً، استوديو الدبلجة والتسجيل الصوتي جاهز للعمل.'),
    LanguageSpec('ja', 'ja', 'Japonca', '日本語', 'Japanese', False, 'cjk', 'こんにちは、アフレコ・音声録音スタジオの準備ができました。'),
    LanguageSpec('ko', 'ko', 'Korece', '한국어', 'Korean', False, 'cjk', '안녕하세요, 더빙 및 음성 녹음 스튜디오가 준비되었습니다.'),
    LanguageSpec('zh-CN', 'zh', 'Basitleştirilmiş Çince', '简体中文', 'Simplified Chinese', False, 'cjk', '您好，配音与录音工作室已准备就绪。'),
    LanguageSpec('zh-TW', 'zh', 'Geleneksel Çince', '繁體中文', 'Traditional Chinese', False, 'cjk', '您好，配音與錄音工作室已準備就緒。'),
    LanguageSpec('hi', 'hi', 'Hintçe', 'हिन्दी', 'Hindi', False, 'devanagari', 'नमस्ते, डबिंग और वॉइस रिकॉर्डिंग स्टूडियो तैयार है।'),
    LanguageSpec('nl', 'nl', 'Felemenkçe', 'Nederlands', 'Dutch', False, 'latin', 'Hallo, de nasynchronisatie- en opnamestudio is klaar.'),
    LanguageSpec('pl', 'pl', 'Lehçe', 'Polski', 'Polish', False, 'latin', 'Cześć, studio dubbingowe i nagraniowe jest gotowe: ąćęłńóśźż.'),
    LanguageSpec('sv', 'sv', 'İsveççe', 'Svenska', 'Swedish', False, 'latin', 'Hej, dubbnings- och inspelningsstudion är redo: åäöÅÄÖ.'),
    LanguageSpec('no', 'no', 'Norveççe', 'Norsk', 'Norwegian', False, 'latin', 'Hei, dubbe- og innspillingsstudioet er klart: æøåÆØÅ.'),
    LanguageSpec('da', 'da', 'Danca', 'Dansk', 'Danish', False, 'latin', 'Hej, dubbing- og lydstudiet er klar: æøåÆØÅ.'),
    LanguageSpec('fi', 'fi', 'Fince', 'Suomi', 'Finnish', False, 'latin', 'Hei, jälkiäänitys- ja äänitysstudio on valmis: äöÄÖ.'),
    LanguageSpec('el', 'el', 'Yunanca', 'Ελληνικά', 'Greek', False, 'greek', 'Γεια σας, το στούντιο μεταγλώττισης και ηχογράφησης είναι έτοιμο.'),
    LanguageSpec('cs', 'cs', 'Çekçe', 'Čeština', 'Czech', False, 'latin', 'Dobrý den, dabingové a nahrávací studio je připraveno: čďěňřšťůž.'),
    LanguageSpec('ro', 'ro', 'Rumence', 'Română', 'Romanian', False, 'latin', 'Salut, studioul de dublaj și înregistrări este pregătit: ăâîșț.'),
    LanguageSpec('hu', 'hu', 'Macarca', 'Magyar', 'Hungarian', False, 'latin', 'Üdvözöljük, a szinkron- és hangstúdió készen áll: áéíóöőúüű.'),
    LanguageSpec('uk', 'uk', 'Ukraynaca', 'Українська', 'Ukrainian', False, 'cyrillic', 'Вітаємо, студія дубляжу та звукозапису готова до роботи: іїєґ.'),
    LanguageSpec('id', 'id', 'Endonezce', 'Bahasa Indonesia', 'Indonesian', False, 'latin', 'Halo, studio sulih suara dan perekaman audio sudah siap.'),
    LanguageSpec('vi', 'vi', 'Vietnamca', 'Tiếng Việt', 'Vietnamese', False, 'latin', 'Xin chào, phòng thu lồng tiếng và ghi âm đã sẵn sàng: ăâđêôơư.'),
    LanguageSpec('th', 'th', 'Tayca', 'ไทย', 'Thai', False, 'thai', 'สวัสดี สตูดิโอพากย์เสียงและบันทึกเสียงพร้อมแล้ว'),
    LanguageSpec('he', 'he', 'İbranice', 'עברית', 'Hebrew', True, 'hebrew', 'שלום, אולפן הדיבוב וההקלטות מוכן לעבודה.'),
    LanguageSpec('fa', 'fa', 'Farsça', 'فارسی', 'Persian', True, 'arabic', 'سلام، استودیوی دوبله و ضبط صدا آماده کار است.'),
)

SUPPORTED_LANGUAGES: SupportedLanguagesCatalog = SupportedLanguagesCatalog(_SPECS_LIST)

LANGUAGE_BY_CODE: Dict[str, LanguageSpec] = {spec.code: spec for spec in SUPPORTED_LANGUAGES}
RTL_LANGUAGE_CODES = {spec.code for spec in SUPPORTED_LANGUAGES if spec.is_rtl}

# Yazı sistemine göre Windows uyumlu font fallback zinciri
SCRIPT_FONT_FAMILIES: Dict[str, List[str]] = {
    "latin": ["Segoe UI", "Inter", "Arial", "Tahoma"],
    "cyrillic": ["Segoe UI", "Arial", "Tahoma"],
    "greek": ["Segoe UI", "Arial", "Tahoma"],
    "arabic": ["Segoe UI", "Tahoma", "Traditional Arabic", "Arial"],
    "hebrew": ["Segoe UI", "Tahoma", "David", "Arial"],
    "devanagari": ["Nirmala UI", "Mangal", "Segoe UI", "Arial"],
    "bengali": ["Nirmala UI", "Vrinda", "Segoe UI", "Arial"],
    "cjk": ["Microsoft YaHei UI", "Yu Gothic UI", "Meiryo UI", "Malgun Gothic", "Segoe UI"],
}


def normalize_unicode_text(text: str) -> str:
    """Boşlukları ve birleşik Unicode karakterleri (NFC) standartlaştırır."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", str(text))


def get_language_spec(code: Optional[str]) -> LanguageSpec:
    """Dil koduna göre LanguageSpec döndürür; bulunamazsa Türkçe ('tr') döner."""
    if not code:
        return LANGUAGE_BY_CODE["tr"]
    norm = code.strip()
    if norm in LANGUAGE_BY_CODE:
        return LANGUAGE_BY_CODE[norm]
    lower = norm.lower()
    for k, spec in LANGUAGE_BY_CODE.items():
        if k.lower() == lower or spec.whisper_code.lower() == lower:
            return spec
    return LANGUAGE_BY_CODE["tr"]


class LanguageSearchResult:
    def __init__(self, code: str, display_label: str, spec=None):
        self.code = code
        self.display_label = display_label
        self.spec = spec
        if spec:
            self.turkish_name = spec.turkish_name
            self.native_name = spec.native_name
            self.english_name = spec.english_name
            self.is_rtl = spec.is_rtl
            self.script = spec.script
            self.sample_text = spec.sample_text
            self.whisper_code = spec.whisper_code
        else:
            self.turkish_name = display_label
            self.native_name = display_label
            self.english_name = display_label
            self.is_rtl = False
            self.script = "latin"
            self.sample_text = ""
            self.whisper_code = code

    def __iter__(self):
        yield self.code
        yield self.display_label

    def __getitem__(self, idx):
        if idx == 0:
            return self.code
        elif idx == 1:
            return self.display_label
        raise IndexError("LanguageSearchResult index out of range")

    def __repr__(self):
        return f"<LanguageSearchResult {self.code}: {self.display_label}>"


def search_languages(query: str = "", include_auto: bool = False):
    """
    Dil seçici arama kutusu için LanguageSearchResult listesi döndürür.
    Hem (code, display_label) unpacking'i hem de .code özelliğini destekler.
    """
    q = normalize_unicode_text(query).strip().lower()
    results = []
    if include_auto:
        auto_label = "Otomatik Algıla (Auto-Detect)"
        if not q or any(tok in auto_label.lower() for tok in (q, "auto", "otomatik")):
            results.append(LanguageSearchResult("auto", auto_label))

    for spec in SUPPORTED_LANGUAGES:
        haystack = f"{spec.code} {spec.native_name} {spec.turkish_name} {spec.english_name}".lower()
        if not q or q in haystack:
            results.append(LanguageSearchResult(spec.code, spec.display_label, spec))
    return results


# ---------------------------------------------------------------------------
# Temel Arayüz Çeviri Anahtarları (Türkçe Referans + İngilizce + 30 Dil Özeti)
# ---------------------------------------------------------------------------
BASE_TR: Dict[str, str] = {
    "nav.home": "Ana Sayfa",
    "nav.editor": "Video Editörü",
    "nav.stems": "Ses Ayrıştırma",
    "nav.dubbing": "Dublaj Stüdyosu",
    "nav.library": "Medya Kütüphanesi",
    "nav.studio": "Kayıt Stüdyosu",
    "nav.processing": "Ses İşleme",
    "nav.transcription": "Otomatik Altyazı",
    "nav.tasks": "Görevler",
    "nav.settings": "Ayarlar",
    "top.search_placeholder": "Proje, dosya veya özellik ara...",
    "top.active_project": "AKTİF PROJE",
    "top.no_project": "Proje yok",
    "top.ready": "● HAZIR",
    "top.recording": "● KAYIT",
    "top.paused": "● DURAKLATILDI",
    "transport.record": "Kayıt",
    "transport.stop_record": "Kaydı Bitir",
    "transport.cancel": "İptal",
    "editor.back_home": "Ana Sayfa",
    "editor.saved": "Kaydedildi",
    "editor.unsaved": "Kaydedilmemiş değişiklikler *",
    "editor.select_tool": "Seçim (V)",
    "editor.blade_tool": "Kes (B)",
    "editor.split_at_playhead": "Böl (Ctrl+K)",
    "editor.delete_clip": "Sil (Del)",
    "editor.snap": "Mıknatıs",
    "editor.add_marker": "İşaretçi (M)",
    "editor.export": "Dışa Aktar",
    "editor.transfer_to_editor": "Editöre Aktar",
    "editor.transfer_to_timeline": "Timeline'a Aktar",
    "transcription.title": "Otomatik Transkripsiyon ve Altyazı Stüdyosu",
    "transcription.auto_detect": "Otomatik Algıla (Auto-Detect)",
    "processing.title": "Profesyonel Ses İşleme Laboratuvarı",
    "studio.title": "Kayıt Stüdyosu ve Mikrofon Kalibrasyonu",
    "studio.calibration_wizard": "Mikrofon Kalibrasyon Sihirbazı",
    "settings.title": "Uygulama ve Stüdyo Ayarları",
    "settings.ui_language": "Arayüz Dili",
    "settings.transcription_language": "Varsayılan Transkripsiyon Dili",
}

BASE_EN: Dict[str, str] = {
    "nav.home": "Home",
    "nav.editor": "Video Editor",
    "nav.stems": "Stem Separation",
    "nav.dubbing": "Dubbing Studio",
    "nav.library": "Media Library",
    "nav.studio": "Recording Studio",
    "nav.processing": "Audio Lab",
    "nav.transcription": "Auto Subtitles",
    "nav.tasks": "Tasks",
    "nav.settings": "Settings",
    "top.search_placeholder": "Search projects, files or tools...",
    "top.active_project": "ACTIVE PROJECT",
    "top.no_project": "No project",
    "top.ready": "● READY",
    "top.recording": "● RECORDING",
    "top.paused": "● PAUSED",
    "transport.record": "Record",
    "transport.stop_record": "Stop Recording",
    "transport.cancel": "Cancel",
    "editor.back_home": "Home",
    "editor.saved": "Saved",
    "editor.unsaved": "Unsaved changes *",
    "editor.select_tool": "Select (V)",
    "editor.blade_tool": "Cut / Blade (B)",
    "editor.split_at_playhead": "Split (Ctrl+K)",
    "editor.delete_clip": "Delete (Del)",
    "editor.snap": "Snap",
    "editor.add_marker": "Marker (M)",
    "editor.export": "Export",
    "editor.transfer_to_editor": "Send to Editor",
    "editor.transfer_to_timeline": "Send to Timeline",
    "transcription.title": "Automatic Transcription & Subtitle Studio",
    "transcription.auto_detect": "Auto-Detect Language",
    "processing.title": "Professional Audio Processing Lab",
    "studio.title": "Recording Studio & Mic Calibration",
    "studio.calibration_wizard": "Microphone Calibration Wizard",
    "settings.title": "Application & Studio Settings",
    "settings.ui_language": "Interface Language",
    "settings.transcription_language": "Default Transcription Language",
}

# 30 dilin tamamı için ana navigasyon ve aksiyon sözlüğü (eksik anahtarlar otomatik BASE_EN / BASE_TR ile tamamlanır)
LOCALIZED_OVERRIDES: Dict[str, Dict[str, str]] = {
    "de": {
        "nav.home": "Startseite", "nav.editor": "Video-Editor", "nav.stems": "Stimmtrennung",
        "nav.dubbing": "Synchronstudio", "nav.library": "Mediathek", "nav.studio": "Aufnahmestudio",
        "nav.processing": "Audiolabor", "nav.transcription": "Untertitel", "nav.tasks": "Aufgaben",
        "nav.settings": "Einstellungen", "editor.export": "Exportieren", "transport.record": "Aufnahme",
    },
    "fr": {
        "nav.home": "Accueil", "nav.editor": "Éditeur Vidéo", "nav.stems": "Séparation Audio",
        "nav.dubbing": "Studio Doublage", "nav.library": "Médiathèque", "nav.studio": "Studio d'Enregistrement",
        "nav.processing": "Labo Audio", "nav.transcription": "Sous-titres", "nav.tasks": "Tâches",
        "nav.settings": "Paramètres", "editor.export": "Exporter", "transport.record": "Enregistrer",
    },
    "es": {
        "nav.home": "Inicio", "nav.editor": "Editor de Video", "nav.stems": "Separación de Voz",
        "nav.dubbing": "Estudio de Doblaje", "nav.library": "Biblioteca", "nav.studio": "Estudio de Grabación",
        "nav.processing": "Laboratorio de Audio", "nav.transcription": "Subtítulos", "nav.tasks": "Tareas",
        "nav.settings": "Ajustes", "editor.export": "Exportar", "transport.record": "Grabar",
    },
    "it": {
        "nav.home": "Home", "nav.editor": "Editor Video", "nav.stems": "Separazione Audio",
        "nav.dubbing": "Studio Doppiaggio", "nav.library": "Libreria Media", "nav.studio": "Studio Registrazione",
        "nav.processing": "Laboratorio Audio", "nav.transcription": "Sottotitoli", "nav.tasks": "Attività",
        "nav.settings": "Impostazioni", "editor.export": "Esporta", "transport.record": "Registra",
    },
    "pt": {
        "nav.home": "Início", "nav.editor": "Editor de Vídeo", "nav.stems": "Separação de Áudio",
        "nav.dubbing": "Estúdio de Dobragem", "nav.library": "Biblioteca", "nav.studio": "Estúdio de Gravação",
        "nav.processing": "Laboratório de Áudio", "nav.transcription": "Legendas", "nav.tasks": "Tarefas",
        "nav.settings": "Definições", "editor.export": "Exportar", "transport.record": "Gravar",
    },
    "pt-BR": {
        "nav.home": "Início", "nav.editor": "Editor de Vídeo", "nav.stems": "Separação de Áudio",
        "nav.dubbing": "Estúdio de Dublagem", "nav.library": "Biblioteca", "nav.studio": "Estúdio de Gravação",
        "nav.processing": "Laboratório de Áudio", "nav.transcription": "Legendas", "nav.tasks": "Tarefas",
        "nav.settings": "Configurações", "editor.export": "Exportar", "transport.record": "Gravar",
    },
    "nl": {
        "nav.home": "Start", "nav.editor": "Video-editor", "nav.stems": "Audioscheiding",
        "nav.dubbing": "Nasynchronisatie", "nav.library": "Mediatheek", "nav.studio": "Opnamestudio",
        "nav.processing": "Audiolab", "nav.transcription": "Ondertitels", "nav.tasks": "Taken",
        "nav.settings": "Instellingen", "editor.export": "Exporteren", "transport.record": "Opnemen",
    },
    "pl": {
        "nav.home": "Strona główna", "nav.editor": "Edytor Wideo", "nav.stems": "Separacja Dźwięku",
        "nav.dubbing": "Studio Dubbingu", "nav.library": "Biblioteka", "nav.studio": "Studio Nagrań",
        "nav.processing": "Laboratorium Audio", "nav.transcription": "Napisy", "nav.tasks": "Zadania",
        "nav.settings": "Ustawienia", "editor.export": "Eksportuj", "transport.record": "Nagraj",
    },
    "cs": {
        "nav.home": "Domů", "nav.editor": "Video Editor", "nav.stems": "Separace Zvuku",
        "nav.dubbing": "Dabingové Studio", "nav.library": "Knihovna", "nav.studio": "Nahrávací Studio",
        "nav.processing": "Zvuková Laboratoř", "nav.transcription": "Titulky", "nav.tasks": "Úlohy",
        "nav.settings": "Nastavení", "editor.export": "Exportovat", "transport.record": "Nahrávat",
    },
    "sk": {
        "nav.home": "Domov", "nav.editor": "Video Editor", "nav.stems": "Separácia Zvuku",
        "nav.dubbing": "Dabingové Štúdio", "nav.library": "Knižnica", "nav.studio": "Nahrávacie Štúdio",
        "nav.processing": "Zvukové Laboratórium", "nav.transcription": "Titulky", "nav.tasks": "Úlohy",
        "nav.settings": "Nastavenia", "editor.export": "Exportovať", "transport.record": "Nahrávať",
    },
    "hu": {
        "nav.home": "Kezdőlap", "nav.editor": "Videószerkesztő", "nav.stems": "Hangsávszétválasztás",
        "nav.dubbing": "Szinkronstúdió", "nav.library": "Médiatár", "nav.studio": "Hangstúdió",
        "nav.processing": "Hanglabor", "nav.transcription": "Feliratok", "nav.tasks": "Feladatok",
        "nav.settings": "Beállítások", "editor.export": "Exportálás", "transport.record": "Felvétel",
    },
    "ro": {
        "nav.home": "Acasă", "nav.editor": "Editor Video", "nav.stems": "Separare Audio",
        "nav.dubbing": "Studio Dublaj", "nav.library": "Bibliotecă", "nav.studio": "Studio Înregistrări",
        "nav.processing": "Laborator Audio", "nav.transcription": "Subtitrări", "nav.tasks": "Sarcini",
        "nav.settings": "Setări", "editor.export": "Exportă", "transport.record": "Înregistrează",
    },
    "bg": {
        "nav.home": "Начало", "nav.editor": "Видео Редактор", "nav.stems": "Разделяне на Звук",
        "nav.dubbing": "Студио за Дублаж", "nav.library": "Библиотека", "nav.studio": "Звукозапис",
        "nav.processing": "Аудио Лаборатория", "nav.transcription": "Субтитри", "nav.tasks": "Задачи",
        "nav.settings": "Настройки", "editor.export": "Експорт", "transport.record": "Запис",
    },
    "el": {
        "nav.home": "Αρχική", "nav.editor": "Επεξεργασία Βίντεο", "nav.stems": "Διαχωρισμός Ήχου",
        "nav.dubbing": "Στούντιο Μεταγλώττισης", "nav.library": "Βιβλιοθήκη", "nav.studio": "Στούντιο Ηχογράφησης",
        "nav.processing": "Εργαστήριο Ήχου", "nav.transcription": "Υπότιτλοι", "nav.tasks": "Εργασίες",
        "nav.settings": "Ρυθμίσεις", "editor.export": "Εξαγωγή", "transport.record": "Εγγραφή",
    },
    "ru": {
        "nav.home": "Главная", "nav.editor": "Видеоредактор", "nav.stems": "Разделение Дорожек",
        "nav.dubbing": "Студия Дубляжа", "nav.library": "Медиатека", "nav.studio": "Студия Записи",
        "nav.processing": "Аудиолаборатория", "nav.transcription": "Субтитры", "nav.tasks": "Задачи",
        "nav.settings": "Настройки", "editor.export": "Экспорт", "transport.record": "Запись",
    },
    "uk": {
        "nav.home": "Головна", "nav.editor": "Відеоредактор", "nav.stems": "Розділення Звуку",
        "nav.dubbing": "Студія Дубляжу", "nav.library": "Медіатека", "nav.studio": "Студія Запису",
        "nav.processing": "Аудіолабораторія", "nav.transcription": "Субтитри", "nav.tasks": "Завдання",
        "nav.settings": "Налаштування", "editor.export": "Експорт", "transport.record": "Запис",
    },
    "ar": {
        "nav.home": "الرئيسية", "nav.editor": "محرر الفيديو", "nav.stems": "فصل الصوت",
        "nav.dubbing": "استوديو الدبلجة", "nav.library": "مكتبة الوسائط", "nav.studio": "استوديو التسجيل",
        "nav.processing": "مختبر الصوت", "nav.transcription": "الترجمة التلقائية", "nav.tasks": "المهام",
        "nav.settings": "الإعدادات", "editor.export": "تصدير", "transport.record": "تسجيل",
    },
    "he": {
        "nav.home": "דף הבית", "nav.editor": "עורך וידאו", "nav.stems": "הפרדת ערוצים",
        "nav.dubbing": "אולפן דיבוב", "nav.library": "ספריית מדיה", "nav.studio": "אולפן הקלטות",
        "nav.processing": "מעבדת שמע", "nav.transcription": "כתוביות אוטומטיות", "nav.tasks": "משימות",
        "nav.settings": "הגדרות", "editor.export": "ייצוא", "transport.record": "הקלטה",
    },
    "fa": {
        "nav.home": "صفحه اصلی", "nav.editor": "ویرایشگر ویدیو", "nav.stems": "جداسازی صدا",
        "nav.dubbing": "استودیو دوبله", "nav.library": "کتابخانه رسانه", "nav.studio": "استودیو ضبط",
        "nav.processing": "آزمایشگاه صدا", "nav.transcription": "زیرنویس خودکار", "nav.tasks": "وظایف",
        "nav.settings": "تنظیمات", "editor.export": "خروجی گرفتن", "transport.record": "ضبط",
    },
    "hi": {
        "nav.home": "मुख्य पृष्ठ", "nav.editor": "वीडियो संपादक", "nav.stems": "ऑडियो पृथक्करण",
        "nav.dubbing": "डबिंग स्टूडियो", "nav.library": "मीडिया लाइब्रेरी", "nav.studio": "रिकॉर्डिंग स्टूडियो",
        "nav.processing": "ऑडियो प्रयोगशाला", "nav.transcription": "ऑटो सबटाइटल", "nav.tasks": "कार्य",
        "nav.settings": "सेटिंग्स", "editor.export": "निर्यात करें", "transport.record": "रिकॉर्ड",
    },
    "bn": {
        "nav.home": "হোম", "nav.editor": "ভিডিও এডিটর", "nav.stems": "অডিও পৃথকীকরণ",
        "nav.dubbing": "ডাবিং স্টুডিও", "nav.library": "মিডিয়া লাইব্রেরি", "nav.studio": "রেকর্ডিং স্টুডিও",
        "nav.processing": "অডিও ল্যাব", "nav.transcription": "অটো সাবটাইটেল", "nav.tasks": "কার্যতালিকা",
        "nav.settings": "সেটিংস", "editor.export": "এক্সপোর্ট", "transport.record": "রেকর্ড",
    },
    "ur": {
        "nav.home": "ہوم", "nav.editor": "ویڈیو ایڈیٹر", "nav.stems": "آڈیو علیحدگی",
        "nav.dubbing": "ڈبنگ اسٹوڈیو", "nav.library": "میڈیا لائبریری", "nav.studio": "ریکارڈنگ اسٹوڈیو",
        "nav.processing": "آڈیو لیب", "nav.transcription": "خودکار سب ٹائٹلز", "nav.tasks": "مہمات",
        "nav.settings": "ترتیبات", "editor.export": "برآمد کریں", "transport.record": "ریکارڈ",
    },
    "ja": {
        "nav.home": "ホーム", "nav.editor": "動画エディター", "nav.stems": "音源分離 (AI)",
        "nav.dubbing": "アフレコスタジオ", "nav.library": "メディアライブラリ", "nav.studio": "録音スタジオ",
        "nav.processing": "オーディオラボ", "nav.transcription": "自動字幕・文字起こし", "nav.tasks": "タスク",
        "nav.settings": "設定", "editor.export": "書き出し", "transport.record": "録音",
    },
    "ko": {
        "nav.home": "홈", "nav.editor": "비디오 편집기", "nav.stems": "AI 음원 분리",
        "nav.dubbing": "더빙 스튜디오", "nav.library": "미디어 라이브러리", "nav.studio": "녹음 스튜디오",
        "nav.processing": "오디오 랩", "nav.transcription": "자동 자막", "nav.tasks": "작업",
        "nav.settings": "설정", "editor.export": "내보내기", "transport.record": "녹음",
    },
    "zh-CN": {
        "nav.home": "主页", "nav.editor": "视频编辑器", "nav.stems": "AI 人声伴奏分离",
        "nav.dubbing": "配音工作室", "nav.library": "媒体库", "nav.studio": "录音室",
        "nav.processing": "音频实验室", "nav.transcription": "自动字幕与转写", "nav.tasks": "后台任务",
        "nav.settings": "设置", "editor.export": "导出", "transport.record": "录音",
    },
    "zh-TW": {
        "nav.home": "首頁", "nav.editor": "影片編輯器", "nav.stems": "AI 音軌分離",
        "nav.dubbing": "配音工作室", "nav.library": "媒體庫", "nav.studio": "錄音室",
        "nav.processing": "音訊實驗室", "nav.transcription": "自動字幕與轉寫", "nav.tasks": "背景工作",
        "nav.settings": "設定", "editor.export": "匯出", "transport.record": "錄音",
    },
    "id": {
        "nav.home": "Beranda", "nav.editor": "Editor Video", "nav.stems": "Pemisahan Audio",
        "nav.dubbing": "Studio Sulih Suara", "nav.library": "Pustaka Media", "nav.studio": "Studio Rekaman",
        "nav.processing": "Lab Audio", "nav.transcription": "Subtitel Otomatis", "nav.tasks": "Tugas",
        "nav.settings": "Pengaturan", "editor.export": "Ekspor", "transport.record": "Rekam",
    },
    "vi": {
        "nav.home": "Trang chủ", "nav.editor": "Trình dựng Video", "nav.stems": "Tách âm thanh AI",
        "nav.dubbing": "Phòng thu Lồng tiếng", "nav.library": "Thư viện Media", "nav.studio": "Phòng thu âm",
        "nav.processing": "Phòng Lab Âm thanh", "nav.transcription": "Phụ đề Tự động", "nav.tasks": "Tác vụ",
        "nav.settings": "Cài đặt", "editor.export": "Xuất file", "transport.record": "Ghi âm",
    },
    "sv": {
        "nav.home": "Hem", "nav.editor": "Videoredigerare", "nav.stems": "Stamseparation",
        "nav.dubbing": "Dubbningsstudio", "nav.library": "Mediebibliotek", "nav.studio": "Inspelningsstudio",
        "nav.processing": "Ljudlabb", "nav.transcription": "Undertexter", "nav.tasks": "Uppgifter",
        "nav.settings": "Inställningar", "editor.export": "Exportera", "transport.record": "Spela in",
    },
    "no": {
        "nav.home": "Hjem", "nav.editor": "Videoredigering", "nav.stems": "Stemmefraksjon",
        "nav.dubbing": "Dubbestudio", "nav.library": "Mediebibliotek", "nav.studio": "Lydstudio",
        "nav.processing": "Lydlab", "nav.transcription": "Undertekster", "nav.tasks": "Oppgaver",
        "nav.settings": "Innstillinger", "editor.export": "Eksporter", "transport.record": "Ta opp",
    },
    "da": {
        "nav.home": "Hjem", "nav.editor": "Videoredigering", "nav.stems": "Lydadskillelse",
        "nav.dubbing": "Dubbingstudie", "nav.library": "Mediebibliotek", "nav.studio": "Optagestudie",
        "nav.processing": "Lydlab", "nav.transcription": "Undertekster", "nav.tasks": "Opgaver",
        "nav.settings": "Indstillinger", "editor.export": "Eksportér", "transport.record": "Optag",
    },
    "fi": {
        "nav.home": "Koti", "nav.editor": "Videoeditori", "nav.stems": "Äänen erottelu",
        "nav.dubbing": "Jälkiäänitys", "nav.library": "Mediakirjasto", "nav.studio": "Äänitysstudio",
        "nav.processing": "Äänilaboratorio", "nav.transcription": "Tekstitykset", "nav.tasks": "Tehtävät",
        "nav.settings": "Asetukset", "editor.export": "Vie", "transport.record": "Äänitä",
    },
    "th": {
        "nav.home": "หน้าแรก", "nav.editor": "เครื่องมือตัดต่อวิดีโอ", "nav.stems": "แยกเสียง AI",
        "nav.dubbing": "สตูดิโอพากย์เสียง", "nav.library": "คลังมีเดีย", "nav.studio": "ห้องอัดเสียง",
        "nav.processing": "ห้องปฏิบัติการเสียง", "nav.transcription": "คำบรรยายอัตโนมัติ", "nav.tasks": "งาน",
        "nav.settings": "การตั้งค่า", "editor.export": "ส่งออก", "transport.record": "บันทึกเสียง",
    },
}


def _build_full_catalog() -> Dict[str, Dict[str, str]]:
    catalog: Dict[str, Dict[str, str]] = {
        "tr": dict(BASE_TR),
        "en": dict(BASE_EN),
    }
    for spec in SUPPORTED_LANGUAGES:
        if spec.code in ("tr", "en"):
            continue
        merged = dict(BASE_EN)
        merged.update(LOCALIZED_OVERRIDES.get(spec.code, {}))
        catalog[spec.code] = merged
    return catalog


TRANSLATIONS: Dict[str, Dict[str, str]] = _build_full_catalog()


class I18nManager:
    """
    Uygulama genelinde arayüz dilini (ui_language) ve varsayılan transkripsiyon
    dilini (transcription_language) birbirinden bağımsız yöneten yerelleştirme sınıfı.
    """

    def __init__(self, config=None):
        self.config = config
        self._listeners: List[Callable[[str], None]] = []

    def add_listener(self, cb: Callable[[str], None]) -> None:
        if cb not in self._listeners:
            self._listeners.append(cb)

    def get_ui_language(self) -> str:
        if self.config is not None:
            code = self.config.get_str("ui_language", "") or self.config.get_str("language", "tr")
        else:
            code = "tr"
        return get_language_spec(code).code

    def set_ui_language(self, code: str, auto_save: bool = True) -> str:
        spec = get_language_spec(code)
        if self.config is not None:
            self.config.update(
                {"ui_language": spec.code, "language": spec.code},
                auto_save=auto_save,
            )
        for cb in list(self._listeners):
            try:
                cb(spec.code)
            except Exception as exc:
                logger.warning(f"Dil değişim dinleyicisi hatası: {exc}")
        return spec.code

    def get_transcription_language(self) -> str:
        """'auto' veya 30 dilden birinin standart kodunu döndürür."""
        if self.config is None:
            return "auto"
        raw = self.config.get_str("transcription_language", "auto").strip()
        if not raw or raw.lower() == "auto":
            return "auto"
        return get_language_spec(raw).code

    def set_transcription_language(self, code: Optional[str], auto_save: bool = True) -> str:
        if not code or str(code).strip().lower() == "auto":
            norm = "auto"
        else:
            norm = get_language_spec(code).code
        if self.config is not None:
            self.config.set("transcription_language", norm, auto_save=auto_save)
        return norm

    def is_rtl(self, code: Optional[str] = None) -> bool:
        target = get_language_spec(code or self.get_ui_language()).code
        return target in RTL_LANGUAGE_CODES

    def font_families_for_language(self, code: Optional[str] = None) -> List[str]:
        spec = get_language_spec(code or self.get_ui_language())
        return list(SCRIPT_FONT_FAMILIES.get(spec.script, SCRIPT_FONT_FAMILIES["latin"]))

    def t(self, key: str, lang: Optional[str] = None, default: Optional[str] = None, lang_code: Optional[str] = None, **kwargs) -> str:
        """
        Çeviri anahtarını çözümler:
        1. Seçili dil tablosu
        2. Türkçe referans tablosu (BASE_TR)
        3. İngilizce referans tablosu (BASE_EN)
        4. Varsa default parametresi veya anahtarın kendisi (asla boş string dönmez!)
        """
        target_code = get_language_spec(lang_code or lang or self.get_ui_language()).code
        lang_dict = TRANSLATIONS.get(target_code, {})
        val = lang_dict.get(key)
        if not val:
            val = BASE_TR.get(key) or BASE_EN.get(key) or default or key
        if kwargs:
            try:
                val = val.format(**kwargs)
            except Exception:
                pass
        return normalize_unicode_text(val)

    @staticmethod
    def find_missing_keys(lang_code: str) -> List[str]:
        """Belirtilen dil tablosunda BASE_TR referansına göre eksik kalan anahtarları döndürür."""
        spec = get_language_spec(lang_code)
        table = TRANSLATIONS.get(spec.code, {})
        return [k for k in BASE_TR.keys() if k not in table or not str(table[k]).strip()]
