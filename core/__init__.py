"""
VOX Studio paket kökü.

Alt modüller:
    core    — yapılandırma, yollar, loglama, sabitler
    audio   — cihaz yönetimi, kayıt, oynatma, seviye ölçümü
    dsp     — ses analizi ve işleme
    project — SQLite proje modeli (Aşama 3)
    takes   — take yönetimi (Aşama 3)
    dubbing — video / zaman çizelgesi (Aşama 5)
    ai      — transkripsiyon ve stem ayrıştırma (Aşama 6)
    ui      — Qt arayüzü
"""

from .constants import APP_NAME, APP_VERSION

__all__ = ["APP_NAME", "APP_VERSION"]
