"""
Ses cihazı yönetimi (PortAudio / sounddevice).

Kurallar
--------
* Cihaz **kimliği index değildir**; index oturumlar arasında değişir. Kalıcı
  kimlik ``(hostapi, ad)`` çiftidir. Index yalnızca o anki oturum içindir.
* Windows'ta MME/WDM-KS cihaz adları bozuk kodlanmış olabilir; adlar
  WASAPI/DirectSound üzerinden tercih edilir.
* Bir örnekleme hızı/kanal kombinasyonu gerçekten destekleniyor mu, PortAudio
  ``check_input_settings`` / ``check_output_settings`` ile denenerek doğrulanır.
  Desteklenmeyen kombinasyon "çalışıyor" gibi gösterilmez.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Sequence, Tuple

import re

import sounddevice as sd

from core.logger import get_logger

logger = get_logger("audio.devices")

# Adları güvenilir olan host API'ler önce gelir.
HOSTAPI_PREFERENCE = ("Windows WASAPI", "Windows DirectSound", "MME", "Windows WDM-KS")
_RELIABLE_NAMES = ("Windows WASAPI", "Windows DirectSound")


def _decode_name(raw: str) -> str:
    """Bozuk kodlanmış cihaz adlarını onarmayı dener; olmazsa olduğu gibi bırakır."""
    text = (raw or "").strip()
    if not text:
        return "(adsız cihaz)"
    if "\ufffd" in text:
        for enc in ("utf-8", "cp1254", "latin-1"):
            try:
                fixed = text.encode(enc, errors="strict").decode("utf-8")
            except Exception:
                continue
            if "\ufffd" not in fixed and fixed:
                return fixed.strip()
    return text


def _hostapi_name(index: int) -> str:
    try:
        return sd.query_hostapis(index).get("name", f"HostAPI {index}")
    except Exception:
        return f"HostAPI {index}"


def _preference_rank(hostapi_name: str) -> int:
    try:
        return HOSTAPI_PREFERENCE.index(hostapi_name)
    except ValueError:
        return len(HOSTAPI_PREFERENCE)


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    hostapi_index: int
    hostapi_name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool = False
    is_default_output: bool = False

    @property
    def key(self) -> Tuple[str, str]:
        return (self.hostapi_name, self.name)

    @property
    def input_capable(self) -> bool:
        return self.max_input_channels > 0

    @property
    def output_capable(self) -> bool:
        return self.max_output_channels > 0

    @property
    def name_is_reliable(self) -> bool:
        return self.hostapi_name in _RELIABLE_NAMES and "\ufffd" not in self.name


class DeviceError(RuntimeError):
    """Kullanıcıya gösterilebilir cihaz hatası."""


class DeviceManager:
    def __init__(self) -> None:
        self._devices: List[AudioDevice] = []
        self._input_default: Optional[int] = None
        self._output_default: Optional[int] = None
        self.refresh()

    # ------------------------------------------------------------------ tarama
    def refresh(self) -> None:
        try:
            raw = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as exc:
            logger.error("Cihaz listesi alınamadı: %s", exc)
            self._devices = []
            return

        try:
            defaults = sd.default.device
            self._input_default = defaults[0] if defaults and defaults[0] is not None and defaults[0] >= 0 else None
            self._output_default = defaults[1] if defaults and defaults[1] is not None and defaults[1] >= 0 else None
        except Exception:
            self._input_default = None
            self._output_default = None

        devices = []
        for idx, info in enumerate(raw):
            hostapi_index = int(info.get("hostapi", 0))
            try:
                hostapi_name = hostapis[hostapi_index].get("name", "")
            except Exception:
                hostapi_name = f"HostAPI {hostapi_index}"
            name = _decode_name(str(info.get("name", "")))
            devices.append(
                AudioDevice(
                    index=idx,
                    name=name,
                    hostapi_index=hostapi_index,
                    hostapi_name=hostapi_name,
                    max_input_channels=int(info.get("max_input_channels", 0)),
                    max_output_channels=int(info.get("max_output_channels", 0)),
                    default_samplerate=float(info.get("default_samplerate", 0) or 0),
                    is_default_input=(self._input_default == idx),
                    is_default_output=(self._output_default == idx),
                )
            )

        # Varsayılan cihazın diğer host API'lerdeki eşlerini de "varsayılan" işaretle
        def _peer_key(text: str) -> str:
            return re.sub(r"\s+", " ", text.strip().lower())[:22]

        input_peer = None
        output_peer = None
        for idx in (self._input_default, self._output_default):
            if idx is None or idx >= len(devices):
                continue
            key = _peer_key(devices[idx].name)
            if idx == self._input_default:
                input_peer = key
            if idx == self._output_default:
                output_peer = key

        for i, dev in enumerate(devices):
            if input_peer and dev.max_input_channels > 0 and _peer_key(dev.name) == input_peer:
                devices[i] = replace(dev, is_default_input=True)
            if output_peer and dev.max_output_channels > 0 and _peer_key(dev.name) == output_peer:
                devices[i] = replace(dev, is_default_output=True)

        devices.sort(key=lambda d: (_preference_rank(d.hostapi_name), d.name.lower()))
        self._devices = devices
        logger.info("%d ses cihazı bulundu", len(devices))

    # ------------------------------------------------------------------ sorgu
    def all_devices(self) -> List[AudioDevice]:
        return list(self._devices)

    def input_devices(self) -> List[AudioDevice]:
        return [d for d in self._devices if d.input_capable]

    def output_devices(self) -> List[AudioDevice]:
        return [d for d in self._devices if d.output_capable]

    def get(self, index: int) -> Optional[AudioDevice]:
        for d in self._devices:
            if d.index == index:
                return d
        return None

    def find(
        self,
        name: str = "",
        hostapi: Optional[str] = None,
        index: Optional[int] = None,
        input_only: bool = True,
    ) -> Optional[AudioDevice]:
        """Kalıcı kimliğe (hostapi + ad) göre cihazı bulur; index yalnızca ipucudur."""
        pool = self.input_devices() if input_only else self.output_devices()
        if not pool:
            return None
        if name:
            exact = [d for d in pool if d.name == name and (hostapi is None or d.hostapi_name == hostapi)]
            if exact:
                return exact[0]
            relaxed = [d for d in pool if d.name == name and d.name_is_reliable] or [
                d for d in pool if d.name == name
            ]
            if relaxed:
                return relaxed[0]
        if index is not None:
            for d in pool:
                if d.index == index:
                    return d
        return None

    def default_input(self) -> Optional[AudioDevice]:
        if self._input_default is not None:
            dev = self.get(self._input_default)
            if dev and dev.input_capable:
                return dev
        inputs = self.input_devices()
        return inputs[0] if inputs else None

    def default_output(self) -> Optional[AudioDevice]:
        if self._output_default is not None:
            dev = self.get(self._output_default)
            if dev and dev.output_capable:
                return dev
        outputs = self.output_devices()
        return outputs[0] if outputs else None

    def best_input(self) -> Optional[AudioDevice]:
        """Otomatik seçim: sistem varsayılanı → adı güvenilir cihaz → ilk giriş."""
        inputs = self.input_devices()
        for dev in inputs:
            if dev.is_default_input and dev.hostapi_name == "Windows WASAPI":
                return dev
        for dev in inputs:
            if dev.is_default_input:
                return dev
        for dev in inputs:
            if dev.name_is_reliable:
                return dev
        return inputs[0] if inputs else None

    def best_output(self) -> Optional[AudioDevice]:
        outputs = self.output_devices()
        for dev in outputs:
            if dev.is_default_output and dev.hostapi_name == "Windows WASAPI":
                return dev
        for dev in outputs:
            if dev.is_default_output:
                return dev
        for dev in outputs:
            if dev.name_is_reliable:
                return dev
        return outputs[0] if outputs else None

    # ------------------------------------------------------------ doğrulama
    @staticmethod
    def _try_setting(device: AudioDevice, kind: str, samplerate: float, channels: int) -> bool:
        try:
            if kind == "in":
                sd.check_input_settings(
                    device=device.index, samplerate=samplerate, channels=channels, dtype="float32"
                )
            else:
                sd.check_output_settings(
                    device=device.index, samplerate=samplerate, channels=channels, dtype="float32"
                )
        except Exception:
            return False
        return True

    def supported_sample_rates(
        self,
        device: AudioDevice,
        channels: int = 1,
        kind: str = "in",
        candidates: Sequence[int] = (44100, 48000, 88200, 96000, 192000),
    ) -> List[int]:
        ok: List[int] = []
        for rate in candidates:
            if self._try_setting(device, kind, float(rate), channels):
                ok.append(int(rate))
        return ok

    def supported_channel_counts(
        self, device: AudioDevice, samplerate: int, kind: str = "in"
    ) -> List[int]:
        limit = device.max_input_channels if kind == "in" else device.max_output_channels
        counts: List[int] = []
        for ch in range(1, max(1, limit) + 1):
            if self._try_setting(device, kind, float(samplerate), ch):
                counts.append(ch)
        return counts

    def validate_input(
        self, device: AudioDevice, samplerate: int, channels: int
    ) -> Tuple[bool, str]:
        """Kombinasyonu doğrular; kullanıcıya gösterilebilir Türkçe mesaj döndürür."""
        if device.max_input_channels <= 0:
            return False, f"'{device.name}' bir giriş cihazı değil."
        if channels > device.max_input_channels:
            return (
                False,
                f"'{device.name}' en fazla {device.max_input_channels} kanal giriş destekliyor; "
                f"istenen {channels} kanal.",
            )
        if self._try_setting(device, "in", float(samplerate), channels):
            return True, ""

        available = self.supported_sample_rates(device, channels, kind="in")
        alt = ""
        if available:
            alt = " Bu cihazda doğrulanan hızlar: " + ", ".join(f"{r} Hz" for r in available) + "."
        return (
            False,
            f"'{device.name}' cihazı {samplerate} Hz / {channels} kanal kombinasyonunu kabul etmedi."
            + alt
            + " Kayıt başlatılamadı; lütfen desteklenen bir hız seçin.",
        )

    def validate_output(
        self, device: AudioDevice, samplerate: int, channels: int
    ) -> Tuple[bool, str]:
        if device.max_output_channels <= 0:
            return False, f"'{device.name}' bir çıkış cihazı değil."
        if channels > device.max_output_channels:
            return (
                False,
                f"'{device.name}' en fazla {device.max_output_channels} kanal çıkış destekliyor.",
            )
        if self._try_setting(device, "out", float(samplerate), channels):
            return True, ""
        return (
            False,
            f"'{device.name}' çıkışı {samplerate} Hz / {channels} kanal ile açılamadı. "
            "Farklı bir çıkış cihazı veya örnekleme hızı seçin.",
        )

    def has_hostapi(self, name: str) -> bool:
        return any(d.hostapi_name == name for d in self._devices)

    def describe(self) -> str:
        """Tanılama için özet metin."""
        lines = [f"{len(self._devices)} cihaz"]
        for dev in self._devices:
            state = []
            if dev.is_default_input and dev.input_capable:
                state.append("varsayılan giriş")
            if dev.is_default_output and dev.output_capable:
                state.append("varsayılan çıkış")
            lines.append(
                f"  [{dev.index:>2}] {dev.name} | {dev.hostapi_name} | "
                f"in={dev.max_input_channels} out={dev.max_output_channels} | "
                f"{dev.default_samplerate:.0f} Hz" + (f" | {', '.join(state)}" if state else "")
            )
        return "\n".join(lines)


def sort_devices(devices: Iterable[AudioDevice]) -> List[AudioDevice]:
    return sorted(devices, key=lambda d: (_preference_rank(d.hostapi_name), d.name.lower()))
