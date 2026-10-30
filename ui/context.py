"""
Uygulama bağlamı: yapılandırma, cihazlar ve ses motorlarının tek noktadan
yönetimi. Arayüz bu nesne üzerinden motora erişir; motorlar Qt'den habersizdir.

Ses iş parçacıklarından gelen olaylar ``AudioBridge`` sinyalleri ile GUI
iş parçacığına aktarılır (Qt otomatik kuyruk bağlantısı).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from audio.devices import AudioDevice, DeviceManager
from audio.monitor import InputMonitor
from audio.player import PlaybackInfo, Player
from audio.recorder import RecordingConfig, RecordingResult, Recorder, build_recording_filename
from core.config import ConfigManager
from core.diagnostics import TaskManager
from core.i18n import I18nManager
from core.logger import get_logger
from project.manager import ProjectManager

logger = get_logger("ui.context")


class AudioBridge(QObject):
    recorderState = Signal(str)
    recorderError = Signal(str)
    playerState = Signal(str)
    playerError = Signal(str)
    monitorError = Signal(str)
    analysisReady = Signal(object)
    analysisFailed = Signal(str)


class AppContext(QObject):
    devicesChanged = Signal()
    projectChanged = Signal(object)
    tasksChanged = Signal()
    languageChanged = Signal(str)
    stemsToTimelineRequested = Signal(object, dict, str)
    transcriptionToTimelineRequested = Signal(object)

    def __init__(self, config: ConfigManager, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.config = config
        self.i18n = I18nManager(config)
        self.i18n.add_listener(lambda code: self.languageChanged.emit(code))
        self.bridge = AudioBridge(self)
        self.devices = DeviceManager()
        self.project_manager = ProjectManager(config)
        self.project_manager.on_changed(lambda proj: self.projectChanged.emit(proj))
        self.task_manager = TaskManager()
        self.task_manager.add_listener(lambda: self.tasksChanged.emit())

        self.recorder = Recorder(
            on_state=lambda s: self.bridge.recorderState.emit(s.value),
            on_error=lambda m: self.bridge.recorderError.emit(m),
        )
        self.player = Player(
            device_index=self.output_device_index(),
            on_state=lambda s: self.bridge.playerState.emit(s.value),
            on_error=lambda m: self.bridge.playerError.emit(m),
        )
        self.monitor = InputMonitor(
            on_error=lambda m: self.bridge.monitorError.emit(m),
        )
        self.last_recording: Optional[RecordingResult] = None
        self.current_info: Optional[PlaybackInfo] = None

    # ---------------------------------------------------------------- cihazlar
    def refresh_devices(self) -> None:
        self.devices.refresh()
        self.devicesChanged.emit()

    def input_device(self) -> Optional[AudioDevice]:
        return self.devices.find(
            name=self.config.get_str("input_device_name", ""),
            hostapi=self.config.get("input_hostapi"),
            index=self.config.get("input_device_id"),
            input_only=True,
        ) or self.devices.best_input()

    def output_device(self) -> Optional[AudioDevice]:
        return self.devices.find(
            name=self.config.get_str("output_device_name", ""),
            hostapi=self.config.get("output_hostapi"),
            index=self.config.get("output_device_id"),
            input_only=False,
        ) or self.devices.best_output()

    def output_device_index(self) -> Optional[int]:
        dev = self.output_device()
        return dev.index if dev else None

    def remember_device(self, dev: AudioDevice, kind: str) -> None:
        if kind == "input":
            self.config.update(
                {
                    "input_device_id": dev.index,
                    "input_device_name": dev.name,
                    "input_hostapi": dev.hostapi_name,
                }
            )
        else:
            self.config.update(
                {
                    "output_device_id": dev.index,
                    "output_device_name": dev.name,
                    "output_hostapi": dev.hostapi_name,
                }
            )
            self.player.set_device(dev.index)

    # --------------------------------------------------------------- kayıt
    def build_recording_config(self, label: str = "") -> RecordingConfig:
        device = self.input_device()
        return RecordingConfig(
            path=Path(self.config.recordings_dir) / "kayit.wav",
            device_index=device.index if device else None,
            device_name=device.name if device else "",
            samplerate=self.config.get_int("sample_rate", 48000),
            channels=self.config.get_int("channels", 1),
            subtype=self.config.get_str("recording_subtype", "PCM_24"),
            blocksize=self.config.get_int("block_size", 1024),
            label=label,
        )

    def new_recording_path(self, label: str = "") -> Path:
        """Çakışmayan yeni kayıt dosyası yolu üretir (var olan dosya korunur)."""
        directory = self.recordings_dir()
        directory.mkdir(parents=True, exist_ok=True)
        name = build_recording_filename(label)
        candidate = directory / name
        if not candidate.exists():
            return candidate
        stem, suffix = candidate.stem, candidate.suffix
        for i in range(1, 1000):
            alt = directory / f"{stem}-{i}{suffix}"
            if not alt.exists():
                return alt
        raise RuntimeError("Kayıt için benzersiz dosya adı üretilemedi.")

    def recordings_dir(self) -> Path:
        return Path(self.config.recordings_dir)

    def imported_media_paths(self) -> list[str]:
        raw = self.config.get("imported_media_files", []) or []
        return [str(p) for p in raw if Path(p).exists()]

    def add_imported_media(self, path: str | Path) -> None:
        resolved = str(Path(path).resolve())
        items = list(self.config.get("imported_media_files", []) or [])
        if resolved in items:
            items.remove(resolved)
        items.insert(0, resolved)
        self.config.set("imported_media_files", items[:100])

    # --------------------------------------------------------------- oynatma
    def load_for_playback(self, path: str | Path) -> PlaybackInfo:
        info = self.player.load(path)
        self.current_info = info
        return info

    def shutdown(self) -> None:
        try:
            if self.recorder.is_active:
                self.recorder.stop()
        except Exception:
            logger.exception("Kapanışta kayıt durdurulamadı")
        try:
            self.player.close()
        except Exception:
            logger.exception("Kapanışta oynatıcı kapatılamadı")
        self.config.save()
