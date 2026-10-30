"""core.config testleri: yükleme/kaydetme, atomiklik, bozuk dosya, eşzamanlılık."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from core.config import ConfigManager, DEFAULT_CONFIG


def test_defaults_written_on_first_use(isolated_home):
    cfg = ConfigManager(root=isolated_home)
    assert cfg.config_path.exists()
    on_disk = json.loads(cfg.config_path.read_text(encoding="utf-8"))
    assert on_disk["sample_rate"] == 48000
    assert on_disk["language"] == "tr"


def test_set_persists_and_reloads(isolated_home):
    cfg = ConfigManager(root=isolated_home)
    cfg.set("sample_rate", 44100)
    cfg.set("channels", 2)
    cfg.add_recent_project(r"C:\work\bolum-01.voxproj")

    again = ConfigManager(root=isolated_home)
    assert again.get("sample_rate") == 44100
    assert again.get("channels") == 2
    assert again.get("recent_projects") == [r"C:\work\bolum-01.voxproj"]


def test_recent_projects_deduplicate_and_cap(isolated_home):
    cfg = ConfigManager(root=isolated_home)
    for i in range(15):
        cfg.add_recent_project(f"p{i}")
    cfg.add_recent_project("p7")  # tekrar eklenince başa taşınır
    recent = cfg.get("recent_projects")
    assert recent[0] == "p7"
    assert len(recent) == 10
    cfg.remove_recent_project("p7")
    assert "p7" not in cfg.get("recent_projects")


def test_recordings_dir_falls_back_to_app_home(isolated_home):
    cfg = ConfigManager(root=isolated_home)
    assert cfg.recordings_dir == isolated_home / "recordings"
    cfg.set_recordings_dir(r"D:\Seslendirme")
    assert cfg.recordings_dir == Path(r"D:\Seslendirme")
    assert ConfigManager(root=isolated_home).recordings_dir == Path(r"D:\Seslendirme")


def test_corrupt_config_is_quarantined_not_deleted(isolated_home):
    isolated_home.mkdir(parents=True, exist_ok=True)
    bad = isolated_home / "config.json"
    bad.write_text("{ bu json degil ", encoding="utf-8")

    cfg = ConfigManager(root=isolated_home)

    assert cfg.last_load_error is not None
    assert cfg.get("sample_rate") == DEFAULT_CONFIG["sample_rate"]
    backups = list(isolated_home.glob("config.json.corrupt-*"))
    assert len(backups) == 1
    assert "bu json degil" in backups[0].read_text(encoding="utf-8")
    # Yeni dosya geçerli olmalı
    assert json.loads(cfg.config_path.read_text(encoding="utf-8"))["language"] == "tr"


def test_save_is_atomic_and_leaves_no_temp_files(isolated_home):
    cfg = ConfigManager(root=isolated_home)
    for i in range(20):
        cfg.set("auto_save_interval_sec", i)
    leftovers = [p.name for p in isolated_home.iterdir() if p.name.startswith(".config-")]
    assert leftovers == []
    assert ConfigManager(root=isolated_home).get("auto_save_interval_sec") == 19


def test_unknown_keys_are_preserved(isolated_home):
    cfg = ConfigManager(root=isolated_home)
    cfg.set("custom_future_key", {"a": 1})
    assert ConfigManager(root=isolated_home).get("custom_future_key") == {"a": 1}


def test_concurrent_writes_do_not_corrupt_file(isolated_home):
    cfg = ConfigManager(root=isolated_home)

    def worker(offset: int) -> None:
        for i in range(25):
            cfg.set(f"k{offset}", i)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = json.loads(cfg.config_path.read_text(encoding="utf-8"))
    for n in range(6):
        assert data[f"k{n}"] == 24


def test_typed_getters_are_defensive(isolated_home):
    cfg = ConfigManager(root=isolated_home)
    cfg.set("channels", "iki")
    assert cfg.get_int("channels", default=1) == 1
    cfg.set("cloud_ai_enabled", "true")
    assert cfg.get_bool("cloud_ai_enabled") is True
    cfg.set("theme", None)
    assert cfg.get_str("theme", "dark_crimson") == "dark_crimson"


def test_reset_to_defaults(isolated_home):
    cfg = ConfigManager(root=isolated_home)
    cfg.set("sample_rate", 96000)
    cfg.reset_to_defaults()
    assert ConfigManager(root=isolated_home).get("sample_rate") == 48000


def test_roundtrip_preserves_unicode(isolated_home):
    cfg = ConfigManager(root=isolated_home)
    cfg.set("input_device_name", "Mikrofon (Şş Ğü İı) 中文")
    raw = cfg.config_path.read_text(encoding="utf-8")
    assert "Ğ" in raw and "中文" in raw  # ensure_ascii=False beklenir
    assert ConfigManager(root=isolated_home).get("input_device_name").endswith("中文")
