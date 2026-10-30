"""Cihaz yönetimi testleri. Cihaz yoksa testler atlanır (CI/güvenli ortam)."""

from __future__ import annotations

import pytest

from audio.devices import AudioDevice, DeviceManager


@pytest.fixture(scope="module")
def manager() -> DeviceManager:
    return DeviceManager()


def _skip_if_no_inputs(manager: DeviceManager) -> None:
    if not manager.input_devices():
        pytest.skip("Bu ortamda ses giriş cihazı bulunmuyor")


def test_enumeration_returns_devices(manager):
    if not manager.all_devices():
        pytest.skip("Bu ortamda ses cihazı bulunmuyor")
    assert len(manager.all_devices()) > 0
    for dev in manager.input_devices():
        assert dev.max_input_channels > 0
        assert dev.hostapi_name
        assert dev.name


def test_default_device_selection(manager):
    _skip_if_no_inputs(manager)
    dev = manager.default_input()
    assert dev is not None
    assert dev.input_capable


def test_validate_rejects_too_many_channels(manager):
    _skip_if_no_inputs(manager)
    dev = manager.input_devices()[0]
    ok, message = manager.validate_input(dev, 48000, dev.max_input_channels + 4)
    assert ok is False
    assert "kanal" in message


def test_validate_rejects_absurd_samplerate(manager):
    _skip_if_no_inputs(manager)
    dev = manager.input_devices()[0]
    ok, message = manager.validate_input(dev, 12345, 1)
    assert ok is False
    assert "12345" in message or "kabul etmedi" in message


def test_supported_sample_rates_are_verified(manager):
    _skip_if_no_inputs(manager)
    dev = manager.input_devices()[0]
    rates = manager.supported_sample_rates(dev, channels=1, kind="in")
    assert isinstance(rates, list)
    for rate in rates:
        assert rate in (44100, 48000, 88200, 96000, 192000)


def test_find_by_persistent_key(manager):
    _skip_if_no_inputs(manager)
    dev = manager.input_devices()[0]
    found = manager.find(dev.name, dev.hostapi_name, input_only=True)
    assert found is not None
    assert found.key == dev.key


def test_find_falls_back_to_index(manager):
    _skip_if_no_inputs(manager)
    dev = manager.input_devices()[0]
    found = manager.find("", None, index=dev.index, input_only=True)
    assert found is not None and found.index == dev.index


def test_describe_contains_device_names(manager):
    if not manager.all_devices():
        pytest.skip("Bu ortamda ses cihazı bulunmuyor")
    text = manager.describe()
    assert str(len(manager.all_devices())) in text


def test_decoding_of_broken_names_is_safe():
    from audio.devices import _decode_name

    assert _decode_name("") == "(adsız cihaz)"
    assert _decode_name("Mikrofon") == "Mikrofon"
    # Bozuk baytlar onarılamıyorsa bile çökme olmamalı
    assert isinstance(_decode_name("E\ufffdle\ufffdtiricisi"), str)
