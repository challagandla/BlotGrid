"""Tests for the cross-platform desktop launcher, paths and access guard."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

import blotgrid_desktop as desktop
from wblot import desktop_guard, paths, roi_editor


# --- paths -----------------------------------------------------------------

@pytest.mark.parametrize(
    ("platform", "expected_parts"),
    [
        ("win32", ("Local", "BlotGrid")),
        ("darwin", ("Library", "Application Support", "BlotGrid")),
        ("linux", (".local", "share", "blotgrid")),
    ],
)
def test_frozen_user_data_dir_is_per_platform(monkeypatch, tmp_path, platform, expected_parts):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    for var in ("BLOTGRID_HOME", "LOCALAPPDATA", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    result = paths.user_data_dir()
    assert result.parts[-len(expected_parts):] == expected_parts
    assert str(result).startswith(str(tmp_path))


def test_relative_xdg_values_are_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")
    assert paths._platform_dir("cache") == tmp_path / ".cache" / "blotgrid"


def test_blotgrid_home_override(monkeypatch, tmp_path):
    monkeypatch.setenv("BLOTGRID_HOME", str(tmp_path / "home"))
    assert paths.user_data_dir() == (tmp_path / "home").resolve()
    root = paths.ensure_user_dirs()
    assert {p.name for p in root.iterdir()} >= {"models", "runs", "data"}


def test_checkout_keeps_repository_layout(monkeypatch):
    monkeypatch.delenv("BLOTGRID_HOME", raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    assert paths.user_data_dir() == Path(__file__).resolve().parent


# --- access guard ----------------------------------------------------------

def test_guard_is_noop_without_launcher_token():
    assert desktop_guard.is_authorized({}, {}, environ={})


def test_guard_requires_matching_token_and_remembers_session():
    env = {desktop_guard.ENV_VAR: "s3cret-token"}
    session: dict = {}
    assert not desktop_guard.is_authorized(session, {}, environ=env)
    assert not desktop_guard.is_authorized(session, {"token": "wrong"}, environ=env)
    assert not desktop_guard.is_authorized(session, {"token": ""}, environ=env)
    assert desktop_guard.is_authorized(session, {"token": "s3cret-token"}, environ=env)
    # Later reruns no longer carry the token in the URL.
    assert desktop_guard.is_authorized(session, {}, environ=env)


# --- launcher --------------------------------------------------------------

def test_security_flags_override_config(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        '[server]\naddress = "0.0.0.0"\nenableXsrfProtection = false\n'
        '[browser]\ngatherUsageStats = true\n'
        '[theme.dark]\nprimaryColor = "#5e6ad2"\n',
        encoding="utf-8",
    )
    flags = desktop.streamlit_flag_options(config, 4321)
    assert flags["server_address"] == "127.0.0.1"
    assert flags["server_enableXsrfProtection"] is True
    assert flags["browser_gatherUsageStats"] is False
    assert flags["server_enableStaticServing"] is False
    assert flags["global_developmentMode"] is False
    assert flags["server_port"] == 4321
    assert flags["theme_dark_primaryColor"] == "#5e6ad2"


def test_bundled_config_is_local_only():
    flags = desktop.streamlit_flag_options(
        Path(__file__).with_name(".streamlit") / "config.toml", 1)
    assert flags["server_address"] == "127.0.0.1"
    assert flags["theme_dark_backgroundColor"] == "#010102"


def test_free_port_is_loopback_bindable():
    port = desktop.free_port()
    assert 0 < port < 65536


def test_server_command_targets_this_launcher(monkeypatch):
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    cmd = desktop._server_command(1234)
    assert cmd[1].endswith("blotgrid_desktop.py")
    assert cmd[-4:] == ["--serve", "--port", "1234", "--watch-parent"]


# --- speed: ROI image encoding memo ----------------------------------------

def test_roi_image_encoding_is_memoized(monkeypatch):
    roi_editor._encode_png_data_url.cache_clear()
    calls = []
    original = roi_editor.Image.fromarray

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(roi_editor.Image, "fromarray", counting)
    image = (np.arange(64 * 48) % 251).astype(np.uint8).reshape(48, 64)
    first = roi_editor._image_data_url(image)
    second = roi_editor._image_data_url(image.copy())
    assert first == second and first.startswith("data:image/png;base64,")
    assert len(calls) == 1
    changed = image.copy()
    changed[0, 0] ^= 1
    assert roi_editor._image_data_url(changed) != first
    assert len(calls) == 2


def test_self_test_passes_in_checkout(capsys):
    assert desktop.self_test() == 0
    assert "self-test: OK" in capsys.readouterr().out


def test_launch_starts_guarded_server_and_stops_it_on_close(monkeypatch, tmp_path):
    """Full parent flow with the window stubbed out (no display needed)."""
    monkeypatch.setenv("BLOTGRID_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(desktop, "log_path", lambda: tmp_path / "server.log")
    started = {}
    real_start = desktop.start_server

    def recording_start(token):
        proc, port = real_start(token)
        started.update(proc=proc, port=port, token=token)
        return proc, port

    seen = {}

    def fake_window(url_for_ready, proc):
        seen["url"] = url_for_ready()
        return True  # the "window" closes immediately

    monkeypatch.setattr(desktop, "start_server", recording_start)
    monkeypatch.setattr(desktop, "_run_webview", fake_window)
    assert desktop.launch() == 0
    assert seen["url"] == (
        f"http://127.0.0.1:{started['port']}/?token={started['token']}")
    assert len(started["token"]) >= 32
    # Closing the window must not leave the server running.
    assert started["proc"].poll() is not None
