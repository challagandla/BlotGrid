#!/usr/bin/env python3
"""BlotGrid desktop launcher for Windows, macOS and Linux.

Starts the Streamlit app as a private, local-only server and shows it in a
native window:

* The server listens on 127.0.0.1 only, on a random free port.
* A random per-launch token is required before the app script runs
  (see wblot/desktop_guard.py). XSRF protection stays on; usage statistics,
  file watching and static file serving are off.
* The window is a native webview (pywebview: WebView2 on Windows, WebKit on
  macOS, GTK/Qt WebKit on Linux). When no webview backend is available it
  falls back to a Chromium/Edge "app" window, then to the default browser.
* Closing the window stops the server. If the launcher dies, the server notices
  its stdin closing and exits too, so no orphaned server is left behind.
* User files (models/, runs/, data/) live in the per-user data directory, not
  inside the read-only app bundle (see wblot/paths.py).

Usage:
    python blotgrid_desktop.py            # native window
    python blotgrid_desktop.py --browser  # force a browser window
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# Make `wblot` importable when run as a plain script from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wblot import paths  # noqa: E402
from wblot.desktop_guard import ENV_VAR as TOKEN_ENV  # noqa: E402

HOST = "127.0.0.1"
READY_TIMEOUT_S = 120.0
WINDOW_SIZE = (1440, 920)
MIN_WINDOW_SIZE = (960, 640)

# Settings that are always enforced, whatever any config.toml says.
SECURITY_FLAGS = {
    "server.address": HOST,
    "server.headless": True,
    "server.runOnSave": False,
    "server.fileWatcherType": "none",
    "server.enableXsrfProtection": True,
    "server.enableCORS": True,
    "server.enableStaticServing": False,
    "browser.gatherUsageStats": False,
    "browser.serverAddress": HOST,
    # PyInstaller bundles are not in site-packages, which makes Streamlit
    # guess "development mode" and look for a Node dev server.
    "global.developmentMode": False,
}

LOADING_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>
  html,body{height:100%;margin:0;background:#010102;color:#f7f8f8;
    font-family:Inter,"SF Pro Text",-apple-system,"Segoe UI",system-ui,sans-serif}
  .wrap{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px}
  .mark{font-size:22px;font-weight:600;letter-spacing:-.4px}
  .sub{color:#8a8f98;font-size:13px}
  .bar{width:180px;height:2px;background:#23252a;border-radius:2px;overflow:hidden}
  .bar i{display:block;width:40%;height:100%;background:#5e6ad2;animation:s 1.1s ease-in-out infinite}
  @keyframes s{0%{transform:translateX(-100%)}100%{transform:translateX(250%)}}
  @media (prefers-reduced-motion:reduce){.bar i{animation:none;width:100%}}
</style></head><body><div class="wrap">
<div class="mark">&#9638; BlotGrid</div><div class="bar"><i></i></div>
<div class="sub">Starting the local analysis engine&hellip;</div></div></body></html>"""


def _error_html(message: str, log_path: Path) -> str:
    from html import escape
    return (
        LOADING_HTML.split("<body>")[0]
        + "<body><div class='wrap'><div class='mark'>BlotGrid could not start</div>"
        + f"<div class='sub'>{escape(message)}</div>"
        + f"<div class='sub'>Log: {escape(str(log_path))}</div></div></body></html>"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import tomllib  # Python 3.11+
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except ModuleNotFoundError:
        import toml  # Streamlit dependency on Python 3.10
        return toml.load(str(path))


def _flatten(table: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in table.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


def streamlit_flag_options(config_path: Path, port: int) -> dict[str, object]:
    """Bundled config.toml values plus enforced security settings.

    Keys use Streamlit's flag spelling (dots become underscores), which is what
    ``streamlit.web.bootstrap`` expects.
    """
    options = _flatten(_load_toml(config_path))
    options.update(SECURITY_FLAGS)
    options["server.port"] = int(port)
    return {name.replace(".", "_"): value for name, value in options.items()}


# ---------------------------------------------------------------------------
# Server (child process)
# ---------------------------------------------------------------------------

def _exit_when_parent_closes_stdin() -> None:
    """Exit as soon as the launcher's end of our stdin pipe closes."""
    def watch() -> None:
        try:
            while sys.stdin.buffer.read(1024):
                pass
        except Exception:
            pass
        os._exit(0)

    if sys.stdin is not None:
        threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def serve(port: int, watch_parent: bool) -> None:
    root = paths.resource_root()
    data_dir = paths.ensure_user_dirs()
    os.environ.setdefault("MPLCONFIGDIR", str(paths.cache_dir() / "matplotlib"))
    # Relative defaults such as models/blotgrid_detector.pt resolve here.
    os.chdir(data_dir)
    if watch_parent:
        _exit_when_parent_closes_stdin()

    from streamlit.web import bootstrap

    flags = streamlit_flag_options(root / ".streamlit" / "config.toml", port)
    bootstrap.load_config_options(flag_options=flags)
    bootstrap.run(str(root / "app.py"), False, [], flags)


# ---------------------------------------------------------------------------
# Launcher (parent process)
# ---------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def _server_command(port: int) -> list[str]:
    if paths.is_frozen():
        return [sys.executable, "--serve", "--port", str(port), "--watch-parent"]
    return [sys.executable, str(Path(__file__).resolve()), "--serve",
            "--port", str(port), "--watch-parent"]


def log_path() -> Path:
    directory = paths.cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "blotgrid-server.log"


def start_server(token: str) -> tuple[subprocess.Popen, int]:
    env = os.environ.copy()
    env[TOKEN_ENV] = token
    env["PYTHONUNBUFFERED"] = "1"
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    log = log_path().open("wb")
    port = free_port()
    proc = subprocess.Popen(
        _server_command(port),
        stdin=subprocess.PIPE,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=creationflags,
    )
    log.close()  # the child holds its own handle
    return proc, port


def wait_until_ready(port: int, proc: subprocess.Popen,
                     timeout: float = READY_TIMEOUT_S) -> bool:
    # Bypass any system proxy: the server is on loopback only.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://{HOST}:{port}/_stcore/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with opener.open(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.15)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=3)
    except Exception:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _run_webview(url_for_ready, proc: subprocess.Popen) -> bool:
    """Show a native window. Returns False if no webview backend works."""
    try:
        import webview
    except Exception:
        return False
    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    webview.settings["ALLOW_FILE_URLS"] = False
    window = webview.create_window(
        "BlotGrid",
        html=LOADING_HTML,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=MIN_WINDOW_SIZE,
        background_color="#010102",
        text_select=True,
        zoomable=True,
    )

    def load() -> None:
        url = url_for_ready()
        if url:
            window.load_url(url)
        else:
            window.load_html(_error_html("The local server did not start.", log_path()))

    storage = paths.cache_dir() / "webview"
    storage.mkdir(parents=True, exist_ok=True)
    try:
        webview.start(load, private_mode=False, storage_path=str(storage))
    except Exception:
        return False
    return True


def _chromium_app_browser() -> str | None:
    candidates: list[str] = []
    if sys.platform.startswith("win"):
        for env in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
            base = os.environ.get(env)
            if base:
                candidates += [
                    str(Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
                    str(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"),
                ]
    elif sys.platform == "darwin":
        candidates += [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium",
                     "chromium-browser", "microsoft-edge", "brave-browser"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
    return next((c for c in candidates if Path(c).is_file()), None)


def _run_browser(url: str, proc: subprocess.Popen) -> None:
    browser = _chromium_app_browser()
    if browser:
        profile = paths.cache_dir() / "browser-profile"
        profile.mkdir(parents=True, exist_ok=True)
        # A dedicated profile makes this a separate browser process, so the
        # app quits when the window closes.
        window = subprocess.Popen([
            browser, f"--app={url}", f"--user-data-dir={profile}",
            "--no-first-run", "--no-default-browser-check",
            f"--window-size={WINDOW_SIZE[0]},{WINDOW_SIZE[1]}",
        ])
        while window.poll() is None and proc.poll() is None:
            time.sleep(0.5)
        return
    import webbrowser
    webbrowser.open(url)
    print("BlotGrid is running in your browser. "
          "Press Ctrl+C here to quit.", flush=True)
    try:
        proc.wait()
    except KeyboardInterrupt:
        pass


def launch(force_browser: bool = False) -> int:
    token = secrets.token_urlsafe(32)
    proc, port = start_server(token)
    url = f"http://{HOST}:{port}/?token={token}"

    def _terminate(*_args) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _terminate)
    try:
        ready_state: dict[str, bool] = {}

        def url_when_ready() -> str | None:
            if "ok" not in ready_state:
                ready_state["ok"] = wait_until_ready(port, proc)
            return url if ready_state["ok"] else None

        if not force_browser and _run_webview(url_when_ready, proc):
            return 0
        if url_when_ready() is None:
            print(f"BlotGrid server failed to start; see {log_path()}", file=sys.stderr)
            return 1
        _run_browser(url, proc)
        return 0
    finally:
        stop_server(proc)


# Modules app.py needs at runtime. PyInstaller cannot see app.py's imports
# (Streamlit executes it as a script), so --self-test proves they are bundled.
SELF_TEST_MODULES = (
    "streamlit", "numpy", "pandas", "scipy.ndimage", "scipy.signal",
    "scipy.optimize", "scipy.stats", "skimage.restoration", "PIL.Image",
    "matplotlib.backends.backend_agg", "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_pdf", "onnxruntime", "rapidocr", "cv2",
    "tifffile",
)


def self_test() -> int:
    """Import every runtime module and check bundled files; 0 when healthy."""
    import importlib
    import pkgutil

    import wblot

    failures: list[str] = []
    wblot_modules = [f"wblot.{m.name}" for m in pkgutil.iter_modules(wblot.__path__)]
    # Torch-backed modules import torch lazily, so all wblot modules must load.
    for name in (*SELF_TEST_MODULES, *wblot_modules):
        try:
            importlib.import_module(name)
        except Exception as exc:  # report every failure, not just the first
            failures.append(f"import {name}: {exc!r}")
    root = paths.resource_root()
    for rel in ("app.py", ".streamlit/config.toml",
                "wblot/roi_editor_frontend/index.html",
                "wblot/roi_editor_frontend/roi_editor.js"):
        if not (root / rel).is_file():
            failures.append(f"missing bundled file: {rel}")
    if len(wblot_modules) < 10:
        failures.append(f"only {len(wblot_modules)} wblot modules found")
    for line in failures:
        print("FAIL", line)
    print("self-test:", "FAILED" if failures else
          f"OK ({len(SELF_TEST_MODULES) + len(wblot_modules)} modules)")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="BlotGrid", description=__doc__.split("\n")[0])
    parser.add_argument("--browser", action="store_true",
                        help="open in a browser window instead of a native window")
    parser.add_argument("--self-test", action="store_true",
                        help="check the installation and exit")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--watch-parent", action="store_true", help=argparse.SUPPRESS)
    # macOS passes -psn_* when launched from Finder on old systems.
    args, _unknown = parser.parse_known_args(argv)
    if args.self_test:
        return self_test()
    if args.serve:
        serve(args.port or free_port(), args.watch_parent)
        return 0
    return launch(force_browser=args.browser)


if __name__ == "__main__":
    sys.exit(main())
