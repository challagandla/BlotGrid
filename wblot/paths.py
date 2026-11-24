"""Cross-platform locations for BlotGrid's bundled resources and user files.

Source checkouts keep their historical layout: user files live in the
repository and the cache in ``<repo>/.cache``. Frozen desktop builds
(PyInstaller) are installed read-only, so their writable files go to the
operating system's per-user locations:

* Windows: ``%LOCALAPPDATA%\\BlotGrid``
* macOS:   ``~/Library/Application Support/BlotGrid`` (cache: ``~/Library/Caches/BlotGrid``)
* Linux:   ``$XDG_DATA_HOME/blotgrid`` (cache: ``$XDG_CACHE_HOME/blotgrid``)

``BLOTGRID_HOME`` overrides the user data directory on every platform.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

APP_NAME = "BlotGrid"


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def resource_root() -> Path:
    """Directory holding read-only bundled files (app.py, wblot/, examples/)."""
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def _platform_dir(kind: str) -> Path:
    home = Path.home()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        root = Path(base) / APP_NAME
        return root / "Cache" if kind == "cache" else root
    if sys.platform == "darwin":
        if kind == "cache":
            return home / "Library" / "Caches" / APP_NAME
        return home / "Library" / "Application Support" / APP_NAME
    env, fallback = (
        ("XDG_CACHE_HOME", home / ".cache") if kind == "cache"
        else ("XDG_DATA_HOME", home / ".local" / "share")
    )
    raw = os.environ.get(env, "")
    # The XDG spec requires absolute paths; ignore relative values.
    base = Path(raw) if raw and Path(raw).is_absolute() else fallback
    return base / APP_NAME.lower()


def user_data_dir() -> Path:
    """Writable per-user directory for models/, runs/, data/ and projects."""
    override = os.environ.get("BLOTGRID_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return _platform_dir("data")
    return resource_root()


def cache_dir() -> Path:
    """Writable cache directory (Matplotlib font cache and similar)."""
    if is_frozen() or os.environ.get("BLOTGRID_HOME", "").strip():
        return _platform_dir("cache")
    # Source checkouts keep the historical behaviour of run_blotgrid.sh.
    raw = os.environ.get("XDG_CACHE_HOME", "")
    if raw and Path(raw).is_absolute():
        return Path(raw) / "blotgrid"
    return resource_root() / ".cache" / "blotgrid"


def ensure_user_dirs() -> Path:
    """Create the user data layout and return the data directory."""
    root = user_data_dir()
    for sub in ("models", "runs", "data"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cache_dir().mkdir(parents=True, exist_ok=True)
    return root
