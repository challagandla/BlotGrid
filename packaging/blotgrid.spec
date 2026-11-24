# PyInstaller spec for the BlotGrid desktop app (Windows, macOS, Linux).
#
#   pip install -r requirements-build.txt
#   pyinstaller packaging/blotgrid.spec --noconfirm
#
# Output: dist/BlotGrid/ (Windows/Linux folder) or dist/BlotGrid.app (macOS).
# onedir, not onefile: onefile unpacks ~300 MB to a temp folder on every launch,
# which is slow and is what antivirus tools flag most often.
#
# The optional ML stack (PyTorch, SAM 2) is deliberately excluded to keep the
# bundle small; classical detection, GelGenie ONNX and OCR are included.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

ROOT = Path(SPECPATH).parent

# rapidocr pulls in opencv-python, whose wheel carries Qt and FFmpeg (~200 MB)
# that BlotGrid never uses. Build with the headless wheel instead:
#   pip uninstall -y opencv-python && pip install opencv-python-headless==<same>
from importlib import metadata as _metadata
try:
    _metadata.version("opencv-python")
except _metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit(
        "opencv-python (GUI build) is installed; replace it with "
        "opencv-python-headless of the same version before building."
    )
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")

datas = [
    (str(ROOT / "app.py"), "."),
    (str(ROOT / ".streamlit" / "config.toml"), ".streamlit"),
    (str(ROOT / "wblot" / "roi_editor_frontend"), "wblot/roi_editor_frontend"),
    (str(ROOT / "examples"), "examples"),
    (str(ROOT / "sample_blot.tif"), "."),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "THIRD_PARTY_DATA.md"), "."),
    (str(ROOT / "THIRD_PARTY_MODELS.md"), "."),
]
datas += collect_data_files("streamlit")
datas += collect_data_files("rapidocr")
datas += copy_metadata("streamlit")
for dist in ("rapidocr", "onnxruntime", "pandas", "numpy", "pillow", "tifffile"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

# app.py is executed by Streamlit at runtime, so PyInstaller cannot see its
# imports: declare them.
# collect_submodules("wblot") runs in a subprocess without the repo on
# sys.path, so list the package's modules from the source tree.
_wblot_modules = ["wblot"] + sorted(
    f"wblot.{p.stem}" for p in (ROOT / "wblot").glob("*.py") if p.stem != "__init__"
)
hiddenimports = (
    _wblot_modules
    + collect_submodules("streamlit")
    + collect_submodules("rapidocr")
    + [
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_svg",
        "matplotlib.backends.backend_pdf",
        "matplotlib.backends.backend_ps",
        "onnxruntime",
        "openpyxl",
        "tifffile",
        "skimage.restoration",
        "scipy.optimize",
        "scipy.signal",
        "scipy.stats",
    ]
)

excludes = [
    "torch", "torchvision", "torchaudio", "sam2", "tensorflow", "jax",
    "IPython", "jupyter", "notebook", "pytest", "tkinter",
    # Arrow Flight (RPC) is unused; Streamlit only needs Arrow IPC.
    "pyarrow.flight", "pyarrow._flight",
]
if IS_LINUX:
    # Linux uses a browser app window; do not drag GTK/Qt into the bundle.
    excludes += ["webview", "gi", "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy"]

a = Analysis(
    [str(ROOT / "blotgrid_desktop.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
    optimize=1,
)
# Drop shared libraries of excluded components that hooks still collect.
_DROP_BINARIES = ("libarrow_flight", "_flight.")
a.binaries = [b for b in a.binaries if not any(t in Path(b[0]).name for t in _DROP_BINARIES)]
pyz = PYZ(a.pure)

icon = str(ROOT / "packaging" / "icon.png")
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BlotGrid",
    console=False,
    disable_windowed_traceback=False,
    upx=False,  # UPX-packed binaries trigger antivirus false positives.
    icon=icon,
    argv_emulation=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BlotGrid",
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name="BlotGrid.app",
        icon=icon,
        bundle_identifier="io.github.challagandla.blotgrid",
        info_plist={
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
