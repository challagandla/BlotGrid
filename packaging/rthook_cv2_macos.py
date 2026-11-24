"""PyInstaller runtime hook (macOS .app only): let OpenCV find its extension.

In a macOS .app, PyInstaller keeps cv2's Python files in Contents/Resources/cv2
but the compiled cv2.abi3.so in Contents/Frameworks/cv2. OpenCV's loader
searches next to its (resolved) __init__.py, misses the binary, re-imports
the package and aborts with "recursion is detected during loading of cv2".

The bundled cv2/config-3.X.py (see blotgrid.spec) adds Frameworks/cv2 to the
loader's search paths; this flag makes the loader put those paths at the front
of sys.path for the duration of the import (it restores sys.path afterwards).
"""
import sys

if sys.platform == "darwin" and getattr(sys, "frozen", False):
    sys.OpenCV_REPLACE_SYS_PATH_0 = True
