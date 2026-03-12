# -*- mode: python ; coding: utf-8 -*-
"""
IdleBrain PyInstaller spec — onedir build for fast startup.
Output: dist/IdleBrainUI/IdleBrainUI.exe
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os, sys
from pathlib import Path

FRONTEND = Path(SPEC).parent           # d:\IdleBrain\project\frontend
PROJECT  = FRONTEND.parent             # d:\IdleBrain\project
SCRIPTS  = PROJECT / "scripts"
CONFIGS  = PROJECT / "configs"
ATLAS    = PROJECT / "annotation_25.nii.gz"

# ── Data files ───────────────────────────────────────────────────────────────
datas = [
    # Frontend HTML/CSS/JS — placed in root of _internal so Flask can serve them
    (str(FRONTEND / "index.html"),  "."),
    (str(FRONTEND / "styles.css"),  "."),
    (str(FRONTEND / "app.js"),      "."),
    # Pipeline scripts (needs __init__.py for package import to work)
    (str(SCRIPTS),                  "scripts"),
    (str(SCRIPTS / "__init__.py"),  "scripts"),
    # Configs
    (str(CONFIGS),                  "configs"),
]

# Bundle atlas only if it exists
if ATLAS.exists():
    datas.append((str(ATLAS), "."))

# nibabel needs its data files (templates etc.)
datas += collect_data_files("nibabel")

# ── Hidden imports ───────────────────────────────────────────────────────────
hidden_imports = [
    # Web framework
    "flask", "werkzeug", "werkzeug.serving", "werkzeug.routing",
    "jinja2", "click",
    # Numerics
    "numpy", "numpy.core", "numpy.lib", "numpy.random",
    "scipy", "scipy.ndimage", "scipy.spatial", "scipy.linalg",
    "scipy.sparse",
    # Image processing
    "skimage", "skimage.transform", "skimage.filters", "skimage.feature",
    "skimage.measure", "skimage.metrics", "skimage.registration",
    "skimage.segmentation",
    "tifffile", "imageio", "imageio.plugins",
    "PIL", "PIL.Image", "PIL.ImageDraw",
    # Atlas / NIfTI
    "nibabel", "nibabel.loadsave", "nibabel.nifti1",
    "nrrd",
    # Data
    "pandas", "pandas.io.formats.style",
    # Tray icon
    "pystray", "pystray._win32",
    # Tkinter
    "tkinter", "tkinter.messagebox",
    # Misc
    "json", "csv", "pathlib", "threading", "subprocess",
    "urllib.request",
]

# Pull in all skimage submodules (it uses lazy imports internally)
hidden_imports += collect_submodules("skimage")
hidden_imports += collect_submodules("scipy.ndimage")
hidden_imports += collect_submodules("imageio")

# ── Analysis ─────────────────────────────────────────────────────────────────
a = Analysis(
    [str(FRONTEND / "desktop_app.py")],
    pathex=[str(PROJECT), str(FRONTEND)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "IPython", "jupyter", "notebook",
              "PyQt5", "PyQt6", "wx", "gi",
              "torch", "torchvision", "torchaudio", "cellpose",
              "tensorflow", "keras", "cv2", "caffe2"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,       # onedir: binaries go to COLLECT
    name="IdleBrainUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,               # no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,                   # add .ico here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="IdleBrainUI",
)
