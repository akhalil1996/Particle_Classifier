# PyInstaller spec: builds dist/CauliQuant/ (one folder, windowed).
#   pyinstaller packaging/cauliquant.spec --noconfirm
import os
from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICON = os.path.join(ROOT, "assets", "icon.ico")

a = Analysis(
    [os.path.join(ROOT, "run_app.py")],
    pathex=[ROOT],
    datas=[(os.path.join(ROOT, "models"), "models"), (os.path.join(ROOT, "assets"), "assets")],
    hiddenimports=collect_submodules("skimage.filters") + collect_submodules("skimage.morphology")
    + collect_submodules("skimage.segmentation") + collect_submodules("skimage.measure")
    + collect_submodules("skimage.transform") + collect_submodules("reportlab.graphics.barcode")
    + ["pypdfium2", "pypdfium2_raw"],
    excludes=["torch", "matplotlib", "pandas", "tkinter", "IPython", "jupyter", "pytest",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
              "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="CauliQuant", icon=ICON,
          console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="CauliQuant", upx=False)
