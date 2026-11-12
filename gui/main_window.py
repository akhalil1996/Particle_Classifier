"""Main window: Home tab plus one tab per analysis."""
import os
import sys

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                               QPushButton, QFileDialog, QMessageBox, QScrollArea, QGridLayout)

from cauliquant import __version__, APP_NAME, COPYRIGHT, GROUP
from cauliquant.resources import DEFAULT_UNET, resource
from cauliquant.classifier import load_card
from .analysis_tab import AnalysisTab
from .developer_tab import DeveloperTab

ABOUT_TEXT = ("Developed at McGill University, " + GROUP + ", for quality control of titanium powder produced by "
              "molten salt electrolysis.")


def logo_label(name, height):
    from PySide6.QtGui import QPixmap
    lab = QLabel()
    p = resource("assets", name)
    if os.path.exists(p):
        pm = QPixmap(p)
        lab.setPixmap(pm.scaledToHeight(height, Qt.SmoothTransformation))
    return lab


def card(title, body, accent="#1F4E79"):
    f = QFrame()
    f.setObjectName("card")
    l = QVBoxLayout(f)
    t = QLabel(title)
    t.setObjectName("cardTitle")
    t.setStyleSheet(f"color: {accent};")
    b = QLabel(body)
    b.setWordWrap(True)
    b.setTextFormat(Qt.RichText)
    b.setOpenExternalLinks(True)
    l.addWidget(t)
    l.addWidget(b)
    l.addStretch(1)
    return f


class HomeTab(QScrollArea):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        w = QWidget()
        w.setObjectName("home")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(36, 28, 36, 28)
        lay.setSpacing(14)

        head = QHBoxLayout()
        icon = QLabel()
        ico = resource("assets", "icon.png")
        if os.path.exists(ico):
            icon.setPixmap(QIcon(ico).pixmap(72, 72))
        head.addWidget(icon)
        tl = QVBoxLayout()
        name = QLabel(APP_NAME)
        name.setObjectName("appName")
        sub = QLabel("SEM quantification of porous cauliflower agglomerates in titanium powder")
        sub.setObjectName("appSub")
        ver = QLabel(f"Version {__version__}")
        ver.setObjectName("muted")
        tl.addWidget(name)
        tl.addWidget(sub)
        tl.addWidget(ver)
        head.addLayout(tl, 1)
        logos = QVBoxLayout()
        logos.setSpacing(6)
        logos.addWidget(logo_label("mcgill_logo.png", 44), 0, Qt.AlignRight)
        logos.addWidget(logo_label("p2am2_logo.png", 40), 0, Qt.AlignRight)
        head.addLayout(logos)
        lay.addLayout(head)

        intro = QLabel(
            "Titanium powder made by molten salt electrolysis contains two kinds of particle: dense particles, "
            "which suit powder bed fusion, and porous <i>cauliflower</i> agglomerates built from fine nodules or "
            "open branches, which flow and pack poorly. CAULIQUANT measures how much of a powder sample is "
            "cauliflower directly from SEM micrographs, particle by particle, and profiles particle shape on the "
            "same five classes used for Microtrac dynamic image analysis. It replaces sorting printed SEM images "
            "by hand with a repeatable count.".replace("CAULIQUANT", APP_NAME))
        intro.setWordWrap(True)
        intro.setObjectName("intro")
        lay.addWidget(intro)

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(card("Particle classification",
                            "Segments every particle above 10 µm and gives it a probability of being "
                            "<b>cauliflower</b>, from a trained U-Net's reading of the surface combined with outline "
                            "shape and surface relief. Reports the cauliflower share of particle area and volume "
                            "inside a counting frame. Overlays show every call; borderline particles are drawn in a "
                            "lighter shade."), 0, 0)
        grid.addWidget(card("Morphology profiling",
                            "Measures circularity, solidity, convexity, aspect ratio and extent on each SEM particle "
                            "outline and assigns it to one of five shape classes, <b>C1 compact</b> to "
                            "<b>C5 branched / agglomerate</b>, with the k-means model fitted on Microtrac SYNC data. "
                            "Also reads Microtrac exports: the particle report (PDF or page pictures), the particle "
                            "table (.xls, .xlsx or .csv) and the laser diffraction size report (PDF or .csv). Open a "
                            "folder and every sub-folder is read; the report adds figures across all batches."),
                       0, 1)
        grid.addWidget(card("Single image or whole folder",
                            "Open one micrograph, or a folder of fields for batch processing. Fields are grouped "
                            "into samples by file name (for example <code>384_x100_0031</code> belongs to sample "
                            "<code>384</code>) and pooled into a per-sample result."), 1, 0)
        grid.addWidget(card("Reports",
                            "Export a PDF report with the settings, a per-sample summary and, for every image, "
                            "the original and analysed views side by side. CSV export gives per-image results or "
                            "every particle with its measurements."), 1, 1)
        lay.addLayout(grid)

        how = card("How to use",
                   "<ol style='margin-left:-20px'>"
                   "<li>Open the <b>Particle classification</b> or <b>Morphology profiling</b> tab.</li>"
                   "<li>Click <b>Open image…</b> for one micrograph or <b>Open folder…</b> for a batch. "
                   "The analysis starts by itself.</li>"
                   "<li>Use <b>Original / Analysed</b> above the image (or the space bar) to compare. "
                   "Scroll to zoom, drag to pan.</li>"
                   "<li>Read the results on the right; in batch mode, click an image in the list to see it.</li>"
                   "<li>Click <b>Export PDF report</b> to save the report.</li></ol>")
        req = card("Images",
                   "Secondary-electron SEM images at <b>x100</b>, ideally 2560 × 1920 px "
                   "(about 0.5 µm per pixel), with particles spread as a monolayer. The pixel size is read from "
                   "the Hitachi SU3500 metadata (the .txt file saved next to each image, or the TIFF header). "
                   "For other images, untick <i>Pixel size from SEM metadata</i> and enter it. Images at another "
                   "pixel size are resampled to 0.5 µm/px before analysis. The data bar at the bottom is detected "
                   "and left out. Accepted formats: TIFF, PNG, JPEG, BMP. The morphology tab also takes Microtrac "
                   "PDF reports, pictures of report pages, particle tables (.xls, .xlsx, .csv) and size exports (.csv), "
                   "and whole folder trees of them.")
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(how, 1)
        row.addWidget(req, 1)
        lay.addLayout(row)

        trust = card("How far to trust a result",
                     "<b>One image is a rough number.</b> Fields of the same stub differ by about 10 points, so a "
                     "sample needs several fields. In batch mode each sample gets a 95 % interval over its fields "
                     "and an estimate of how many fields a ±5 point interval would take."
                     "<br><b>Some particles are borderline.</b> About a third of particles sit between the two "
                     "classes. They are drawn in a lighter shade and their share of the area is reported, so you can "
                     "see how much of a result rests on them."
                     "<br><b>Warnings matter.</b> An image is flagged when its SEM settings differ from the ones the "
                     "model was validated on, when the field is crowded, soft or saturated, or when it holds few "
                     "particles. The model's measured error on batches it never saw is printed in every report.")
        lay.addWidget(trust)

        mf = QFrame()
        mf.setObjectName("card")
        ml = QHBoxLayout(mf)
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        ml.addWidget(self.model_label, 1)
        b1 = QPushButton("Choose model file…")
        b1.clicked.connect(mw.choose_model)
        b2 = QPushButton("Use built-in model")
        b2.clicked.connect(mw.reset_model)
        ml.addWidget(b1)
        ml.addWidget(b2)
        lay.addWidget(mf)

        about = QLabel(f"{ABOUT_TEXT}<br><b>{COPYRIGHT}</b><br>{APP_NAME} {__version__} · Python {sys.version.split()[0]}")
        about.setObjectName("muted")
        about.setWordWrap(True)
        lay.addWidget(about)
        lay.addStretch(1)
        self.setWidget(w)
        self.update_model()

    def update_model(self):
        p = self.mw.model_path()
        builtin = os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(DEFAULT_UNET))
        state = "" if os.path.exists(p) else " <span style='color:#B42318'>(file not found)</span>"
        c = load_card(p) if os.path.exists(p) else {}
        v = c.get("validated", {})
        if c.get("classifier"):
            rv = int(c.get("revision", 0))
            name = c.get("name", "") + (f", classifier revision {rv}" if rv else "")
            if rv and c.get("refined") and rv > int(load_card(p, refined=False).get("revision", 0)):
                name += " (retrained on this PC)"
            extra = (f"<br>{name}. Validated for x{v.get('magnification', 100):.0f}, {v.get('kv', 15):.0f} kV, "
                     f"spot intensity {v.get('spot', 50):.0f}, scan {v.get('scan', '')}, {v.get('um_px', 0.496):.3f} µm/px.")
        else:
            extra = ("<br><span style='color:#B54708'>No model card next to this file: the raw network score is used "
                     "and results are not validated.</span>") if os.path.exists(p) else ""
        self.model_label.setText(f"<b>Classification model:</b> {'built-in' if builtin else os.path.basename(p)}"
                                 f"{state}{extra}<br><span style='color:#656D76'>{p}</span>")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.qs = QSettings()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        ico = resource("assets", "icon.png")
        if os.path.exists(ico):
            self.setWindowIcon(QIcon(ico))
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.home = HomeTab(self)
        self.tab_class = AnalysisTab("classification", self)
        self.tab_morph = AnalysisTab("morphology", self)
        self.tabs.addTab(self.home, "Home")
        self.tabs.addTab(self.tab_class, "Particle classification")
        self.tabs.addTab(self.tab_morph, "Morphology profiling")
        self.tab_dev = DeveloperTab(self)
        self.tabs.addTab(self.tab_dev, "Developer")
        self.tabs.currentChanged.connect(lambda i: self.home.update_model() if self.tabs.widget(i) is self.home else None)
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage(COPYRIGHT)
        self.resize(1440, 900)

        m = self.menuBar().addMenu("&File")
        for text, fn in (("Open image for classification…", lambda: self._go(self.tab_class).open_image()),
                         ("Open folder for classification…", lambda: self._go(self.tab_class).open_folder()),
                         ("Open files for morphology…", lambda: self._go(self.tab_morph).open_image()),
                         ("Open folder for morphology…", lambda: self._go(self.tab_morph).open_folder())):
            a = QAction(text, self)
            a.triggered.connect(fn)
            m.addAction(a)
        m.addSeparator()
        a = QAction("Choose classification model…", self)
        a.triggered.connect(self.choose_model)
        m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self)
        a.triggered.connect(self.close)
        m.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction(f"About {APP_NAME}", self)
        a.triggered.connect(self.about)
        h.addAction(a)

    def _go(self, tab):
        self.tabs.setCurrentWidget(tab)
        return tab

    def model_path(self):
        p = self.qs.value("model_path", "")
        return p if p else DEFAULT_UNET

    def choose_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Choose classification model", os.path.dirname(self.model_path()),
                                           "ONNX model (*.onnx)")
        if f:
            try:
                from cauliquant.infer import UNetONNX
                UNetONNX(f)
            except Exception as e:
                QMessageBox.critical(self, "Not a usable model", f"{type(e).__name__}: {e}")
                return
            self.qs.setValue("model_path", f)
            self.home.update_model()

    def reset_model(self):
        self.qs.remove("model_path")
        self.home.update_model()

    def about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<h3>{APP_NAME} {__version__}</h3><p>SEM quantification of porous cauliflower "
                          f"agglomerates and particle shape classes in titanium powder.</p><p>{ABOUT_TEXT}</p>"
                          f"<p><b>{COPYRIGHT}</b></p>")

    def closeEvent(self, e):
        for t in (self.tab_class, self.tab_morph):
            if t.worker:
                t.worker.cancel()
            if t.thread:
                t.thread.wait(30000)
        if self.tab_dev.thread:
            self.tab_dev.thread.wait(30000)
        super().closeEvent(e)
