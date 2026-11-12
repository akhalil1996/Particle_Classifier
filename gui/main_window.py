"""Main window: Home tab plus one tab per analysis."""
import os
import sys

from PySide6.QtCore import Qt, QSettings, QRectF
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                               QPushButton, QFileDialog, QMessageBox, QScrollArea, QGridLayout)

from cauliquant import __version__, APP_NAME, COPYRIGHT, GROUP, AUTHOR, SUPERVISION
from cauliquant.resources import DEFAULT_UNET, resource
from cauliquant.classifier import load_card
from .analysis_tab import AnalysisTab
from .developer_tab import DeveloperTab

ABOUT_TEXT = ("Developed by " + AUTHOR + ", " + SUPERVISION[0].lower() + SUPERVISION[1:] + ", at McGill University, " + GROUP + ", for quality control of titanium powder produced by "
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


class HelpTab(QScrollArea):
    """The user guide, one picture scaled to the width of the window."""

    def __init__(self):
        super().__init__()
        from PySide6.QtGui import QPixmap
        self.setWidgetResizable(False)
        self.setFrameShape(QFrame.NoFrame)
        self.setAlignment(Qt.AlignHCenter)
        self.setStyleSheet("QScrollArea { background: #F6F8FA; }")
        self.pm = QPixmap(resource("assets", "help.png"))
        self.lab = QLabel()
        self.lab.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.setWidget(self.lab)
        self._w = 0

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.pm.isNull():
            return
        w = min(self.viewport().width() - 4, self.pm.width())
        if w > 0 and w != self._w:
            self._w = w
            self.lab.setPixmap(self.pm.scaledToWidth(w, Qt.SmoothTransformation))
            self.lab.adjustSize()


class Banner(QWidget):
    """Photo banner that fills the width and keeps its left edge (where the title is)."""

    def __init__(self, path, height=190):
        super().__init__()
        from PySide6.QtGui import QPixmap
        self.pm = QPixmap(path)
        self.setFixedHeight(height)

    def paintEvent(self, e):
        from PySide6.QtGui import QPainter
        if self.pm.isNull():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        s = max(w / self.pm.width(), h / self.pm.height())
        sw, sh = w / s, h / s
        p.drawPixmap(self.rect(), self.pm, QRectF(0, (self.pm.height() - sh) / 2, sw, sh).toRect())
        p.end()


def numbered(n, title, body):
    f = QFrame()
    f.setObjectName("card")
    l = QVBoxLayout(f)
    l.setContentsMargins(14, 12, 14, 12)
    c = QLabel(str(n))
    c.setFixedSize(30, 30)
    c.setAlignment(Qt.AlignCenter)
    c.setStyleSheet("background:#1F4E79; color:white; border-radius:15px; font-weight:700;")
    t = QLabel(title)
    t.setStyleSheet("font-weight:700; font-size:11pt;")
    b = QLabel(body)
    b.setWordWrap(True)
    b.setObjectName("muted")
    b.setStyleSheet("font-size:9.5pt;")
    l.addWidget(c)
    l.addWidget(t)
    l.addWidget(b)
    l.addStretch(1)
    return f


def action_card(title, body, buttons):
    f = QFrame()
    f.setObjectName("card")
    l = QVBoxLayout(f)
    l.setContentsMargins(18, 14, 18, 16)
    head = QHBoxLayout()
    bar = QFrame()
    bar.setFixedSize(7, 22)
    bar.setStyleSheet("background:#1F4E79; border-radius:3px;")
    head.addWidget(bar)
    t = QLabel(title)
    t.setObjectName("cardTitle")
    head.addWidget(t)
    head.addStretch(1)
    l.addLayout(head)
    b = QLabel(body)
    b.setWordWrap(True)
    b.setStyleSheet("color:#33383F;")
    l.addWidget(b)
    l.addStretch(1)
    row = QHBoxLayout()
    for i, (text, fn) in enumerate(buttons):
        btn = QPushButton(text)
        if i == 0:
            btn.setObjectName("primary")
        btn.clicked.connect(fn)
        row.addWidget(btn)
    row.addStretch(1)
    l.addLayout(row)
    return f


class HomeTab(QScrollArea):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        w = QWidget()
        w.setObjectName("home")
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(Banner(resource("assets", "home_banner.jpg")))

        strip = QFrame()
        strip.setStyleSheet("QFrame#strip { background:#FFFFFF; border-bottom:3px solid #C00000; }")
        strip.setObjectName("strip")
        sl = QHBoxLayout(strip)
        sl.setContentsMargins(40, 12, 40, 10)
        sl.addWidget(logo_label("mcgill_logo.png", 46))
        sl.addStretch(1)
        sl.addWidget(logo_label("p2am2_logo.png", 48))
        outer.addWidget(strip)

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(40, 22, 40, 26)
        lay.setSpacing(14)
        intro = QLabel(
            "Titanium powder made by molten salt electrolysis holds dense particles, which suit powder bed fusion, and "
            "porous <i>cauliflower</i> agglomerates built from fine nodules or open branches, which flow and pack "
            "poorly. " + APP_NAME + " measures how much of a sample is cauliflower directly from SEM micrographs, "
            "particle by particle, and profiles particle shape on the five classes used for Microtrac dynamic image "
            "analysis.")
        intro.setWordWrap(True)
        intro.setObjectName("intro")
        lay.addWidget(intro)

        sec = QLabel("Start an analysis")
        sec.setObjectName("homeSection")
        lay.addWidget(sec)
        row = QHBoxLayout()
        row.setSpacing(14)
        row.addWidget(action_card(
            "Particle classification",
            "Every particle above 10 µm gets a probability of being cauliflower, from a trained network's reading of "
            "the surface together with outline shape and relief. The result is the cauliflower share of particle "
            "area, per image and pooled per sample.",
            [("Open image…", lambda: mw._go(mw.tab_class).open_image()),
             ("Open folder…", lambda: mw._go(mw.tab_class).open_folder())]), 1)
        row.addWidget(action_card(
            "Morphology profiling",
            "Shape classes C1 compact to C5 branched, from SEM micrographs or Microtrac files: particle reports, "
            "particle tables and laser diffraction size reports. Open a folder tree of batches to get a table and "
            "figures across all of them.",
            [("Open files…", lambda: mw._go(mw.tab_morph).open_image()),
             ("Open folder…", lambda: mw._go(mw.tab_morph).open_folder())]), 1)
        lay.addLayout(row)

        sec = QLabel("Getting started")
        sec.setObjectName("homeSection")
        lay.addWidget(sec)
        steps = QHBoxLayout()
        steps.setSpacing(12)
        for i, (t, b) in enumerate((
                ("Open files", "One micrograph, or a folder for a batch. The analysis starts by itself."),
                ("Check the overlay", "Space bar switches original and analysed. Scroll to zoom, drag to pan."),
                ("Read the result", "Results on the right; in a batch, click an image in the list to see it."),
                ("Export", "PDF report, figures as PNG, or CSV of every image or every particle.")), 1):
            steps.addWidget(numbered(i, t, b), 1)
        lay.addLayout(steps)

        mf = QFrame()
        mf.setObjectName("card")
        ml = QHBoxLayout(mf)
        ml.setContentsMargins(18, 12, 18, 12)
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        ml.addWidget(self.model_label, 1)
        b1 = QPushButton("Choose model file…")
        b1.clicked.connect(mw.choose_model)
        b2 = QPushButton("Use built-in model")
        b2.clicked.connect(mw.reset_model)
        b3 = QPushButton("Help (F1)")
        b3.clicked.connect(lambda: mw._go(mw.tab_help))
        for b in (b1, b2, b3):
            ml.addWidget(b)
        lay.addWidget(mf)
        lay.addStretch(1)
        outer.addWidget(body, 1)

        foot = QFrame()
        foot.setObjectName("homeFoot")
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(40, 18, 40, 18)
        left = QVBoxLayout()
        left.setSpacing(2)
        for text, style in (("DEVELOPED BY", "font-size:8.5pt; letter-spacing:1px; color:#BFD0E2;"),
                            (AUTHOR, "font-size:17pt; font-weight:700; color:white;"),
                            (SUPERVISION, "font-size:10.5pt; color:white;"),
                            (GROUP + " · McGill University, Montréal", "font-size:9.5pt; color:#DCE6F0;")):
            q = QLabel(text)
            q.setStyleSheet(style)
            left.addWidget(q)
        fl.addLayout(left, 1)
        right = QLabel(f"{COPYRIGHT}<br>{APP_NAME} {__version__} · Python {sys.version.split()[0]}")
        right.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        right.setStyleSheet("color:#DCE6F0; font-size:9pt;")
        fl.addWidget(right)
        outer.addWidget(foot)
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
        self.tab_help = HelpTab()
        self.tabs.addTab(self.tab_help, "Help")
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
        a = QAction("User guide", self)
        a.setShortcut("F1")
        a.triggered.connect(lambda: self._go(self.tab_help))
        h.addAction(a)
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
