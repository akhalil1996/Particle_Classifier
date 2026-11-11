"""One tab per analysis: open an image or a folder, view original vs analysed, results, export."""
import math
import os

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QSettings, QRectF
from PySide6.QtGui import QColor, QPainter, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QListWidget,
                               QListWidgetItem, QSplitter, QLabel, QProgressBar, QGroupBox, QFormLayout,
                               QDoubleSpinBox, QComboBox, QCheckBox, QTableWidget, QTableWidgetItem,
                               QHeaderView, QButtonGroup, QMessageBox, QToolButton, QMenu, QScrollArea,
                               QFrame, QSizePolicy, QAbstractItemView)

from cauliquant.analysis import (Settings, list_images, IMAGE_EXT, MORPH_EXT, CAULI_RGB, DENSE_RGB, CAULI_SOFT_RGB,
                                 DENSE_SOFT_RGB, SHAPE_RGB,
                                 SHAPE_NAMES, FRAME_RGB, summarise_classification, summarise_morphology)
from cauliquant.segment import data_bar_row
from cauliquant.report import build_report, export_csv, export_summary_csv
from .viewer import ImageViewer
from . import worker as wk

FRAME_RULES = [("half", "50% rule"), ("edge", "Whole particle inside"), ("centroid", "Centre inside")]
FRAME_RULES_CLASS = FRAME_RULES + [("clip", "Area inside frame"), ("ml", "Miles-Lantuejoul")]


def fmt(v, d=1, unit=""):
    try:
        if v is None or math.isnan(float(v)):
            return "n/a"
        return f"{float(v):.{d}f}{unit}"
    except (TypeError, ValueError):
        return str(v)


class ShareBars(QWidget):
    """Horizontal bars, one per class, labelled with their share."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self.setMinimumHeight(40)

    def set_items(self, items):            # [(label, value %, (r, g, b))]
        self.items = items
        self.setMinimumHeight(24 * max(1, len(items)) + 6)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        lab_w = int(w * 0.47)
        bar_w = w - lab_w - 52
        f = p.font()
        f.setPointSizeF(8.5)
        p.setFont(f)
        for k, (label, v, rgb) in enumerate(self.items):
            y = 4 + 24 * k
            p.setPen(QColor("#33383F"))
            p.drawText(QRectF(0, y, lab_w - 6, 18), Qt.AlignVCenter | Qt.AlignLeft, label)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#E6E9EE"))
            p.drawRoundedRect(QRectF(lab_w, y + 3, bar_w, 12), 3, 3)
            if v is not None and not math.isnan(v):
                p.setBrush(QColor(*rgb))
                p.drawRoundedRect(QRectF(lab_w, y + 3, max(2.0, bar_w * min(v, 100) / 100), 12), 3, 3)
            p.setPen(QColor("#33383F"))
            p.drawText(QRectF(lab_w + bar_w + 4, y, 48, 18), Qt.AlignVCenter | Qt.AlignLeft, fmt(v, 1, " %"))
        p.end()


def legend(items):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(4, 2, 4, 2)
    for rgb, text in items:
        sw = QLabel()
        sw.setFixedSize(12, 12)
        sw.setStyleSheet(f"background: rgb{tuple(rgb)}; border-radius: 2px;")
        lay.addWidget(sw)
        t = QLabel(text)
        t.setObjectName("legendText")
        lay.addWidget(t)
        lay.addSpacing(10)
    lay.addStretch(1)
    return w


def make_table(cols):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.NoSelection)
    t.setAlternatingRowColors(True)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(False)
    t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    t.horizontalHeader().setMinimumSectionSize(40)
    t.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
    t.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return t


def fill_table(t, rows):
    t.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            it = QTableWidgetItem(str(v))
            if j > 0:
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            t.setItem(i, j, it)
    t.setFixedHeight(t.horizontalHeader().height() + sum(t.rowHeight(i) for i in range(len(rows))) + 4)


class AnalysisTab(QWidget):
    def __init__(self, kind, main_window):
        super().__init__()
        self.kind = kind                      # "classification" or "morphology"
        self.mw = main_window
        self.paths, self.results, self.errors = [], {}, {}
        self.current = None
        self.thread = self.worker = None
        self.qs = QSettings()
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        bar = QHBoxLayout()
        self.btn_image = QPushButton("Open image…" if self.kind == "classification" else "Open files…")
        self.btn_folder = QPushButton("Open folder…")
        self.btn_run = QPushButton("Analyse")
        self.btn_run.setObjectName("primary")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(360)
        self.status = QLabel("")
        self.status.setObjectName("muted")
        self.btn_pdf = QToolButton()
        self.btn_pdf.setText("Export PDF report")
        self.btn_pdf.setPopupMode(QToolButton.MenuButtonPopup)
        m = QMenu(self)
        m.addAction("Report for all analysed images", lambda: self.export_pdf(all_images=True))
        m.addAction("Report for the image shown", lambda: self.export_pdf(all_images=False))
        self.btn_pdf.setMenu(m)
        self.btn_pdf.clicked.connect(lambda: self.export_pdf(all_images=True))
        self.btn_csv = QToolButton()
        self.btn_csv.setText("Export CSV")
        self.btn_csv.setPopupMode(QToolButton.InstantPopup)
        m2 = QMenu(self)
        m2.addAction("Summary per image", lambda: self.export_table(per_particle=False))
        m2.addAction("Every particle", lambda: self.export_table(per_particle=True))
        self.btn_csv.setMenu(m2)
        for b in (self.btn_image, self.btn_folder, self.btn_run, self.btn_cancel):
            bar.addWidget(b)
        bar.addSpacing(8)
        bar.addWidget(self.progress)
        bar.addWidget(self.status, 1)
        bar.addWidget(self.btn_pdf)
        bar.addWidget(self.btn_csv)
        root.addLayout(bar)

        root.addWidget(self._settings_box())

        split = QSplitter(Qt.Horizontal)
        # left: list of images
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        cap = QLabel("Images" if self.kind == "classification" else "Files")
        cap.setObjectName("section")
        ll.addWidget(cap)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.show_index)
        ll.addWidget(self.list, 1)
        split.addWidget(left)
        # centre: toggle + viewer + legend
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.title = QLabel("")
        self.title.setObjectName("section")
        top.addWidget(self.title, 1)
        self.b_orig = QPushButton("Original")
        self.b_anal = QPushButton("Analysed")
        grp = QButtonGroup(self)
        for i, b in enumerate((self.b_orig, self.b_anal)):
            b.setCheckable(True)
            b.setObjectName("segLeft" if i == 0 else "segRight")
            grp.addButton(b)
            top.addWidget(b)
        top.setSpacing(0)
        self.b_orig.setChecked(True)
        self.b_orig.clicked.connect(self.refresh_view)
        self.b_anal.clicked.connect(self.refresh_view)
        fit = QPushButton("Fit")
        fit.setToolTip("Fit the image to the window (or double-click the image)")
        fit.clicked.connect(lambda: self.viewer.fit())
        top.addSpacing(8)
        top.addWidget(fit)
        ml.addLayout(top)
        self.viewer = ImageViewer()
        if self.kind == "morphology":
            self.viewer.set_placeholder("Open SEM images, or Microtrac exports:\nparticle report (PDF or page pictures), "
                                        "particle table (.xls), size report (PDF)")
        ml.addWidget(self.viewer, 1)
        if self.kind == "classification":
            ml.addWidget(legend([(CAULI_RGB, "Cauliflower"), (CAULI_SOFT_RGB, "Borderline, cauliflower"),
                                 (DENSE_SOFT_RGB, "Borderline, dense"), (DENSE_RGB, "Dense"),
                                 (FRAME_RGB, "Frame"), ((150, 150, 150), "Not counted")]))
        else:
            ml.addWidget(legend([(SHAPE_RGB[k], SHAPE_NAMES[k]) for k in range(5)] + [(FRAME_RGB, "Frame")]))
        hint = QLabel("Scroll to zoom, drag to pan, double-click to fit. Space switches original / analysed.")
        hint.setObjectName("muted")
        ml.addWidget(hint)
        split.addWidget(mid)
        # right: results
        self.res_area = QScrollArea()
        self.res_area.setWidgetResizable(True)
        self.res_area.setFrameShape(QFrame.NoFrame)
        res = QWidget()
        self.res_lay = QVBoxLayout(res)
        self.res_lay.setContentsMargins(4, 0, 4, 0)
        self._build_results()
        self.res_area.setWidget(res)
        split.addWidget(self.res_area)
        split.setSizes([220, 720, 450])
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        self.btn_image.clicked.connect(self.open_image)
        self.btn_folder.clicked.connect(self.open_folder)
        self.btn_run.clicked.connect(self.run_all)
        self.btn_cancel.clicked.connect(self.cancel)
        sc = QShortcut(QKeySequence(Qt.Key_Space), self)
        sc.activated.connect(self.toggle_view)
        self._update_buttons()

    def _settings_box(self):
        box = QGroupBox("Settings")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(10, 6, 10, 6)
        self.chk_auto_px = QCheckBox("Pixel size from SEM metadata")
        self.chk_auto_px.setChecked(True)
        self.sp_px = QDoubleSpinBox()
        self.sp_px.setRange(0.01, 20)
        self.sp_px.setDecimals(4)
        self.sp_px.setValue(0.4961)
        self.sp_px.setSuffix(" µm/px")
        self.sp_px.setEnabled(False)
        self.sp_px.setToolTip("Used when the image has no Hitachi SU3500 metadata, or when the box on the left is unticked")
        self.chk_auto_px.toggled.connect(lambda on: self.sp_px.setEnabled(not on))
        lay.addWidget(self.chk_auto_px)
        lay.addWidget(self.sp_px)
        lay.addSpacing(14)
        self.chk_resample = QCheckBox("Resample to 0.5 µm/px")
        self.chk_resample.setChecked(True)
        self.chk_resample.setToolTip("The classifier was trained on x100 images at 0.496 µm/px. Images taken at "
                                     "another pixel size are rescaled to it before analysis.")
        lay.addWidget(self.chk_resample)
        lay.addSpacing(14)
        if self.kind == "classification":
            lay.addWidget(QLabel("Cauliflower cut-off"))
            self.sp_cut = QDoubleSpinBox()
            self.sp_cut.setRange(0.05, 0.95)
            self.sp_cut.setSingleStep(0.05)
            self.sp_cut.setValue(0.5)
            self.sp_cut.setToolTip("A particle is drawn and counted as cauliflower when its probability is above "
                                   "this value. The headline area % weighs every particle by its probability, so "
                                   "it does not depend on this cut-off.")
            lay.addWidget(self.sp_cut)
            lay.addSpacing(14)
        lay.addWidget(QLabel("Counting rule"))
        self.cb_rule = QComboBox()
        for k, t in (FRAME_RULES_CLASS if self.kind == "classification" else FRAME_RULES):
            self.cb_rule.addItem(t, k)
        self.cb_rule.setToolTip("Which particles count toward the result:\n"
                                "50% rule: more than half of the particle inside the frame, not cut by the image border\n"
                                "Whole particle inside: only particles entirely inside the frame\n"
                                "Centre inside: particle centre inside the frame, not cut by the border")
        lay.addWidget(self.cb_rule)
        lay.addWidget(QLabel("Frame inset"))
        self.sp_inset = QDoubleSpinBox()
        self.sp_inset.setRange(0, 400)
        self.sp_inset.setValue(50)
        self.sp_inset.setSuffix(" µm")
        lay.addWidget(self.sp_inset)
        lay.addStretch(1)
        return box

    def _build_results(self):
        L = self.res_lay
        self.head_title = QLabel("Results")
        self.head_title.setObjectName("section")
        L.addWidget(self.head_title)
        self.card = QFrame()
        self.card.setObjectName("card")
        cl = QVBoxLayout(self.card)
        self.big = QLabel("–")
        self.big.setObjectName("bigNumber")
        self.big_caption = QLabel("Open an image to see results")
        self.big_caption.setObjectName("muted")
        self.big_caption.setWordWrap(True)
        cl.addWidget(self.big)
        cl.addWidget(self.big_caption)
        self.bars = ShareBars()
        cl.addWidget(self.bars)
        L.addWidget(self.card)
        if self.kind == "classification":
            self.tbl = make_table(["Measure", "Value"])
        else:
            self.tbl = make_table(["Class", "% number", "% volume", "% vol, 30–100 µm"])
            self.tbl2 = make_table(["Descriptor (median)", "Value"])
        L.addWidget(self.tbl)
        if self.kind == "morphology":
            L.addWidget(self.tbl2)
        self.warn = QLabel("")
        self.warn.setObjectName("warnBox")
        self.warn.setWordWrap(True)
        self.warn.setVisible(False)
        L.addWidget(self.warn)
        self.info = QLabel("")
        self.info.setObjectName("muted")
        self.info.setWordWrap(True)
        L.addWidget(self.info)
        self.batch_title = QLabel("Batch summary by sample")
        self.batch_title.setObjectName("section")
        L.addWidget(self.batch_title)
        if self.kind == "classification":
            self.batch = make_table(["Sample", "Fields", "Cauli. area %", "95 % interval", "Field SD"])
        else:
            self.batch = make_table(["Sample", "Files", "Particles", "C5 num %", "C5 vol %", "C5 vol % 30–100"])
        L.addWidget(self.batch)
        self.batch_note_text = ("Samples are grouped by the part of the file name before “_x” "
                                "(e.g. 384_x100_0031 → 384). Each sample pools all particles of its fields."
                                + (" The 95 % interval shows how far the pooled value could move with other fields "
                                   "of the same stub; it needs at least 3 fields." if self.kind == "classification" else
                                   " Microtrac (MT) files are grouped by their batch folder and kind. Report "
                                   "silhouettes carry no sizes, so they give a share by number only; opening the PDF "
                                   "and pictures of the same report counts its particles twice."))
        self.batch_note = QLabel(self.batch_note_text)
        self.batch_note.setObjectName("muted")
        self.batch_note.setWordWrap(True)
        L.addWidget(self.batch_note)
        L.addStretch(1)
        for w in (self.tbl, self.batch_title, self.batch, self.batch_note):
            w.setVisible(False)
        if self.kind == "morphology":
            self.tbl2.setVisible(False)

    # ---------------------------------------------------------------- settings
    def settings(self):
        return Settings(um_px_override=0.0 if self.chk_auto_px.isChecked() else self.sp_px.value(),
                        resample=self.chk_resample.isChecked(),
                        cutoff=self.sp_cut.value() if self.kind == "classification" else 0.5,
                        frame_rule=self.cb_rule.currentData(), inset_um=self.sp_inset.value(),
                        model_path=self.mw.model_path())

    # ------------------------------------------------------------------ input
    def open_image(self):
        start = self.qs.value("last_dir", os.path.expanduser("~"))
        pats = " ".join("*" + e for e in IMAGE_EXT)
        if self.kind == "classification":
            title, filt = "Open SEM images", f"Images ({pats});;All files (*)"
        else:
            title = "Open SEM images or Microtrac files"
            filt = (f"SEM images and Microtrac files ({pats} *.pdf *.xls);;Images ({pats});;"
                    "Microtrac reports (*.pdf);;Microtrac particle tables (*.xls);;All files (*)")
        files, _ = QFileDialog.getOpenFileNames(self, title, start, filt)
        if files:
            self.qs.setValue("last_dir", os.path.dirname(files[0]))
            self.load(sorted(files))

    def open_folder(self):
        start = self.qs.value("last_dir", os.path.expanduser("~"))
        morph = self.kind == "morphology"
        d = QFileDialog.getExistingDirectory(self, "Open a folder of SEM images" + (" or Microtrac files" if morph else ""),
                                             start)
        if d:
            self.qs.setValue("last_dir", d)
            files = list_images(d, MORPH_EXT if morph else IMAGE_EXT)
            if not files:
                QMessageBox.information(self, "Nothing to open", "That folder has no .tif, .png or .jpg images"
                                        + (", and no .pdf or .xls Microtrac files." if morph else "."))
                return
            self.load(files)

    def load(self, files):
        if self.thread is not None:
            return
        self.paths, self.results, self.errors = list(files), {}, {}
        self.list.clear()
        for f in files:
            self.list.addItem(QListWidgetItem(os.path.splitext(os.path.basename(f))[0]))
        self.list.setCurrentRow(0)
        self.run_all()

    # -------------------------------------------------------------- running
    def run_all(self):
        if not self.paths or self.thread is not None:
            return
        if self.kind == "classification" and not os.path.exists(self.mw.model_path()):
            QMessageBox.warning(self, "Model missing", f"Model file not found:\n{self.mw.model_path()}\n\n"
                                "Choose a model on the Home tab.")
            return
        self.results, self.errors = {}, {}
        for i in range(self.list.count()):
            self._label_item(i)
        items = list(enumerate(self.paths))
        self.worker = wk.Worker(self.kind, items, self.settings())
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(self._on_result)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.thread = wk.start(self.worker)
        self.progress.setVisible(True)
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.btn_cancel.setVisible(True)
        self._update_buttons()
        self.refresh_view()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("Stopping after the current image…")

    def _on_progress(self, idx, frac, msg):
        n = len(self.paths)
        self.progress.setValue(int(1000 * (idx + frac) / n))
        self.progress.setFormat(f"{'Image' if self.kind == 'classification' else 'File'} {idx + 1} of {n}: {msg}")

    def _on_result(self, idx, r):
        self.results[idx] = r
        self._label_item(idx)
        if self.list.currentRow() == idx:
            if not self.b_anal.isChecked():
                self.b_anal.setChecked(True)
            self.show_index(idx)
        self._update_batch()

    def _on_failed(self, idx, err):
        self.errors[idx] = err
        self._label_item(idx)
        if self.list.currentRow() == idx:
            self.show_index(idx)

    def _on_finished(self):
        self.thread.quit()
        self.thread.wait()
        self.thread = self.worker = None
        self.progress.setVisible(False)
        self.btn_cancel.setVisible(False)
        done = len(self.results)
        what = "image" if self.kind == "classification" else "file"
        msg = f"{done} of {len(self.paths)} {what}{'s' if len(self.paths) > 1 else ''} analysed"
        if self.errors:
            msg += f", {len(self.errors)} failed (see the red entries)"
        self.status.setText(msg)
        self._update_buttons()

    def _label_item(self, i):
        it = self.list.item(i)
        name = os.path.splitext(os.path.basename(self.paths[i]))[0]
        if i in self.results:
            st = self.results[i]["stats"]
            src = self.results[i].get("source", "sem")
            flag = "  ⚠" if self.results[i].get("warnings") else ""
            if self.kind == "classification":
                line = f"{fmt(st.get('cauliflower_area_pct'), 1, ' %')} cauliflower"
            elif src == "size":
                line = f"size report, D50 {fmt(st.get('LD_D50'), 0, ' µm')}"
            elif src == "gallery":
                line = f"{fmt(st.get('num_pct_C5'), 1, ' %')} C5 by number"
            else:
                line = f"{fmt(st.get('vol_pct_C5'), 1, ' %')} C5 by volume"
            it.setText(f"{name}{flag}\n   {line}")
            it.setForeground(QColor("#1F2328"))
            it.setToolTip(self.paths[i] + ("\n" + "\n".join(self.results[i]["warnings"]) if flag else ""))
        elif i in self.errors:
            it.setText(f"{name}\n   failed")
            it.setForeground(QColor("#B42318"))
            it.setToolTip(self.errors[i])
        else:
            it.setText(f"{name}\n   waiting")
            it.setForeground(QColor("#7D8590"))

    def _update_buttons(self):
        busy = self.thread is not None
        has = bool(self.results)
        self.btn_image.setEnabled(not busy)
        self.btn_folder.setEnabled(not busy)
        self.btn_run.setEnabled(not busy and bool(self.paths))
        self.btn_pdf.setEnabled(not busy and has)
        self.btn_csv.setEnabled(not busy and has)

    # ---------------------------------------------------------------- viewing
    def toggle_view(self):
        if self.b_anal.isEnabled():
            (self.b_orig if self.b_anal.isChecked() else self.b_anal).setChecked(True)
            self.refresh_view()

    def show_index(self, idx):
        self.current = idx
        if idx is None or idx < 0 or idx >= len(self.paths):
            return
        self.title.setText(os.path.basename(self.paths[idx]))
        self.refresh_view()
        self.show_results(self.results.get(idx), self.errors.get(idx))

    def refresh_view(self):
        idx = self.current
        if idx is None or idx < 0 or idx >= len(self.paths):
            return
        r = self.results.get(idx)
        self.b_anal.setEnabled(r is not None)
        if r is None:
            self.b_orig.setChecked(True)
            if not self.paths[idx].lower().endswith(IMAGE_EXT):
                self.viewer.set_placeholder("Failed to read this file" if idx in self.errors else "Reading the file…")
                return
            try:
                a = np.array(Image.open(self.paths[idx]).convert("L"))
                self.viewer.set_image(a[: data_bar_row(a)], keep_view=True)
            except Exception as e:
                self.viewer.set_placeholder(f"Cannot open this image:\n{e}")
            return
        self.viewer.set_image(r["analysed"] if self.b_anal.isChecked() else r["original"], keep_view=True)

    def show_results(self, r, err=None):
        if r is None:
            self.big.setText("…" if err is None else "Failed")
            self.big_caption.setText("Analysing…" if err is None and self.thread else
                                     (err.splitlines()[0] if err else "Not analysed yet"))
            self.bars.set_items([])
            self.tbl.setVisible(False)
            if self.kind == "morphology":
                self.tbl2.setVisible(False)
            self.info.setText("")
            self.warn.setVisible(False)
            return
        st, info = r["stats"], r["image_info"]
        w = r.get("warnings") or []
        what = "image" if r.get("source", "sem") == "sem" else "file"
        self.warn.setText(f"<b>Check this {what}</b><br>" + "<br>".join("• " + x for x in w))
        self.warn.setVisible(bool(w))
        if self.kind == "classification":
            self.big.setText(fmt(st["cauliflower_area_pct"], 1, " %"))
            v = (r.get("model") or {}).get("validation") or {}
            cap = "cauliflower, as a share of the projected area of all counted particles"
            if v.get("field_mae_pp") is not None:
                cap += (f". On batches the model never saw, one field was within about {v['field_mae_pp']:.0f} points "
                        "of the reference on average.")
            self.big_caption.setText(cap)
            self.bars.set_items([("Cauliflower (area)", st["cauliflower_area_pct"], CAULI_RGB),
                                 ("Dense (area)", 100 - st["cauliflower_area_pct"] if not math.isnan(st["cauliflower_area_pct"]) else float("nan"), DENSE_RGB),
                                 ("Cauliflower (volume)", st["cauliflower_vol_pct"], CAULI_RGB)])
            fill_table(self.tbl, [
                ["Counting only particles called cauliflower", fmt(st.get("calls_area_pct"), 1, " % of area")],
                ["Area held by borderline particles", fmt(st.get("borderline_area_pct"), 1, " %")],
                ["Particles counted", st["n_counted"]],
                ["  cauliflower", st["n_cauliflower"]],
                ["  dense", st["n_dense"]],
                ["  of which borderline", st.get("n_borderline", 0)],
                ["Median size, cauliflower", fmt(st["median_ecd_cauliflower_um"], 0, " µm")],
                ["Median size, dense", fmt(st["median_ecd_dense_um"], 0, " µm")],
                ["Not counted: below 10 µm", st["n_small"]],
                ["Not counted: outside frame", st["n_outside_frame"]],
                ["Counting frame", f"{st['frame_um'][0]:.0f} × {st['frame_um'][1]:.0f} µm"],
                ["Image covered by particles", fmt(st.get("coverage_pct"), 0, " %")],
                ["Large-particle area in touching clusters", fmt(st.get("touching_area_pct"), 0, " %")],
            ])
        else:
            self._show_morphology(r)
            return
        self.tbl.setVisible(True)
        self.info.setText(self._image_text(info))

    def _image_text(self, info):
        txt = f"{info['native_size'][0]} × {info['native_size'][1]} px"
        if info.get("magnification"):
            txt += f", x{info['magnification']}"
        txt += f". Pixel size {info['native_um_px']:.3f} µm/px ({info['um_source']})"
        if info["scale"] != 1.0:
            txt += f", resampled ×{info['scale']:.2f} to {info['um_px']:.3f} µm/px"
        txt += "."
        if info["um_source"].startswith("default"):
            txt = ("<span style='color:#B54708'><b>No pixel size in this image's metadata:</b> the default "
                   "0.496 µm/px was used. If that is wrong, untick “Pixel size from SEM metadata”, enter the "
                   "right value and press Analyse.</span><br>" + txt)
        return txt

    def _show_morphology(self, r):
        st, src = r["stats"], r.get("source", "sem")
        names = ["C1 compact", "C2 elongated", "C3 blocky", "C4 concave", "C5 branched"]
        nan = float("nan")
        if src == "size":
            self.big.setText(fmt(st.get("LD_D50"), 0, " µm"))
            self.big_caption.setText("median particle size by volume (D50), laser diffraction")
            self.bars.set_items([])
            self.tbl.setVisible(False)
            self.tbl2.setHorizontalHeaderLabels(["Size by volume", "Value"])
            fill_table(self.tbl2, [["D10", fmt(st.get("LD_D10"), 1, " µm")], ["D50", fmt(st.get("LD_D50"), 1, " µm")],
                                   ["D90", fmt(st.get("LD_D90"), 1, " µm")],
                                   ["Span, (D90 − D10) / D50", fmt(st.get("LD_span"), 2)],
                                   ["Mean by volume (MV)", fmt(st.get("LD_MV"), 1, " µm")],
                                   ["Mean by area (MA)", fmt(st.get("LD_MA"), 1, " µm")],
                                   ["Mean by number (MN)", fmt(st.get("LD_MN"), 1, " µm")]])
            self.tbl2.setVisible(True)
            self.info.setText(r.get("note", ""))
            return
        by_number = src == "gallery"
        dash = lambda v: "–" if by_number else fmt(v)
        if by_number:
            self.big.setText(fmt(st.get("num_pct_C5", nan), 1, " %"))
            self.big_caption.setText(f"of the {st.get('n_counted', 0)} particles shown in the report are C5 "
                                     "(branched / agglomerate), by number")
            self.bars.set_items([(SHAPE_NAMES[k], st.get(f"num_pct_C{k + 1}", nan), SHAPE_RGB[k]) for k in range(5)])
            last = ["Particles", st.get("n_counted", 0), "", ""]
            med = []
        else:
            self.big.setText(fmt(st.get("vol_pct_C5", nan), 1, " %"))
            self.big_caption.setText("of particle volume in C5 (branched / agglomerate)" +
                                     (", all sizes, from Microtrac's own descriptors" if src == "table" else ""))
            self.bars.set_items([(SHAPE_NAMES[k], st.get(f"vol_pct_C{k + 1}", nan), SHAPE_RGB[k]) for k in range(5)])
            last = ["Particles", st.get("n_counted", 0), "", f"{st.get('n_band', 0)} in band"]
            med = [["D50 by number", fmt(st.get("D50_num_um"), 1, " µm")]]
        fill_table(self.tbl, [[names[k - 1], fmt(st.get(f"num_pct_C{k}")), dash(st.get(f"vol_pct_C{k}")),
                               dash(st.get(f"band_vol_pct_C{k}"))] for k in range(1, 6)] + [last])
        self.tbl2.setHorizontalHeaderLabels(["Descriptor (median)", "Value"])
        fill_table(self.tbl2, med + [[k, fmt(st.get(f"median_{k}"), 3)]
                                     for k in ("Circularity", "Solidity", "Convexity", "W/L Aspect Ratio", "Extent")])
        self.tbl.setVisible(True)
        self.tbl2.setVisible(True)
        self.info.setText(self._image_text(r["image_info"]) if src == "sem" else r.get("note", ""))

    def _update_batch(self):
        res = [self.results[i] for i in sorted(self.results)]
        show = len(self.paths) > 1 and bool(res)
        for w in (self.batch_title, self.batch, self.batch_note):
            w.setVisible(show)
        if not show:
            return
        if self.kind == "classification":
            rows = [[s["sample"], s["fields"], fmt(s["cauliflower_area_pct"]),
                     f"{fmt(s['ci_low_pct'])} – {fmt(s['ci_high_pct'])}" if s["fields"] >= 3 else "–",
                     fmt(s["field_sd_pct"])] for s in summarise_classification(res)]
        else:
            rows, sizes = [], []
            for s in summarise_morphology(res):
                if s["source"] == "size":
                    sizes.append(f"{s['sample'].split(' · ')[0]}: D10 {fmt(s.get('LD_D10'), 0)}, "
                                 f"D50 {fmt(s.get('LD_D50'), 0)}, D90 {fmt(s.get('LD_D90'), 0)} µm")
                    continue
                vol = (lambda v: "–") if s["source"] == "gallery" else fmt
                rows.append([s["sample"], s["fields"], s.get("n_counted", 0), fmt(s.get("num_pct_C5")),
                             vol(s.get("vol_pct_C5")), vol(s.get("band_vol_pct_C5"))])
            self.batch.setVisible(bool(rows))
            self.batch_note.setText(self.batch_note_text + ("<br><b>Size reports (laser diffraction, by volume):</b> "
                                                            + "; ".join(sizes) + "." if sizes else ""))
        fill_table(self.batch, rows)

    # ---------------------------------------------------------------- export
    def _chosen(self, all_images):
        if all_images:
            return [self.results[i] for i in sorted(self.results)]
        r = self.results.get(self.current)
        return [r] if r else []

    def export_pdf(self, all_images=True):
        res = self._chosen(all_images)
        if not res:
            QMessageBox.information(self, "Nothing to export", "Analyse an image first.")
            return
        stem = res[0]["name"] if len(res) == 1 else os.path.basename(os.path.dirname(res[0]["path"])) or "batch"
        suffix = "classification" if self.kind == "classification" else "morphology"
        start = os.path.join(self.qs.value("last_out", self.qs.value("last_dir", os.path.expanduser("~"))),
                             f"{stem}_{suffix}_report.pdf")
        f, _ = QFileDialog.getSaveFileName(self, "Save PDF report", start, "PDF (*.pdf)")
        if not f:
            return
        self.qs.setValue("last_out", os.path.dirname(f))
        try:
            self.status.setText("Writing report…")
            self.repaint()
            build_report(res, f)
            self.status.setText(f"Report saved: {f}")
        except Exception as e:
            QMessageBox.critical(self, "Report failed", f"{type(e).__name__}: {e}")

    def export_table(self, per_particle):
        res = self._chosen(True)
        if not res:
            return
        suffix = ("particles" if per_particle else "summary")
        stem = res[0]["name"] if len(res) == 1 else os.path.basename(os.path.dirname(res[0]["path"])) or "batch"
        start = os.path.join(self.qs.value("last_out", self.qs.value("last_dir", os.path.expanduser("~"))),
                             f"{stem}_{self.kind}_{suffix}.csv")
        f, _ = QFileDialog.getSaveFileName(self, "Save CSV", start, "CSV (*.csv)")
        if not f:
            return
        self.qs.setValue("last_out", os.path.dirname(f))
        (export_csv if per_particle else export_summary_csv)(res, f)
        self.status.setText(f"Saved: {f}")
