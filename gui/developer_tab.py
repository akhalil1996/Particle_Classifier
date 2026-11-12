"""Developer tab: correct the classifier's calls particle by particle and refit it on this PC.

Locked by a password. Open SEM photos, click particles to change the call,
submit; the classifier is refitted on the reference particles plus everything
judged so far, and the refined version is used by the classification tab at
once. "Publish to GitHub" sends the refined classifier to the repository,
which builds a new installer with it.
"""
import hashlib
import hmac
import io
import os
import traceback

import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, Signal, QSettings
from PySide6.QtGui import QGuiApplication, QCursor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QStackedWidget,
                               QFileDialog, QMessageBox, QProgressBar, QDoubleSpinBox, QTextBrowser, QGroupBox,
                               QDialog, QCheckBox, QDialogButtonBox, QFormLayout, QFrame, QSplitter)
from scipy import ndimage as ndi
from skimage import segmentation

from cauliquant import trainer, publish, __version__
from cauliquant.analysis import classify_image, IMAGE_EXT, list_images, CAULI_RGB, DENSE_RGB, FRAME_RGB
from cauliquant.classifier import load_card
from cauliquant.resources import DEFAULT_UNET, local_card_path
from .viewer import ImageViewer
from .analysis_tab import legend

_SALT = bytes.fromhex("9f65aa5a2795abaf3ec69c2250be3359")
_HASH = bytes.fromhex("76c96c4108f310bf1cfa212a191021317381ad7b402966109bacd2a78c56e397")
CHANGED_RGB = (230, 0, 200)        # outline of a particle whose call you changed
SKIP_RGB = (90, 90, 90)


def password_ok(text):
    h = hashlib.pbkdf2_hmac("sha256", text.encode("utf-8"), _SALT, 300000)
    return hmac.compare_digest(h, _HASH)


class ClickViewer(ImageViewer):
    """Image view that reports clicks (not drags) in image pixel coordinates."""
    clicked = Signal(int, int, bool)        # x, y, right button

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        p = e.position().toPoint()
        if getattr(self, "_press", None) is not None and (p - self._press).manhattanLength() < 5 and self._size:
            s = self.mapToScene(p)
            self.clicked.emit(int(s.x()), int(s.y()), e.button() == Qt.RightButton)
        self._press = None


class _Job(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(float, str)

    def __init__(self, path, settings):
        super().__init__()
        self.path, self.settings = path, settings

    def run(self):
        try:
            r = classify_image(self.path, self.settings, progress=lambda f, m="": self.progress.emit(f, m),
                               keep_labels=True)
            self.done.emit(r)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")


class DeveloperTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.qs = QSettings()
        self.queue, self.pos = [], -1
        self.result = None
        self.judged, self.model_call, self.skipped = {}, {}, set()
        self.thread = self.job = None
        self.stack = QStackedWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.stack)
        self.stack.addWidget(self._lock_page())
        self.stack.addWidget(self._work_page())

    # ------------------------------------------------------------------ lock
    def _lock_page(self):
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.addStretch(1)
        box = QFrame()
        box.setObjectName("card")
        box.setMaximumWidth(460)
        l = QVBoxLayout(box)
        t = QLabel("Developer: classifier training")
        t.setObjectName("cardTitle")
        l.addWidget(t)
        d = QLabel("Correct the classifier particle by particle and retrain it. This tab is for the developer of "
                   "the software and needs a password.")
        d.setWordWrap(True)
        d.setObjectName("muted")
        l.addWidget(d)
        self.pw = QLineEdit()
        self.pw.setEchoMode(QLineEdit.Password)
        self.pw.setPlaceholderText("Password")
        self.pw.returnPressed.connect(self.unlock)
        l.addWidget(self.pw)
        b = QPushButton("Unlock")
        b.setObjectName("primary")
        b.clicked.connect(self.unlock)
        l.addWidget(b)
        self.pw_msg = QLabel("")
        self.pw_msg.setStyleSheet("color:#B42318")
        l.addWidget(self.pw_msg)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(box)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(2)
        return w

    def unlock(self):
        if password_ok(self.pw.text()):
            self.pw.clear()
            self.pw_msg.setText("")
            self.stack.setCurrentIndex(1)
            self._update_model_box()
        else:
            self.pw_msg.setText("Wrong password.")
            self.pw.selectAll()

    def lock(self):
        self.stack.setCurrentIndex(0)

    # ------------------------------------------------------------------ work
    def _work_page(self):
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(10, 10, 10, 10)
        bar = QHBoxLayout()
        self.b_open = QPushButton("Open photos…")
        self.b_open.clicked.connect(self.open_photos)
        self.b_folder = QPushButton("Open folder…")
        self.b_folder.clicked.connect(self.open_folder)
        self.b_prev = QPushButton("◀")
        self.b_prev.clicked.connect(lambda: self.go(self.pos - 1))
        self.b_next = QPushButton("▶")
        self.b_next.clicked.connect(lambda: self.go(self.pos + 1))
        self.lbl_pos = QLabel("")
        self.b_submit = QPushButton("Submit and retrain")
        self.b_submit.setObjectName("primary")
        self.b_submit.clicked.connect(self.submit)
        self.b_again = QPushButton("Analyse again")
        self.b_again.setToolTip("Run the current photo again with the classifier as it is now")
        self.b_again.clicked.connect(lambda: self.go(self.pos))
        for x in (self.b_open, self.b_folder, self.b_prev, self.lbl_pos, self.b_next, self.b_again, self.b_submit):
            bar.addWidget(x)
        bar.addStretch(1)
        b_lock = QPushButton("Lock")
        b_lock.clicked.connect(self.lock)
        bar.addWidget(b_lock)
        root.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.viewer = ClickViewer()
        self.viewer.set_placeholder("Open SEM photos to judge")
        self.viewer.clicked.connect(self.on_click)
        ll.addWidget(self.viewer, 1)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 1000)
        ll.addWidget(self.progress)
        split.addWidget(left)

        right = QWidget()
        right.setMinimumWidth(360)
        right.setMaximumWidth(460)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        how = QLabel("<b>Left-click</b> a particle to switch it between cauliflower and dense. <b>Right-click</b> to "
                     "leave it out of training (bad outline, cannot tell). Drag to pan, wheel to zoom, double-click "
                     "to fit. Particles you leave unchanged count as confirmed when you submit.")
        how.setWordWrap(True)
        how.setObjectName("muted")
        rl.addWidget(how)
        rl.addWidget(legend([(CAULI_RGB, "Cauliflower"), (DENSE_RGB, "Dense"), (CHANGED_RGB, "Changed by you")]))
        rl.addWidget(legend([(SKIP_RGB, "Left out"), (FRAME_RGB, "Counting frame")]))
        g = QGroupBox("This photo")
        gl = QVBoxLayout(g)
        self.lbl_photo = QLabel("No photo yet.")
        self.lbl_photo.setWordWrap(True)
        self.lbl_photo.setTextFormat(Qt.RichText)
        gl.addWidget(self.lbl_photo)
        rl.addWidget(g)
        g2 = QGroupBox("Classifier")
        f2 = QVBoxLayout(g2)
        self.lbl_model = QLabel("")
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setTextFormat(Qt.RichText)
        f2.addWidget(self.lbl_model)
        wr = QHBoxLayout()
        wr.addWidget(QLabel("Weight of your particles"))
        self.sp_weight = QDoubleSpinBox()
        self.sp_weight.setRange(0.5, 10)
        self.sp_weight.setSingleStep(0.5)
        self.sp_weight.setValue(float(self.qs.value("dev_weight", 2.0)))
        self.sp_weight.setToolTip("How much one of your judged particles counts against one reference particle")
        self.sp_weight.valueChanged.connect(lambda v: self.qs.setValue("dev_weight", v))
        wr.addWidget(self.sp_weight)
        f2.addLayout(wr)
        self.chk_confirm = QCheckBox("Unchanged particles count as confirmed")
        self.chk_confirm.setChecked(self.qs.value("dev_confirm", "true") in (True, "true"))
        self.chk_confirm.setToolTip("On: every particle you did not leave out is a training example (look at all of "
                                    "them before submitting). Off: only the particles you changed are used.")
        self.chk_confirm.toggled.connect(lambda v: self.qs.setValue("dev_confirm", "true" if v else "false"))
        f2.addWidget(self.chk_confirm)
        br = QHBoxLayout()
        self.b_revert = QPushButton("Back to shipped")
        self.b_revert.clicked.connect(self.revert)
        self.b_publish = QPushButton("Publish to GitHub…")
        self.b_publish.clicked.connect(self.publish)
        br.addWidget(self.b_revert)
        br.addWidget(self.b_publish)
        f2.addLayout(br)
        rl.addWidget(g2)
        self.report = QTextBrowser()
        self.report.setOpenExternalLinks(True)
        rl.addWidget(self.report, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        root.addWidget(split, 1)
        self._buttons()
        return w

    def _buttons(self):
        busy = self.thread is not None
        have = self.result is not None
        self.b_prev.setEnabled(not busy and self.pos > 0)
        self.b_next.setEnabled(not busy and 0 <= self.pos < len(self.queue) - 1)
        self.b_submit.setEnabled(not busy and have)
        self.b_again.setEnabled(not busy and self.pos >= 0)
        self.b_open.setEnabled(not busy)
        self.b_folder.setEnabled(not busy)
        self.lbl_pos.setText(f"{self.pos + 1} / {len(self.queue)}" if self.queue else "")

    def _update_model_box(self):
        shipped = load_card(DEFAULT_UNET, refined=False)
        active = load_card(DEFAULT_UNET)
        rev, srev = int(active.get("revision", 0)), int(shipped.get("revision", 0))
        n_local = len(trainer._read(trainer.local_csv()))
        if rev > srev:
            ref = active.get("refined", {})
            txt = (f"<b>Refined on this PC, revision {rev}</b> (installer has revision {srev}).<br>"
                   f"Trained on {ref.get('base_particles', '?')} reference particles and {ref.get('judged_particles', '?')} "
                   f"of yours from {ref.get('judged_photos', '?')} photo(s). The classification tab uses it now.")
        else:
            txt = f"<b>Shipped classifier, revision {srev}</b>, as in this installer ({__version__})."
        if n_local:
            txt += f"<br><span style='color:#656D76'>{n_local} particles judged on this PC.</span>"
        self.lbl_model.setText(txt)
        self.b_publish.setEnabled(rev > srev)
        self.b_revert.setEnabled(rev > srev)

    # ----------------------------------------------------------------- input
    def open_photos(self):
        start = self.qs.value("dev_dir", self.qs.value("last_dir", os.path.expanduser("~")))
        pats = " ".join("*" + e for e in IMAGE_EXT)
        files, _ = QFileDialog.getOpenFileNames(self, "Open SEM photos", start, f"Images ({pats})")
        if files:
            self.qs.setValue("dev_dir", os.path.dirname(files[0]))
            self.queue = sorted(files)
            self.go(0)

    def open_folder(self):
        start = self.qs.value("dev_dir", self.qs.value("last_dir", os.path.expanduser("~")))
        d = QFileDialog.getExistingDirectory(self, "Open a folder of SEM photos", start)
        if d:
            self.qs.setValue("dev_dir", d)
            files = list_images(d, IMAGE_EXT)
            if files:
                self.queue = files
                self.go(0)

    def _settings(self):
        s = self.mw.tab_class.settings()
        s.model_path = DEFAULT_UNET
        return s

    def go(self, i):
        if not (0 <= i < len(self.queue)) or self.thread is not None:
            return
        if self._unsaved() and i != self.pos:
            if QMessageBox.question(self, "Not submitted", "You changed calls on this photo and have not submitted "
                                    "them. Move on and drop the changes?") != QMessageBox.Yes:
                return
        self.pos = i
        self.result = None
        self.viewer.set_placeholder("Analysing…")
        self.lbl_photo.setText(os.path.basename(self.queue[i]))
        self.job = _Job(self.queue[i], self._settings())
        self.thread = QThread()
        self.job.moveToThread(self.thread)
        self.thread.started.connect(self.job.run)
        self.job.done.connect(self._loaded)
        self.job.failed.connect(self._failed)
        self.job.progress.connect(lambda f, m: (self.progress.setValue(int(1000 * f)), self.progress.setFormat(m)))
        self.progress.setVisible(True)
        self.thread.start()
        self._buttons()

    def _end_job(self):
        self.thread.quit()
        self.thread.wait()
        self.thread = self.job = None
        self.progress.setVisible(False)

    def _failed(self, err):
        self._end_job()
        self.viewer.set_placeholder("Could not analyse this photo")
        self.report.setPlainText(err)
        self._buttons()

    def _loaded(self, r):
        self._end_job()
        self.result = r
        self._submitted = False
        lab = r["labels"]
        self.lab = lab
        self.slices = ndi.find_objects(lab)
        self.edges = segmentation.find_boundaries(lab, mode="inner")
        self.rows = {x["id"]: x for x in r["rows"]}
        self.model_call = {x["id"]: x["class"] for x in r["rows"] if x["counted"]}
        self.judged = dict(self.model_call)
        self.skipped = {i for i, x in self.rows.items() if i in self.model_call and x.get("edge")}
        g = r["gray"].astype(np.float32)
        self.base = np.stack([g] * 3, -1)
        self.img = self._paint_window((0, lab.shape[0], 0, lab.shape[1]))
        self._frame()
        self.viewer.set_image(self.img, keep_view=True)
        self._update_photo()
        self._buttons()

    def _unsaved(self):
        return self.result is not None and (self.judged != self.model_call) and not getattr(self, "_submitted", False)

    # --------------------------------------------------------------- drawing
    def _colour(self, i):
        if i in self.skipped:
            return None
        return CAULI_RGB if self.judged.get(i) == "cauliflower" else DENSE_RGB

    def _paint_window(self, win):
        y0, y1, x0, x1 = win
        lab = self.lab[y0:y1, x0:x1]
        out = self.base[y0:y1, x0:x1].copy()
        ids = np.unique(lab)
        ids = ids[ids > 0]
        n = int(self.lab.max())
        lut = np.zeros((n + 1, 3), np.float32)
        has = np.zeros(n + 1, bool)
        skip = np.zeros(n + 1, bool)
        for i in ids:
            if i in self.judged:
                c = self._colour(int(i))
                if c is None:
                    skip[i] = True
                else:
                    lut[i] = c
                    has[i] = True
        sel = has[lab]
        out[sel] = 0.5 * out[sel] + 0.5 * lut[lab[sel]]
        sk = skip[lab]
        out[sk] = out[sk] * 0.35
        e = self.edges[y0:y1, x0:x1]
        out[e & sel] = lut[lab[e & sel]] * 0.75
        out[e & ~sel & (lab > 0)] = (150, 150, 150)
        for i in ids:
            i = int(i)
            if i in self.judged and i not in self.skipped and self.judged[i] != self.model_call.get(i):
                reg = lab == i
                ring = reg & ~ndi.binary_erosion(reg, iterations=4)
                out[ring] = CHANGED_RGB
        return out.clip(0, 255).astype(np.uint8)

    def _frame(self):
        y0, x0, y1, x1 = self.result["frame_box"]
        w = max(2, int(round(min(self.img.shape[:2]) / 600)))
        o = self.img
        o[y0:y0 + w, x0:x1] = FRAME_RGB
        o[y1 - w:y1, x0:x1] = FRAME_RGB
        o[y0:y1, x0:x0 + w] = FRAME_RGB
        o[y0:y1, x1 - w:x1] = FRAME_RGB

    def on_click(self, x, y, right):
        if self.result is None or not (0 <= y < self.lab.shape[0] and 0 <= x < self.lab.shape[1]):
            return
        i = int(self.lab[y, x])
        if i == 0 or i not in self.judged:
            return
        if right:
            self.skipped.symmetric_difference_update({i})
        elif i in self.skipped:
            self.skipped.discard(i)
        else:
            self.judged[i] = "dense" if self.judged[i] == "cauliflower" else "cauliflower"
        self._submitted = False
        sl = self.slices[i - 1]
        H, W = self.lab.shape
        win = (max(0, sl[0].start - 2), min(H, sl[0].stop + 2), max(0, sl[1].start - 2), min(W, sl[1].stop + 2))
        self.img[win[0]:win[1], win[2]:win[3]] = self._paint_window(win)
        self._frame()
        self.viewer.set_image(self.img, keep_view=True)
        self._update_photo()

    def _update_photo(self):
        r = self.result
        use = [i for i in self.judged if i not in self.skipped]
        area = lambda ids: sum(self.rows[i]["counted_um2"] for i in ids)
        tot = area(use)
        mine = area([i for i in use if self.judged[i] == "cauliflower"])
        ch = sum(1 for i in use if self.judged[i] != self.model_call[i])
        to_c = sum(1 for i in use if self.judged[i] == "cauliflower" and self.model_call[i] == "dense")
        st = r["stats"]
        rev = r.get("model", {}).get("revision", 0)
        self.lbl_photo.setText(
            f"<b>{r['name']}</b><br>"
            f"Classifier (revision {rev}): <b>{st['cauliflower_area_pct']:.1f} %</b> cauliflower by area "
            f"({st['calls_area_pct']:.1f} % by calls)<br>"
            f"Your judgement: <b>{100 * mine / tot if tot else float('nan'):.1f} %</b><br>"
            f"{len(self.judged)} particles in the frame; you changed {ch} ({to_c} to cauliflower, {ch - to_c} to dense) "
            f"and left out {len(self.skipped)}.")

    # ------------------------------------------------------------- training
    def submit(self):
        if self.result is None:
            return
        judged = {i: c for i, c in self.judged.items() if i not in self.skipped}
        if not self.chk_confirm.isChecked():
            judged = {i: c for i, c in judged.items() if c != self.model_call.get(i)}
            if not judged:
                QMessageBox.information(self, "Nothing to submit", "You have not changed any call on this photo, and "
                                        "unchanged particles are set not to count.")
                return
        QGuiApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            rep = trainer.submit(self.result, judged, self.queue[self.pos], DEFAULT_UNET, self.sp_weight.value())
        except Exception as e:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Could not retrain", f"{type(e).__name__}: {e}")
            return
        QGuiApplication.restoreOverrideCursor()
        self._submitted = True
        pc = lambda v: f"{100 * v:.1f} %" if v == v else "n/a"
        html = (f"<p><b>Retrained: classifier revision {rep['revision']}</b>.</p>"
                f"<p>{rep['photo_particles']} particles from {self.result['name']} stored "
                f"({rep['photo_changed']} changed by you). Training set now: {rep['base_particles']} reference "
                f"particles + {rep['your_particles']} of yours from {rep['your_photos']} photo{'s' if rep['your_photos'] != 1 else ''}, weight "
                f"{rep['weight']:g}.</p>")
        if rep.get("photo_changed"):
            html += (f"<p>Of the {rep['photo_changed']} particles you changed, {rep.get('changed_now_agree', 0)} are now "
                     f"called your way. The others sit too close to particles labelled the other way; more photos "
                     f"with the same kind of correction will move them.</p>")
        if "base_agree_before" in rep:
            html += (f"<p>Agreement with the reference particles: {pc(rep['base_agree_before'])} before, "
                     f"{pc(rep['base_agree_after'])} after.<br>Agreement with all your judged particles: "
                     f"{pc(rep['your_agree_before'])} before, {pc(rep['your_agree_after'])} after.")
            if "your_agree_unseen" in rep:
                html += (f"<br>Each of your photos scored by a classifier trained without it: "
                         f"{pc(rep['your_agree_unseen'])}. This is the figure to watch.")
            html += "</p>"
        if rep.get("base_agree_before") == rep.get("base_agree_before") and \
                rep.get("base_agree_after", 1) < rep.get("base_agree_before", 0) - 0.03:
            html += ("<p style='color:#B54708'>Agreement with the reference particles dropped by more than 3 points. "
                     "Your judgement may differ from the reference in a systematic way, or the weight is high.</p>")
        html += "<p>The classification tab uses this classifier now. Use <b>Publish to GitHub</b> to put it in the installer.</p>"
        self.report.setHtml(html)
        self._update_model_box()
        if self.pos < len(self.queue) - 1:
            self.go(self.pos + 1)

    def revert(self):
        if QMessageBox.question(self, "Back to the shipped classifier",
                                "Use the classifier that came with this installer again? The particles you judged stay "
                                "stored and come back with the next submit.") == QMessageBox.Yes:
            trainer.revert()
            self._update_model_box()
            self.report.setHtml("<p>Back to the shipped classifier.</p>")

    # -------------------------------------------------------------- publish
    def publish(self):
        card_file = local_card_path()
        if not os.path.exists(card_file):
            return
        dlg = PublishDialog(self, card_file)
        if dlg.exec() == QDialog.Accepted and dlg.outcome:
            o = dlg.outcome
            self.report.setHtml(
                f"<p><b>Published.</b> Version {o['version']} is building now; the installer appears on the "
                f"<a href='{o['release_url']}'>release page</a> in about five minutes "
                f"(<a href='{o['actions_url']}'>build progress</a>).</p>"
                f"<p>Everyone who installs {o['version']} gets this classifier. On this PC nothing changes.</p>")


class PublishDialog(QDialog):
    def __init__(self, parent, card_file):
        super().__init__(parent)
        import json
        self.setWindowTitle("Publish the trained classifier to GitHub")
        self.card_file = card_file
        self.outcome = None
        self.qs = QSettings()
        card = json.load(open(card_file, encoding="utf-8"))
        ref = card.get("refined", {})
        self.rev = card.get("revision", 0)
        self.n = ref.get("judged_particles", 0)
        self.k = ref.get("judged_photos", 0)
        lay = QVBoxLayout(self)
        t = QLabel(f"This sends classifier revision {self.rev} ({self.n} judged particles from {self.k} photos) to "
                   f"<b>{publish.OWNER}/{publish.REPO}</b> as one commit, raises the version from {__version__} to the "
                   f"next number and starts a new installer build. Commits are made in the name of "
                   f"{publish.AUTHOR['name']}.")
        t.setWordWrap(True)
        t.setTextFormat(Qt.RichText)
        lay.addWidget(t)
        f = QFormLayout()
        self.tok = QLineEdit(self.qs.value("gh_token", ""))
        self.tok.setEchoMode(QLineEdit.Password)
        self.tok.setPlaceholderText("github_pat_…")
        f.addRow("GitHub token", self.tok)
        self.remember = QCheckBox("Remember the token on this PC")
        self.remember.setChecked(bool(self.qs.value("gh_token", "")))
        f.addRow("", self.remember)
        lay.addLayout(f)
        h = QLabel("Token: github.com → Settings → Developer settings → Fine-grained tokens → Generate. Repository "
                   "access: only Particle_Classifier. Permissions: Contents, Read and write.")
        h.setWordWrap(True)
        h.setObjectName("muted")
        lay.addWidget(h)
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        self.msg.setStyleSheet("color:#B42318")
        lay.addWidget(self.msg)
        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.go = bb.addButton("Publish", QDialogButtonBox.AcceptRole)
        self.go.setObjectName("primary")
        bb.accepted.connect(self.run)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(520, 260)

    def run(self):
        token = self.tok.text().strip()
        if not token:
            self.msg.setText("Paste a token first.")
            return
        if self.remember.isChecked():
            self.qs.setValue("gh_token", token)
        else:
            self.qs.remove("gh_token")
        QGuiApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            publish.check_token(token)
            buf = io.StringIO()
            import csv
            w = csv.DictWriter(buf, fieldnames=trainer.COLUMNS, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(trainer.judged_rows())
            files = {"models/cauliflower_unet.json": open(self.card_file, "rb").read(),
                     "models/classifier_user.csv": buf.getvalue().encode("utf-8")}
            msg = (f"Classifier revision {self.rev}: refitted with {self.n} judged particles from {self.k} photos")
            self.outcome = publish.publish(token, files, msg, bump=True)
        except publish.PublishError as e:
            QGuiApplication.restoreOverrideCursor()
            self.msg.setText(str(e))
            return
        except Exception as e:
            QGuiApplication.restoreOverrideCursor()
            self.msg.setText(f"{type(e).__name__}: {e}")
            return
        QGuiApplication.restoreOverrideCursor()
        self.accept()
