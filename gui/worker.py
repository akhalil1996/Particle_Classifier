"""Runs analyses in a background thread so the window stays responsive."""
import io
import traceback

from PIL import Image
from PySide6.QtCore import QObject, QThread, Signal

from cauliquant.analysis import classify_image, morphology_image


def _jpeg(a, q=92):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=q)
    return buf.getvalue()


class Worker(QObject):
    progress = Signal(int, float, str)      # image index, fraction of that image, message
    result = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal()

    def __init__(self, kind, items, settings):
        super().__init__()
        self.kind, self.items, self.settings = kind, items, settings
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        fn = classify_image if self.kind == "classification" else morphology_image
        for idx, path in self.items:
            if self._cancel:
                break
            try:
                r = fn(path, self.settings, progress=lambda f, m="", i=idx: self.progress.emit(i, f, m))
                r["original"] = _jpeg(r["original"])
                r["analysed"] = _jpeg(r["analysed"])
                self.result.emit(idx, r)
            except Exception as e:
                self.failed.emit(idx, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
        self.finished.emit()


def start(worker):
    th = QThread()
    worker.moveToThread(th)
    th.started.connect(worker.run)
    worker.finished.connect(th.quit)
    th.start()
    return th
