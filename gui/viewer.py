"""Zoomable image view that keeps the same zoom and position when the image is swapped."""
import io

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFrame


def to_array(img):
    if img is None:
        return None
    if isinstance(img, (bytes, bytearray)):
        return np.array(Image.open(io.BytesIO(img)).convert("RGB"))
    return img


def to_pixmap(img):
    a = np.ascontiguousarray(to_array(img))
    if a.ndim == 2:
        a = np.ascontiguousarray(np.stack([a] * 3, -1))
    h, w, _ = a.shape
    q = QImage(a.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self._item)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#2B2F36"))
        self.setFrameShape(QFrame.NoFrame)
        self._size = None
        self._placeholder = self.scene().addText("")
        self._placeholder.setDefaultTextColor(QColor("#C9D1D9"))
        self.set_placeholder("Open an SEM image or a folder of images to start")

    def set_placeholder(self, text):
        self._item.setPixmap(QPixmap())
        self._size = None
        self._placeholder.setPlainText(text)
        self._placeholder.setVisible(True)
        f = self._placeholder.font()
        f.setPointSize(12)
        self._placeholder.setFont(f)
        self.resetTransform()
        self.scene().setSceneRect(self._placeholder.boundingRect())
        self.centerOn(self._placeholder)

    def set_image(self, img, keep_view=True):
        pm = to_pixmap(img)
        same = keep_view and self._size == (pm.width(), pm.height())
        self._placeholder.setVisible(False)
        self._item.setPixmap(pm)
        self.scene().setSceneRect(QRectF(pm.rect()))
        if not same:
            self._size = (pm.width(), pm.height())
            self.fit()

    def fit(self):
        if self._size:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def wheelEvent(self, e):
        if not self._size:
            return
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._size and self.transform().m11() <= self._fit_scale() * 1.01:
            self.fit()

    def _fit_scale(self):
        if not self._size:
            return 1
        vw, vh = self.viewport().width(), self.viewport().height()
        return min(vw / self._size[0], vh / self._size[1])
