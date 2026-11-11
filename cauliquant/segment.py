"""Load SU3500 SEM images and split them into particles."""
import os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation

DEFAULT_UM_PX = 0.496
MIN_ECD_UM = 10.0


def sem_metadata(path):
    """SU3500 metadata as a dict, from the .txt next to the image or from TIFF tag 40092."""
    txt = os.path.splitext(path)[0] + ".txt"
    text = None
    if os.path.exists(txt):
        text = open(txt, encoding="latin-1").read()
    else:
        try:
            v = Image.open(path).tag_v2.get(40092)
            if isinstance(v, bytes):
                text = v.decode("utf-16-le", errors="ignore")
        except Exception:
            pass
    out = {}
    for line in (text or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def pixel_size_info(path):
    """(um per pixel, where it came from)."""
    meta = sem_metadata(path)
    try:
        return float(meta["PixelSize"]) / 1000.0, "SEM metadata"
    except (KeyError, ValueError):
        return DEFAULT_UM_PX, "default (no metadata found)"


def pixel_size_um(tif_path):
    """Read PixelSize (nm) from the SU3500 metadata of the image."""
    return pixel_size_info(tif_path)[0]


def data_bar_row(a):
    """First row of the burned-in data bar (dark band at the bottom)."""
    for y in range(int(a.shape[0] * 0.85), a.shape[0]):
        if (a[y] < 15).mean() > 0.6:
            return y
    return a.shape[0]


def overlay_top(path):
    """First row hidden by a transparent text overlay, or None.

    Some exports (the Feb 2026 fluidized-bed fields) print the data bar as
    white text over the image instead of in a dark band below it. The letters
    would otherwise be picked up as particles.
    """
    a = np.array(Image.open(path).convert("L"))
    if data_bar_row(a) < a.shape[0]:
        return None
    lo = int(a.shape[0] * 0.88)
    rows = np.where((a[lo:, : a.shape[1] // 2] == 255).mean(1) > 0.15)[0]
    if len(rows) < 20:
        return None
    return lo + int(rows.min()) - 50          # 50 rows up also clears the scale ticks


def load_sem(path):
    """Grey image without the data bar, and its pixel size in um."""
    a = np.array(Image.open(path).convert("L"))
    return a[: data_bar_row(a)], pixel_size_um(path)


def segment(gray, um_px=DEFAULT_UM_PX, h=4.0):
    """Label map of particles; touching particles split by a mild watershed."""
    s = filters.gaussian(gray.astype(float), 1, preserve_range=True)
    m = s > filters.threshold_otsu(s)
    m = morphology.remove_small_holes(m, int(200 / um_px ** 2))
    m = morphology.binary_opening(m, morphology.disk(1))
    m = morphology.remove_small_objects(m, 20)
    d = filters.gaussian(ndi.distance_transform_edt(m), 2)
    markers = measure.label(morphology.h_maxima(d, h) & m)
    lab = segmentation.watershed(-d, markers, mask=m)
    rest = measure.label(m & (lab == 0))
    lab[rest > 0] = rest[rest > 0] + lab.max()
    lab, _, _ = segmentation.relabel_sequential(lab)
    return lab


def min_area_px(um_px):
    return np.pi * (MIN_ECD_UM / 2) ** 2 / um_px ** 2
