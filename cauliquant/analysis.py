"""Analyses behind the app's two tabs, as plain functions that return results in memory.

classify_image()   cauliflower vs dense particles, area % inside the counting frame
morphology_image() five Microtrac shape classes (C1-C5) on the SEM particle outlines
summarise_*()      pool several images (fields) by sample

Both start the same way: read the SEM image, drop the data bar, read the pixel
size from the SU3500 metadata, and resample to the model's pixel size
(0.496 um/px, x100 at 2560 px) when the image was taken at another one.
"""
import os
from dataclasses import dataclass, field, asdict

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import segmentation, transform

from . import __version__
from .segment import data_bar_row, overlay_top, pixel_size_info, sem_metadata, segment, MIN_ECD_UM
from .features import particle_table
from .frame import counted_area, frame_box
from .resources import DEFAULT_UNET

TARGET_UM_PX = 0.496
IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

DENSE_RGB = (47, 127, 168)
CAULI_RGB = (217, 119, 31)
FRAME_RGB = (255, 215, 0)
SHAPE_RGB = [(76, 120, 168), (114, 183, 178), (238, 202, 59), (178, 121, 162), (228, 87, 86)]
SHAPE_NAMES = ["C1 compact", "C2 elongated smooth", "C3 blocky rough", "C4 elongated concave",
               "C5 branched / agglomerate"]
BAND = (30.0, 100.0)


@dataclass
class Settings:
    um_px_override: float = 0.0          # 0 = read from the image metadata
    resample: bool = True
    cutoff: float = 0.5                  # cauliflower if its score is above this
    frame_rule: str = "half"
    inset_um: float = 50.0
    model_path: str = DEFAULT_UNET

    def as_dict(self):
        d = asdict(self)
        d["model_path"] = os.path.basename(self.model_path or "")
        return d


def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXT) and not f.startswith("."))


def sample_of(name):
    """Sample name: the part of the image name before _x<magnification>, e.g. 384_x100_0031 -> 384."""
    i = name.lower().find("_x")
    return name[:i] if i > 0 else name


# ---------------------------------------------------------------- image input

def prepare(path, s: Settings):
    a = np.array(Image.open(path).convert("L"))
    bar = data_bar_row(a)
    gray = a[:bar]
    top = overlay_top(path)
    if s.um_px_override and s.um_px_override > 0:
        um, src = float(s.um_px_override), "entered by user"
    else:
        um, src = pixel_size_info(path)
    native_um = um
    scale = 1.0
    if s.resample and abs(um / TARGET_UM_PX - 1) > 0.05:
        scale = um / TARGET_UM_PX
        gray = transform.rescale(gray, scale, order=1, anti_aliasing=scale < 1,
                                 preserve_range=True).round().clip(0, 255).astype(np.uint8)
        um = TARGET_UM_PX
        if top is not None:
            top = int(top * scale)
    meta = sem_metadata(path)
    return dict(path=path, name=os.path.splitext(os.path.basename(path))[0], gray=gray, um_px=um,
                native_um_px=native_um, um_source=src, scale=scale, top=top,
                native_size=(a.shape[1], a.shape[0]),
                magnification=meta.get("Magnification", ""),
                voltage=meta.get("AcceleratingVoltage", meta.get("Condition", ""))[:40],
                signal=meta.get("SignalName", ""))


def _label_mean(values, lab, n):
    s = np.bincount(lab.ravel(), weights=values.ravel(), minlength=n + 1)
    c = np.bincount(lab.ravel(), minlength=n + 1)
    return s / np.maximum(c, 1)


def _paint(gray, lab, colour_of_id, counted_ids, box, top=None):
    """RGB overlay: counted particles tinted by class, others outlined in grey, frame in yellow."""
    rgb = np.stack([gray] * 3, -1).astype(np.float32)
    n = int(lab.max())
    lut = np.zeros((n + 1, 3), np.float32)
    has = np.zeros(n + 1, bool)
    for i, c in colour_of_id.items():
        if i in counted_ids:
            lut[i] = c
            has[i] = True
    sel = has[lab]
    rgb[sel] = 0.5 * rgb[sel] + 0.5 * lut[lab[sel]]
    edges = segmentation.find_boundaries(lab, mode="inner")
    e_cnt = edges & sel
    rgb[e_cnt] = lut[lab[e_cnt]] * 0.75
    rgb[edges & ~sel & (lab > 0)] = (150, 150, 150)
    out = rgb.clip(0, 255).astype(np.uint8)
    y0, x0, y1, x1 = box
    w = max(2, int(round(min(out.shape[:2]) / 600)))
    out[y0:y0 + w, x0:x1] = FRAME_RGB
    out[y1 - w:y1, x0:x1] = FRAME_RGB
    out[y0:y1, x0:x0 + w] = FRAME_RGB
    out[y0:y1, x1 - w:x1] = FRAME_RGB
    if top is not None:
        out[top:] = (out[top:] * 0.35).astype(np.uint8)
    return out


# ------------------------------------------------------------ classification

_model_cache = {}


def load_unet(path):
    if path not in _model_cache:
        from .infer import UNetONNX
        _model_cache.clear()
        _model_cache[path] = UNetONNX(path)
    return _model_cache[path]


def classify_image(path, s: Settings, progress=None):
    """Cauliflower vs dense for one SEM image. Returns a result dict (images as RGB arrays)."""
    step = progress or (lambda f, msg="": None)
    p = prepare(path, s)
    gray, um, top = p["gray"], p["um_px"], p["top"]
    step(0.05, "segmenting particles")
    lab = segment(gray, um)
    n = int(lab.max())
    rows = particle_table(gray, lab, um)
    step(0.25, "running the network")
    net = load_unet(s.model_path)
    probs = net.predict_probs(gray, progress=lambda f: step(0.25 + 0.6 * f, "running the network"))
    pc = _label_mean(probs[2], lab, n)
    pd_ = _label_mean(probs[1], lab, n)
    step(0.88, "counting")
    hidden = set(np.unique(lab[top:]).tolist()) - {0} if top is not None else set()
    counted = counted_area(lab, um, s.frame_rule, s.inset_um, top)
    for r in rows:
        i = r["id"]
        r["score"] = round(float(pc[i] / max(pc[i] + pd_[i], 1e-6)), 3)
        if i in hidden:
            r["class"] = "excluded (text overlay)"
        elif r["ecd_um"] < MIN_ECD_UM:
            r["class"] = "excluded (< 10 um)"
        else:
            r["class"] = "cauliflower" if r["score"] > s.cutoff else "dense"
        c, lost = counted.get(i, (0.0, False))
        r["counted_um2"] = round(c, 2)
        r["counted"] = bool(c > 0 and r["class"] in ("dense", "cauliflower"))
    kept = [r for r in rows if r["counted"]]
    a_c = sum(r["counted_um2"] for r in kept if r["class"] == "cauliflower")
    a_d = sum(r["counted_um2"] for r in kept if r["class"] == "dense")
    vol = lambda r: np.pi / 6 * r["ecd_um"] ** 3
    v_c = sum(vol(r) for r in kept if r["class"] == "cauliflower")
    v_d = sum(vol(r) for r in kept if r["class"] == "dense")
    ecd = lambda c: [r["ecd_um"] for r in kept if r["class"] == c]
    box = frame_box(lab.shape, um, s.inset_um, top)
    colours = {r["id"]: (CAULI_RGB if r["class"] == "cauliflower" else DENSE_RGB) for r in kept}
    analysed = _paint(gray, lab, colours, set(colours), box, top)
    tot = a_c + a_d
    stats = {
        "cauliflower_area_pct": 100 * a_c / tot if tot else float("nan"),
        "cauliflower_vol_pct": 100 * v_c / (v_c + v_d) if v_c + v_d else float("nan"),
        "n_counted": len(kept),
        "n_cauliflower": sum(r["class"] == "cauliflower" for r in kept),
        "n_dense": sum(r["class"] == "dense" for r in kept),
        "cauliflower_area_um2": a_c, "dense_area_um2": a_d,
        "median_ecd_cauliflower_um": float(np.median(ecd("cauliflower"))) if ecd("cauliflower") else float("nan"),
        "median_ecd_dense_um": float(np.median(ecd("dense"))) if ecd("dense") else float("nan"),
        "n_segmented": n,
        "n_small": sum(r["class"].startswith("excluded (<") for r in rows),
        "n_outside_frame": sum(1 for r in rows if r["class"] in ("dense", "cauliflower") and not r["counted"]),
        "n_lost_border": sum(1 for v in counted.values() if v[1]),
        "frame_um": ((box[3] - box[1]) * um, (box[2] - box[0]) * um),
    }
    step(1.0, "done")
    return dict(kind="classification", name=p["name"], path=path, sample=sample_of(p["name"]),
                original=np.stack([gray] * 3, -1), analysed=analysed, rows=rows, stats=stats,
                image_info={k: p[k] for k in ("um_px", "native_um_px", "um_source", "scale", "native_size",
                                              "magnification", "voltage", "signal")},
                settings=s.as_dict(), version=__version__)


# ---------------------------------------------------------------- morphology

def morphology_image(path, s: Settings, progress=None):
    """C1-C5 shape classes for every counted particle of one SEM image."""
    from .shape_classes import ShapeClasses, FEAT, sem_descriptors
    step = progress or (lambda f, msg="": None)
    sc = ShapeClasses()
    p = prepare(path, s)
    gray, um, top = p["gray"], p["um_px"], p["top"]
    step(0.05, "segmenting particles")
    lab = segment(gray, um)
    rule = s.frame_rule if s.frame_rule in ("edge", "half", "centroid") else "half"
    counted = counted_area(lab, um, rule, s.inset_um, top)
    hidden = set(np.unique(lab[top:]).tolist()) - {0} if top is not None else set()
    ids = {i for i, (c, _) in counted.items() if c > 0 and i not in hidden}
    step(0.35, "measuring shapes")
    desc = sem_descriptors(lab, um, ids)
    rows = []
    for i, d in desc.items():
        if d["Da"] < max(sc.min_da, MIN_ECD_UM):
            continue
        rows.append(dict(id=int(i), **{k: float(v) for k, v in d.items()}))
    if rows:
        X = np.array([[r[k] for k in FEAT] for r in rows])
        cls = sc.predict(X)
        for r, c in zip(rows, cls):
            r["shape_class"] = int(c) + 1
    step(0.9, "drawing")
    box = frame_box(lab.shape, um, s.inset_um, top)
    colours = {r["id"]: SHAPE_RGB[r["shape_class"] - 1] for r in rows if r.get("shape_class", 0) > 0}
    analysed = _paint(gray, lab, colours, set(colours), box, top)
    stats = shape_stats(rows)
    stats["n_segmented"] = int(lab.max())
    stats["frame_um"] = ((box[3] - box[1]) * um, (box[2] - box[0]) * um)
    step(1.0, "done")
    return dict(kind="morphology", name=p["name"], path=path, sample=sample_of(p["name"]),
                original=np.stack([gray] * 3, -1), analysed=analysed, rows=rows, stats=stats,
                image_info={k: p[k] for k in ("um_px", "native_um_px", "um_source", "scale", "native_size",
                                              "magnification", "voltage", "signal")},
                settings=dict(s.as_dict(), frame_rule=rule), version=__version__)


def shape_stats(rows):
    rows = [r for r in rows if r.get("shape_class", 0) > 0]
    out = {"n_counted": len(rows)}
    if not rows:
        return out
    da = np.array([r["Da"] for r in rows])
    vol = np.array([r["Volume"] for r in rows])
    cls = np.array([r["shape_class"] for r in rows])
    band = (da >= BAND[0]) & (da < BAND[1])
    for k in range(1, 6):
        out[f"num_pct_C{k}"] = 100 * float((cls == k).mean())
        out[f"vol_pct_C{k}"] = 100 * float(vol[cls == k].sum() / vol.sum())
        out[f"band_vol_pct_C{k}"] = 100 * float(vol[band & (cls == k)].sum() / vol[band].sum()) if band.any() else float("nan")
    out["n_band"] = int(band.sum())
    out["D50_num_um"] = float(np.median(da))
    for k in ("Circularity", "Solidity", "Convexity", "W/L Aspect Ratio", "Extent"):
        out[f"median_{k}"] = float(np.median([r[k] for r in rows]))
    return out


# ------------------------------------------------------------------ batches

def summarise_classification(results):
    """Per sample: pooled cauliflower area % over its fields, plus field mean and SD."""
    by = {}
    for r in results:
        by.setdefault(r["sample"], []).append(r)
    out = []
    for s, rs in by.items():
        a_c = sum(r["stats"]["cauliflower_area_um2"] for r in rs)
        a_d = sum(r["stats"]["dense_area_um2"] for r in rs)
        f = np.array([r["stats"]["cauliflower_area_pct"] for r in rs], float)
        f = f[np.isfinite(f)]
        out.append(dict(sample=s, fields=len(rs), particles=sum(r["stats"]["n_counted"] for r in rs),
                        cauliflower_area_pct=100 * a_c / (a_c + a_d) if a_c + a_d else float("nan"),
                        field_mean_pct=float(f.mean()) if len(f) else float("nan"),
                        field_sd_pct=float(f.std(ddof=1)) if len(f) > 1 else float("nan")))
    return out


def summarise_morphology(results):
    """Per sample: C1-C5 shares from all its fields' particles pooled."""
    by = {}
    for r in results:
        by.setdefault(r["sample"], []).append(r)
    out = []
    for s, rs in by.items():
        rows = [p for r in rs for p in r["rows"]]
        st = shape_stats(rows)
        out.append(dict(sample=s, fields=len(rs), **st))
    return out
