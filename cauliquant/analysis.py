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
from .classifier import load_card, network_evidence, probabilities
from .shape_classes import SHAPE_RGB, SHAPE_NAMES, BAND, shape_stats, number_stats
from . import qc
from .resources import DEFAULT_UNET

TARGET_UM_PX = 0.496
IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

DENSE_RGB = (47, 127, 168)
CAULI_RGB = (217, 119, 31)
CAULI_SOFT_RGB = (243, 186, 125)      # borderline, called cauliflower
DENSE_SOFT_RGB = (140, 190, 220)      # borderline, called dense
FRAME_RGB = (255, 215, 0)


@dataclass
class Settings:
    um_px_override: float = 0.0          # 0 = read from the image metadata
    resample: bool = True
    cutoff: float = 0.5                  # cauliflower if its probability is above this
    frame_rule: str = "half"
    inset_um: float = 50.0
    model_path: str = DEFAULT_UNET

    def as_dict(self):
        d = asdict(self)
        d["model_path"] = os.path.basename(self.model_path or "")
        return d


MORPH_EXT = IMAGE_EXT + (".pdf", ".xls", ".xlsx", ".csv")      # the morphology tab also reads Microtrac exports


def _natural(p):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p)]


def list_images(folder, ext=IMAGE_EXT, recursive=False):
    """Files of a folder with one of the extensions, sorted naturally; with recursive=True also every sub-folder.

    Pictures of Microtrac report pages are left out when the report PDF they were made from is in the list too,
    so its particles are not counted twice.
    """
    found = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            found += [os.path.join(root, f) for f in files]
    else:
        found = [os.path.join(folder, f) for f in os.listdir(folder)]
    found = [f for f in found if os.path.isfile(f) and f.lower().endswith(ext)
             and not os.path.basename(f).startswith((".", "~$"))]
    stems = {os.path.splitext(os.path.basename(f))[0].lower() for f in found if f.lower().endswith(".pdf")}
    keep = []
    for f in found:
        st = os.path.splitext(os.path.basename(f))[0].lower()
        if f.lower().endswith(IMAGE_EXT) and "_viewerreport" in st and st.split("_viewerreport")[0] in stems:
            continue
        keep.append(f)
    return sorted(keep, key=lambda f: _natural(os.path.relpath(f, folder)))


def sample_of(name):
    """Sample name: the part of the image name before _x<magnification>, e.g. 384_x100_0031 -> 384,
    or, without one, the name less a trailing field number, e.g. 381_0016 -> 381."""
    import re
    i = name.lower().find("_x")
    if i > 0:
        return name[:i]
    m = re.match(r"(.+?)[_ -]\d{3,}$", name)
    return m.group(1) if m else name


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
                meta=meta, magnification=meta.get("Magnification", ""),
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


def _info(p):
    d = {k: p[k] for k in ("um_px", "native_um_px", "um_source", "scale", "native_size", "magnification", "voltage",
                           "signal")}
    m = p.get("meta", {})
    d["scan"] = m.get("ScanSpeed", "")
    d["spot"] = m.get("Condencer1", "")
    return d


def classify_image(path, s: Settings, progress=None):
    """Cauliflower vs dense for one SEM image. Returns a result dict (images as RGB arrays).

    Every particle gets a cauliflower probability. The headline result is the
    probability-weighted share of particle area: each particle adds its area
    times its probability. Counting only the particles called cauliflower
    under-counts when the classifier misses more than it over-calls, which is
    the usual case for a minority class; the weighted share does not.
    """
    step = progress or (lambda f, msg="": None)
    p = prepare(path, s)
    gray, um, top = p["gray"], p["um_px"], p["top"]
    card = load_card(s.model_path)
    lo, hi = card.get("borderline", [0.35, 0.65])
    step(0.05, "segmenting particles")
    lab = segment(gray, um)
    n = int(lab.max())
    step(0.15, "measuring particles")
    rows = particle_table(gray, lab, um)
    step(0.25, "running the network")
    net = load_unet(s.model_path)
    probs = net.predict_probs(gray, progress=lambda f: step(0.25 + 0.6 * f, "running the network"))
    score, pix3, pix5 = network_evidence(probs, lab)
    for r in rows:
        i = r["id"]
        r["score"] = round(float(score[i]), 4)
        r["pix0.3"] = round(float(pix3[i]), 4)
        r["pix0.5"] = round(float(pix5[i]), 4)
    prob = probabilities(rows, card)
    step(0.88, "counting")
    hidden = set(np.unique(lab[top:]).tolist()) - {0} if top is not None else set()
    counted = counted_area(lab, um, s.frame_rule, s.inset_um, top)
    for r, pr in zip(rows, prob):
        i = r["id"]
        r["probability"] = round(float(pr), 3)
        if i in hidden:
            r["class"] = "excluded (text overlay)"
        elif r["ecd_um"] < MIN_ECD_UM:
            r["class"] = "excluded (< 10 um)"
        else:
            r["class"] = "cauliflower" if pr > s.cutoff else "dense"
        r["borderline"] = bool(lo < pr <= hi)
        c, lost = counted.get(i, (0.0, False))
        r["counted_um2"] = round(c, 2)
        r["counted"] = bool(c > 0 and r["class"] in ("dense", "cauliflower"))
    kept = [r for r in rows if r["counted"]]
    area = np.array([r["counted_um2"] for r in kept], float)
    pk = np.array([r["probability"] for r in kept], float)
    vol = np.array([np.pi / 6 * r["ecd_um"] ** 3 for r in kept], float)
    call = pk > s.cutoff
    tot = float(area.sum())
    pct = lambda x: 100 * float(x) / tot if tot else float("nan")
    use_prob = bool(card.get("classifier"))
    a_c = float((area * pk).sum()) if use_prob else float(area[call].sum())
    v_c = float((vol * pk).sum()) if use_prob else float(vol[call].sum())
    ecd = lambda c: [r["ecd_um"] for r in kept if r["class"] == c]
    box = frame_box(lab.shape, um, s.inset_um, top)
    colours = {}
    for r in kept:
        if r["class"] == "cauliflower":
            colours[r["id"]] = CAULI_SOFT_RGB if r["borderline"] else CAULI_RGB
        else:
            colours[r["id"]] = DENSE_SOFT_RGB if r["borderline"] else DENSE_RGB
    analysed = _paint(gray, lab, colours, set(colours), box, top)
    stats = {
        "cauliflower_area_pct": pct(a_c),
        "cauliflower_vol_pct": 100 * v_c / float(vol.sum()) if vol.sum() else float("nan"),
        "calls_area_pct": pct(area[call].sum()),
        "firm_area_pct": pct(area[pk > hi].sum()),
        "with_borderline_area_pct": pct(area[pk > lo].sum()),
        "borderline_area_pct": pct(area[(pk > lo) & (pk <= hi)].sum()),
        "n_counted": len(kept),
        "n_cauliflower": int(call.sum()),
        "n_dense": int((~call).sum()),
        "n_borderline": int(((pk > lo) & (pk <= hi)).sum()),
        "cauliflower_area_um2": a_c, "dense_area_um2": tot - a_c,
        "median_ecd_cauliflower_um": float(np.median(ecd("cauliflower"))) if ecd("cauliflower") else float("nan"),
        "median_ecd_dense_um": float(np.median(ecd("dense"))) if ecd("dense") else float("nan"),
        "n_segmented": n,
        "n_small": sum(r["class"].startswith("excluded (<") for r in rows),
        "n_outside_frame": sum(1 for r in rows if r["class"] in ("dense", "cauliflower") and not r["counted"]),
        "n_lost_border": sum(1 for v in counted.values() if v[1]),
        "frame_um": ((box[3] - box[1]) * um, (box[2] - box[0]) * um),
    }
    info = _info(p)
    fm = qc.field_metrics(gray, lab, um)
    stats.update({k: round(v, 3) for k, v in fm.items()})
    warnings = qc.settings_check(p.get("meta", {}), info, card) + qc.field_check(fm, len(kept), card)
    if not use_prob:
        warnings.append("This model file has no model card: the raw network score is used and the result is not "
                        "validated.")
    step(1.0, "done")
    return dict(kind="classification", name=p["name"], path=path, sample=sample_of(p["name"]),
                original=np.stack([gray] * 3, -1), analysed=analysed, rows=rows, stats=stats, warnings=warnings,
                image_info=info, settings=s.as_dict(), version=__version__,
                model=dict(name=card.get("name", ""), estimator="probability-weighted area" if use_prob else "particle calls",
                           sha256=card.get("network", {}).get("sha256", ""), validated=card.get("validated", {}),
                           validation=card.get("validation", {}), borderline=[lo, hi]))


# ---------------------------------------------------------------- morphology

def morphology_input(path, s: Settings, progress=None):
    """Morphology profile of one file: an SEM image, or a Microtrac export (particle report as PDF or page
    picture, particle table .xls, size report PDF). See microtrac.py for what each one gives."""
    from . import microtrac
    step = progress or (lambda f, msg="": None)
    r = microtrac.analyse(path, s.as_dict(), step)
    if r is not None:
        return r
    rgb = np.asarray(Image.open(path).convert("RGB"), np.int16)
    colour = float(((rgb.max(-1) - rgb.min(-1)) > 40).mean())
    white = float((rgb.min(-1) > 235).mean())
    if colour > 0.002 or white > 0.6:
        raise ValueError("Not an SEM micrograph (it is in colour or mostly white) and not a page of a Microtrac "
                         "particle report.")
    return morphology_image(path, s, progress)


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
    info = _info(p)
    card = load_card(s.model_path)
    fm = qc.field_metrics(gray, lab, um)
    stats.update({k: round(v, 3) for k, v in fm.items()})
    warnings = qc.settings_check(p.get("meta", {}), info, card) + qc.field_check(fm, len(rows), card)
    step(1.0, "done")
    return dict(kind="morphology", source="sem", name=p["name"], path=path, sample=sample_of(p["name"]),
                original=np.stack([gray] * 3, -1), analysed=analysed, rows=rows, stats=stats, warnings=warnings,
                image_info=info, settings=dict(s.as_dict(), frame_rule=rule), version=__version__)


# ------------------------------------------------------------------ batches

def summarise_classification(results):
    """Per sample: pooled cauliflower area % over its fields, field spread and a 95 % interval over fields."""
    by = {}
    for r in results:
        by.setdefault(r["sample"], []).append(r)
    out = []
    for s, rs in by.items():
        a_c = sum(r["stats"]["cauliflower_area_um2"] for r in rs)
        a_d = sum(r["stats"]["dense_area_um2"] for r in rs)
        tot = a_c + a_d
        f = np.array([r["stats"]["cauliflower_area_pct"] for r in rs], float)
        f = f[np.isfinite(f)]
        sd = float(f.std(ddof=1)) if len(f) > 1 else float("nan")
        w = lambda key: (sum(r["stats"][key] * (r["stats"]["cauliflower_area_um2"] + r["stats"]["dense_area_um2"])
                             for r in rs if np.isfinite(r["stats"][key])) / tot) if tot else float("nan")
        ci = qc.field_interval(rs)
        out.append(dict(sample=s, fields=len(rs), particles=sum(r["stats"]["n_counted"] for r in rs),
                        cauliflower_area_pct=100 * a_c / tot if tot else float("nan"),
                        ci_low_pct=ci[0] if ci else float("nan"), ci_high_pct=ci[1] if ci else float("nan"),
                        field_mean_pct=float(f.mean()) if len(f) else float("nan"), field_sd_pct=sd,
                        fields_for_5pp=qc.fields_needed(sd, 5.0),
                        calls_area_pct=w("calls_area_pct"), firm_area_pct=w("firm_area_pct"),
                        with_borderline_area_pct=w("with_borderline_area_pct"),
                        warnings=sum(1 for r in rs if r.get("warnings"))))
    return out


SOURCE_ORDER = {"table": 0, "gallery": 1, "sem": 2, "size": 3}
SOURCE_NAME = {"sem": "SEM images", "gallery": "Microtrac report", "table": "Microtrac table", "size": "Size (LD)"}


def sample_labels(results):
    """Name each result's sample for the summary: the batch, with the measurement date added when the same batch
    was measured on more than one day (e.g. 382 (2026-05-06) and 382 (2026-11-06))."""
    runs = {}
    for r in results:
        runs.setdefault(r["sample"], set()).add(r.get("run", ""))
    for r in results:
        many = len(runs[r["sample"]] - {""}) > 1
        r["label"] = f"{r['sample']} ({r['run']})" if many and r.get("run") else r["sample"]
    return results


def summarise_morphology(results):
    """Per sample and source: C1-C5 shares from all its files' particles pooled.

    SEM fields and Microtrac tables carry sizes, so they give shares by number, by volume and in the band.
    Microtrac report silhouettes carry none and give shares by number. Size reports have no particles and
    are returned with their percentiles (the mean of the reports when there are several).
    """
    sample_labels(results)
    by = {}
    for r in results:
        by.setdefault((r["label"], r.get("source", "sem")), []).append(r)
    out = []
    for (s, src), rs in sorted(by.items(), key=lambda kv: (_natural(kv[0][0]), SOURCE_ORDER.get(kv[0][1], 9))):
        rows = [p for r in rs for p in r["rows"]]
        if src == "size":
            keys = {k for r in rs for k in r["stats"] if k.startswith("LD_")}
            st = {k: float(np.nanmean([r["stats"].get(k, np.nan) for r in rs])) for k in keys}
            st["n_counted"] = 0
        elif src == "gallery":
            st = number_stats(rows)
        else:
            st = shape_stats(rows)
        out.append(dict(sample=s, batch=rs[0]["sample"], run=rs[0].get("run", ""), source=src,
                        source_name=SOURCE_NAME.get(src, src), fields=len(rs),
                        pages=sum(r.get("pages", 1) for r in rs), **st))
    return out


def batch_table(results):
    """One row per sample with what each source gives: Microtrac table shares (or report shares, or SEM shares)
    and the laser diffraction percentiles."""
    rows = {}
    for s in summarise_morphology(results):
        row = rows.setdefault(s["sample"], {"sample": s["sample"], "sources": []})
        row["sources"].append(s["source"])
        if s["source"] == "size":
            row.update({k: v for k, v in s.items() if k.startswith("LD_")})
            continue
        if "shape_source" in row and SOURCE_ORDER[s["source"]] >= SOURCE_ORDER[row["shape_source"]]:
            continue
        row["shape_source"] = s["source"]
        row.update({k: v for k, v in s.items() if k.startswith(("num_pct", "vol_pct", "band_vol_pct", "median_"))
                    or k in ("n_counted", "n_band", "D50_num_um")})
    return sorted(rows.values(), key=lambda r: _natural(r["sample"]))
