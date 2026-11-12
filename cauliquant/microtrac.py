"""Microtrac exports as inputs to morphology profiling.

Three kinds of file are read, next to SEM micrographs:

  particle gallery   the SYNC "Particles" viewer report, as a PDF or as page
                     pictures (JPG / PNG). Every particle is printed as a small
                     silhouette. The silhouettes are measured again here and
                     given the same five shape classes (C1-C5). The report has
                     no sizes (every silhouette is scaled to the same cell) and
                     holds only the first particles of the viewer list, so the
                     result is a share by number of the particles shown.
  particle table     the "(All particles).xls" export (or the same table saved
                     as .xlsx or .csv): Microtrac's own descriptors for every
                     particle, with sizes. Gives shares by number, by volume
                     and in the 30-100 um band.
  size report        the laser diffraction "Particle Size Analysis", as the
                     PDF report or the <batch>.csv export: percentiles and
                     mean sizes.

Silhouette descriptors follow Microtrac's definitions (W/L = minimum Feret /
maximum Feret, Extent = area / (min Feret x max Feret)). A 50 px silhouette
gives a slightly longer outline than Microtrac measures on the camera image, so
a linear correction per descriptor, fitted on reports paired with their particle
tables (models/microtrac_gallery.json), puts them on the Microtrac scale.
"""
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage import measure, transform

from . import __version__
from .resources import resource
from .shape_classes import ShapeClasses, FEAT, SHAPE_RGB, shape_stats, number_stats

GALLERY_CAL = resource("models", "microtrac_gallery.json")
MICROTRAC_EXT = (".pdf", ".xls", ".xlsx", ".csv")
EXTRA = ["Sphericity", "Compactness", "Roundness", "Ellipse Ratio", "Ellipticity", "Concavity", "Transparency"]
N_EXEMPLARS = 12
CELL_PX = 50                      # silhouette size the correction was fitted at
WHITE = 235
INK = 200
SHAPE_SHORT = ["C1 compact", "C2 elongated smooth", "C3 blocky rough", "C4 elongated concave", "C5 branched"]
SOURCE_TEXT = {"sem": "SEM", "gallery": "MT report", "table": "MT table", "size": "MT size"}


# ------------------------------------------------------------------ names

def sample_name(path):
    """Batch name for a Microtrac file.

    Exports are named after the SOP and a time stamp, so the batch is the folder
    they sit in ("382/H0131..._20261106_110724(All particles).pdf" -> 382). A
    file with a name of its own ("382.pdf", "382.csv") takes the folder's name too
    when the folder is a batch folder (it holds a time-stamped export) or carries a
    longer form of the same name ("collette 387-392/387-392.csv" -> "collette
    387-392"); otherwise it keeps its own name.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    d = os.path.dirname(os.path.abspath(path))
    for _ in range(3):
        b = os.path.basename(d)
        if b and not re.match(r"(?i)new folder|pages?$|images?$|photos?$|pictures?$|viewer", b):
            break
        d = os.path.dirname(d)
    folder = os.path.basename(d)
    if re.search(r"_\d{8}_\d{6}", stem) or "(all particles)" in stem.lower():
        return folder or stem
    stem = re.sub(r"(?i)[ _-]*(viewerreport)?[ _-]*page[ _-]*\d+$", "", stem) or stem
    if folder and stem != folder and (stem.lower() in folder.lower() or _batch_folder(d)):
        return folder
    return stem


def _batch_folder(d):
    """True when the folder holds a time-stamped Microtrac export, i.e. it is one batch's folder."""
    try:
        return any("(all particles)" in f.lower() for f in os.listdir(d))
    except OSError:
        return False


def run_date(path, text=""):
    """Measurement date (YYYY-MM-DD) from the export's time stamp, or from the report text."""
    m = re.search(r"_(20\d{2})(\d{2})(\d{2})_\d{6}", os.path.basename(path))
    if m:
        return "-".join(m.groups())
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", text or "")
    return m.group(1) if m else ""


# ------------------------------------------------------- silhouette shapes

def outline_descriptors(mask):
    """Scale-free Microtrac-style descriptors of one filled silhouette (bool array), before correction."""
    cs = measure.find_contours(np.pad(mask, 1).astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    if len(c) < 8:
        return None
    P = np.hypot(*np.diff(c, axis=0).T).sum()
    x, y = c[:, 1], c[:, 0]
    A = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    hull = ConvexHull(c)
    h = hull.points[hull.vertices]
    HP = np.hypot(*np.diff(np.vstack([h, h[:1]]), axis=0).T).sum()
    L = np.hypot(h[:, None, 0] - h[None, :, 0], h[:, None, 1] - h[None, :, 1]).max()      # maximum Feret
    e = np.roll(h, -1, axis=0) - h
    n = np.hypot(e[:, 0], e[:, 1])
    ok = n > 0
    nv = np.stack([-e[ok, 1], e[ok, 0]], 1) / n[ok, None]
    proj = h @ nv.T
    W = (proj.max(0) - proj.min(0)).min()                                                # minimum Feret
    if P <= 0 or L <= 0 or W <= 0:
        return None
    return {"Circularity": 4 * np.pi * A / P ** 2, "Solidity": A / hull.volume, "Convexity": HP / P,
            "W/L Aspect Ratio": W / L, "Extent": A / (W * L)}


def load_calibration(path=GALLERY_CAL):
    try:
        c = json.load(open(path))
        assert c["features"] == FEAT
        return np.array(c["slope"], float), np.array(c["intercept"], float), c
    except (OSError, ValueError, KeyError, AssertionError):
        return np.ones(len(FEAT)), np.zeros(len(FEAT)), {}


def _ink(rgb):
    return rgb.min(-1) < INK


def _largest_filled(m):
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return None
    big = int(np.argmax(np.bincount(lab.ravel())[1:])) + 1
    return ndi.binary_fill_holes(lab == big)


def _to_cell(mask):
    """Bring a silhouette to the size the correction was fitted at (longer side = 50 px)."""
    d = max(mask.shape)
    if abs(d - CELL_PX) <= 0.1 * CELL_PX:
        return mask
    g = transform.rescale(np.pad(mask, 2).astype(float), CELL_PX / d, order=1, anti_aliasing=d > CELL_PX)
    return g > 0.5


# ------------------------------------------------------------ gallery input

def looks_like_gallery(rgb):
    """A report page is mostly white paper with dark grey silhouettes; an SEM field is never mostly white, and a
    chart or a slide carries coloured marks rather than dark grey ones."""
    if float((rgb.min(-1) > WHITE).mean()) <= 0.6:
        return False
    a = rgb.astype(np.int16)
    grey_dark = (a.max(-1) < 110) & (a.max(-1) - a.min(-1) < 30)
    return float(grey_dark.mean()) > 0.03


def gallery_from_image(rgb):
    """Particles of one page picture: ([(mask, y, x)] in reading order, silhouette size in px).

    Pieces of ink a few pixels apart are grouped first, so a particle printed with a detached lobe stays one
    particle. Every silhouette is scaled to fill its cell, so the commonest group size is the cell size;
    header text and stray specks are smaller or larger, and a word is many letters of similar size where a
    particle is one main piece.
    """
    ink = _ink(rgb)
    gap = max(2, int(round(min(rgb.shape[:2]) / 400)))
    grp, n = ndi.label(ndi.binary_dilation(ink, iterations=gap), structure=np.ones((3, 3)))
    sl = ndi.find_objects(grp)
    H, W = rgb.shape[:2]
    if not sl:
        raise ValueError("No particle silhouettes found in this picture.")
    dims = np.array([max(s[0].stop - s[0].start, s[1].stop - s[1].start) - 2 * gap for s in sl])
    ok = (dims >= 12) & (dims <= 0.15 * max(H, W))
    if ok.sum() < 3:
        raise ValueError("No particle silhouettes found in this picture.")
    hist = np.bincount(dims[ok])
    cell = float(np.argmax(np.convolve(hist, np.ones(5), "same")))
    parts = {}
    for i in np.nonzero(ok)[0]:
        if not 0.85 * cell <= dims[i] <= 1.15 * cell:
            continue
        s = sl[i]
        m = ink[s] & (grp[s] == i + 1)
        lab, k = ndi.label(m, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())[1:]
        if k == 0 or sizes.max() < 0.4 * sizes.sum():
            continue                                   # text: many letters, no main piece
        parts[i] = (ndi.binary_fill_holes(m), s[0].start, s[1].start, s[0].start + s[0].stop, s[1].start + s[1].stop)
    if len(parts) < 3:
        raise ValueError("No particle silhouettes found in this picture.")
    cell = float(max(dims[i] for i in parts))
    cy = {i: v[3] / 2 for i, v in parts.items()}
    cx = {i: v[4] / 2 for i, v in parts.items()}
    rows, cur = [], []
    for i in sorted(parts, key=lambda k: cy[k]):
        if cur and cy[i] - cy[cur[-1]] > 0.5 * cell:
            rows.append(cur)
            cur = []
        cur.append(i)
    if cur:
        rows.append(cur)
    out = [(parts[i][0], parts[i][1], parts[i][2]) for row in rows for i in sorted(row, key=lambda k: cx[k])]
    return out, cell


def _rgb_of(bitmap):
    a = bitmap.to_numpy()
    if a.ndim == 2:
        return np.stack([a] * 3, -1)
    if a.shape[2] == 1:
        return np.repeat(a, 3, 2)
    a = a[..., :3]
    mode = str(getattr(bitmap, "mode", "BGR"))
    return a[..., ::-1].copy() if mode.upper().startswith("BGR") else a.copy()


def _pdf_text(page):
    tp = page.get_textpage()
    try:
        return tp.get_text_bounded()
    finally:
        tp.close()


def pdf_kind(path):
    """'gallery', 'size' or None for a PDF, from the first page's text and pictures."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    pdf = pdfium.PdfDocument(path)
    try:
        page = pdf[0]
        text = _pdf_text(page)
        n_img = sum(1 for _ in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE], max_depth=3))
        if "Particle Size Analysis" in text or "%Tile" in text:
            return "size"
        if n_img >= 10:
            return "gallery"
        return None
    finally:
        pdf.close()


def gallery_from_pdf(path, step=None):
    """Silhouette pictures embedded in a viewer report, in report order, and the header fields."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    pdf = pdfium.PdfDocument(path)
    thumbs, head, pages = [], {}, len(pdf)
    try:
        for pi in range(pages):
            page = pdf[pi]
            if pi == 0:
                t = _pdf_text(page)
                for key, pat in (("date", r"Date time:\s*([^\r\n]+)"), ("sort", r"Sort View by:\s*([^\r\n]+)"),
                                 ("sop", r"SOP Name:\s*([^\r\n]+)")):
                    m = re.search(pat, t)
                    if m:
                        head[key] = m.group(1).strip()
            cells = []
            for o in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE], max_depth=3):
                b = (o.get_bounds if hasattr(o, "get_bounds") else o.get_pos)()
                a = _rgb_of(o.get_bitmap(render=False))
                if a.min() > WHITE or min(a.shape[:2]) < 8:
                    continue                                   # empty cell
                cells.append((-round(b[3]), b[0], a))
            cells.sort(key=lambda c: (c[0], c[1]))
            thumbs += [(c[2], pi + 1) for c in cells]
            if step:
                step(0.05 + 0.25 * (pi + 1) / pages, "reading the report")
    finally:
        pdf.close()
    return thumbs, head, pages


def _classify(masks, step=None):
    """Corrected descriptors and class for each silhouette; rows keep the order of masks (None where unusable)."""
    slope, icpt, _ = load_calibration()
    sc = ShapeClasses()
    rows = []
    for k, m in enumerate(masks):
        d = outline_descriptors(_to_cell(m)) if m is not None and m.sum() >= 12 else None
        if d is None:
            rows.append(None)
        else:
            x = np.clip(slope * np.array([d[f] for f in FEAT]) + icpt, 0.01, 1.0)
            r = {f: float(v) for f, v in zip(FEAT, x)}
            r["shape_class"] = int(sc.predict(x[None])[0]) + 1
            rows.append(r)
        if step and k % 100 == 0:
            step(0.3 + 0.55 * k / max(1, len(masks)), "measuring silhouettes")
    return rows


def _tint(canvas, mask, y, x, rgb):
    """Fill one silhouette with its class colour, darker along the outline."""
    h, w = mask.shape
    reg = canvas[y:y + h, x:x + w]
    edge = mask & ~ndi.binary_erosion(mask)
    c = np.array(rgb, float)
    reg[mask] = (0.2 * reg[mask] + 0.8 * c).astype(np.uint8)
    reg[edge] = (0.6 * c).astype(np.uint8)


def _result(path, source, original, analysed, rows, stats, note, settings, warnings=(), pages=1, run="", **extra):
    return dict(kind="morphology", source=source, name=os.path.splitext(os.path.basename(path))[0], path=path,
                sample=sample_name(path), batch=sample_name(path), run=run,
                original=original, analysed=analysed, rows=rows, stats=stats, warnings=list(warnings),
                note=note, pages=pages, image_info={}, settings=settings, version=__version__, **extra)


def _exemplars(thumbs, rows, seed=0):
    """Up to N_EXEMPLARS silhouette pictures per class, picked at random, for the class figure of the report."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(1, 6):
        idx = [i for i, r in enumerate(rows) if r is not None and r["shape_class"] == k]
        if idx:
            pick = rng.choice(idx, size=min(N_EXEMPLARS, len(idx)), replace=False)
            out[k] = [np.ascontiguousarray(thumbs[i]) for i in sorted(pick)]
    return out


GALLERY_NOTE = ("A viewer report prints only part of the sample (the first particles of the viewer list, the widest "
                "ones when it is sorted by FWidth) and carries no sizes, so these are shares by number of the "
                "particles shown. For shares by volume and in the 30–100 µm band, open the particle table (.xls).")


def gallery_pdf_result(path, settings, step):
    thumbs, head, pages = gallery_from_pdf(path, step)
    if not thumbs:
        raise ValueError("This PDF has no particle silhouettes.")
    masks = [_largest_filled(_ink(a)) for a, _ in thumbs]
    rows_all = _classify(masks, step)
    step(0.88, "drawing")
    n = len(thumbs)
    cell = max(max(a.shape[:2]) for a, _ in thumbs) + 6
    cols = 40 if n > 400 else max(1, int(np.ceil(np.sqrt(n * 1.3))))
    nrow = int(np.ceil(n / cols))
    original = np.full((nrow * cell, cols * cell, 3), 255, np.uint8)
    place = []
    for k, (a, _) in enumerate(thumbs):
        y = (k // cols) * cell + (cell - a.shape[0]) // 2
        x = (k % cols) * cell + (cell - a.shape[1]) // 2
        original[y:y + a.shape[0], x:x + a.shape[1]] = a
        place.append((y, x))
    analysed = original.copy()
    rows = []
    for k, r in enumerate(rows_all):
        if r is None:
            continue
        _tint(analysed, masks[k], place[k][0], place[k][1], SHAPE_RGB[r["shape_class"] - 1])
        rows.append(dict(id=k + 1, page=thumbs[k][1], **r))
    stats = number_stats(rows)
    stats["n_shown"] = n
    note = f"Microtrac particle report, {n} particles on {pages} page{'s' if pages > 1 else ''}"
    if head.get("sort"):
        note += f", sorted by {head['sort']}"
    if head.get("date"):
        note += f", measured {head['date']}"
    note += ". " + GALLERY_NOTE
    return _result(path, "gallery", original, analysed, rows, stats, note, settings, pages=pages,
                   run=run_date(path, head.get("date", "")), exemplars=_exemplars([a for a, _ in thumbs], rows_all))


def gallery_image_result(path, rgb, settings, step):
    step(0.1, "finding silhouettes")
    parts, cell = gallery_from_image(rgb)
    rows_all = _classify([_largest_filled(m) for m, _, _ in parts], step)
    step(0.88, "drawing")
    analysed = rgb.copy()
    rows = []
    for k, r in enumerate(rows_all):
        if r is None:
            continue
        m, y, x = parts[k]
        _tint(analysed, m, y, x, SHAPE_RGB[r["shape_class"] - 1])
        rows.append(dict(id=k + 1, page=1, **r))
    stats = number_stats(rows)
    stats["n_shown"] = len(parts)
    note = (f"Picture of a Microtrac particle report page, {len(parts)} particles, silhouettes about "
            f"{cell:.0f} px across. " + GALLERY_NOTE)
    warn = []
    if cell < 30:
        warn.append(f"The silhouettes are only about {cell:.0f} px across; shape classes are less reliable below 30 px.")
    crops = []
    for m, y, x in parts:
        c = np.full(m.shape + (3,), 255, np.uint8)
        c[m] = rgb[y:y + m.shape[0], x:x + m.shape[1]][m]
        crops.append(c)
    return _result(path, "gallery", rgb, analysed, rows, stats, note, settings, warn, run=run_date(path),
                   exemplars=_exemplars(crops, rows_all))


# -------------------------------------------------------------- table input

def _cells(path):
    """Rows of cells (strings or numbers) of a text table (tab or comma separated) or an Excel workbook."""
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic.startswith(b"PK"):                                   # .xlsx
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            return [list(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
        finally:
            wb.close()
    if magic.startswith(b"\xd0\xcf\x11\xe0"):                  # binary .xls saved by Excel
        try:
            import xlrd
        except ImportError:
            raise ValueError("This is a binary Excel file; save it as .xlsx or .csv and open it again.")
        sh = xlrd.open_workbook(path).sheet_by_index(0)
        return [sh.row_values(i) for i in range(sh.nrows)]
    text = open(path, encoding="latin1").read()
    lines = text.splitlines()
    sep = "\t" if lines and lines[0].count("\t") >= lines[0].count(",") else ","
    import csv
    return list(csv.reader(lines, delimiter=sep))


def read_table(path):
    """Microtrac particle table (.xls tab text as exported, or .xlsx / .csv) as {column: float array}."""
    cells = [r for r in _cells(path) if r and any(c not in (None, "") for c in r)]
    if not cells:
        raise ValueError("This file is empty.")
    head = [str(c).strip() if c is not None else "" for c in cells[0]]
    head = [re.sub(r"\s+", " ", h) for h in head]
    if "Da" not in head or not all(f in head for f in FEAT):
        raise ValueError("This file is not a Microtrac particle table (columns Da, Circularity, Solidity, "
                         "Convexity, W/L Aspect Ratio and Extent are needed).")
    ida = head.index("Da")
    data = []
    for r in cells[1:]:
        try:
            float(r[ida])
        except (TypeError, ValueError, IndexError):
            continue                                              # units row
        row = []
        for j in range(len(head)):
            try:
                row.append(float(r[j]))
            except (TypeError, ValueError, IndexError):
                row.append(np.nan)
        data.append(row)
    a = np.array(data, float).reshape(-1, len(head))
    return {h: a[:, j] for j, h in enumerate(head) if h}


def is_size_csv(path):
    try:
        with open(path, encoding="latin1") as fh:
            t = fh.read(4000)
    except OSError:
        return False
    return "Percentiles" in t and "MV(um)" in t


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                 # older Pillow
        return ImageFont.load_default()


def share_chart(title, subtitle, panels):
    """White chart with one panel of five class bars per (heading, values) pair."""
    W, H = 1800, 980
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((60, 40), title, fill=(31, 78, 121), font=_font(44))
    d.text((60, 104), subtitle, fill=(87, 96, 106), font=_font(26))
    pw = (W - 120) // len(panels)
    for j, (heading, vals) in enumerate(panels):
        x0 = 60 + j * pw
        d.text((x0, 190), heading, fill=(31, 35, 40), font=_font(30))
        top = max([v for v in vals if v == v] or [1.0])
        scale = (pw - 150) / max(top, 1e-9)
        for k, v in enumerate(vals):
            y = 260 + k * 132
            d.text((x0, y), SHAPE_SHORT[k], fill=(87, 96, 106), font=_font(24))
            if v == v:
                d.rectangle([x0, y + 38, x0 + max(3, int(v * scale)), y + 88], fill=SHAPE_RGB[k])
                d.text((x0 + max(3, int(v * scale)) + 12, y + 46), f"{v:.1f} %", fill=(31, 35, 40), font=_font(28))
            else:
                d.text((x0, y + 46), "n/a", fill=(140, 149, 159), font=_font(28))
    return np.array(im)


def table_result(path, settings, step):
    step(0.1, "reading the table")
    t = read_table(path)
    sc = ShapeClasses()
    n_all = len(t["Da"])
    keep = t["Da"] >= sc.min_da
    X = np.column_stack([t[f][keep] for f in FEAT])
    cls = sc.predict(X) + 1
    step(0.6, "classifying")
    ids = t["Id"][keep] if "Id" in t else np.arange(1, keep.sum() + 1)
    vol = t["Volume"][keep] if "Volume" in t else np.pi / 6 * t["Da"][keep] ** 3
    extra = [e for e in EXTRA if e in t]
    E = np.column_stack([t[e][keep] for e in extra]) if extra else np.zeros((int(keep.sum()), 0))
    rows = [dict(id=int(i), Da=float(d), Volume=float(v), **{f: float(x) for f, x in zip(FEAT, xr)},
                 **{e: float(x) for e, x in zip(extra, er)}, shape_class=int(c))
            for i, d, v, xr, er, c in zip(ids, t["Da"][keep], vol, X, E, cls)]
    stats = shape_stats(rows)
    stats["n_all"] = n_all
    stats["fines_lt20um_num_pct"] = 100 * float((t["Da"] < 20).mean()) if n_all else float("nan")
    g = lambda p: [stats.get(f"{p}_C{k}", float("nan")) for k in range(1, 6)]
    chart = share_chart(f"{sample_name(path)}: shape classes from the Microtrac particle table",
                        f"{stats['n_counted']} particles of {sc.min_da:g} um or more, {stats.get('n_band', 0)} of them "
                        f"between 30 and 100 um",
                        [("% by number", g("num_pct")), ("% by volume", g("vol_pct")),
                         ("% by volume, 30-100 um", g("band_vol_pct"))])
    note = (f"Microtrac particle table, {n_all} particles, {stats['n_counted']} of {sc.min_da:g} µm or more "
            "classified from Microtrac's own descriptors. Sizes are area-equivalent diameters (Da).")
    warn = []
    if stats.get("n_band", 0) < 200:
        warn.append(f"Only {stats.get('n_band', 0)} particles fall in the 30–100 µm band; its shares rest on few particles.")
    step(1.0, "done")
    return _result(path, "table", chart, chart, rows, stats, note, settings, warn, run=run_date(path))


# --------------------------------------------------------------- size report

def parse_size_text(text):
    out = {}
    for k in ("MV", "MN", "MA"):
        m = re.search(k + r"\(um\):\s*([\d.]+)", text)
        if m:
            out[f"LD_{k}"] = float(m.group(1))
    m = re.search(r"SD:\s*([\d.]+)", text)
    if m:
        out["LD_SD"] = float(m.group(1))
    m = re.search(r"%Tile\s+Size\(um\)\s*((?:[\d.]+\s+[\d.]+\s*)+)", text)
    if m:
        v = m.group(1).split()
        for p, s in zip(v[0::2], v[1::2]):
            try:
                out[f"LD_D{int(float(p))}"] = float(s)
            except ValueError:
                pass
    for k, pat in (("date", r"\bDate\s+(\d{4}-\d{2}-\d{2})"), ("time", r"\bTime\s+(\d{1,2}:\d{2}\s*[AP]M)")):
        m = re.search(pat, text)
        if m:
            out[k] = m.group(1)
    return out


def size_result(path, settings, step):
    import pypdfium2 as pdfium
    step(0.2, "reading the report")
    pdf = pdfium.PdfDocument(path)
    try:
        page = pdf[0]
        st = parse_size_text(_pdf_text(page))
        img = np.array(page.render(scale=2.0).to_pil().convert("RGB"))
    finally:
        pdf.close()
    if "LD_D50" not in st:
        raise ValueError("No percentile table found in this size report.")
    when = " ".join(str(st.pop(k)) for k in ("date", "time") if k in st)
    if "LD_D10" in st and "LD_D90" in st:
        st["LD_span"] = (st["LD_D90"] - st["LD_D10"]) / st["LD_D50"]
    st["n_counted"] = 0
    note = "Microtrac laser diffraction size report" + (f", measured {when}" if when else "") + \
           ". Percentiles are by volume. A size report has no particle shapes, so it gets no shape classes."
    step(1.0, "done")
    return _result(path, "size", img, img, [], st, note, settings, run=run_date(path, when))


def size_csv_result(path, settings, step):
    from .shape_classes import read_ld_summary
    step(0.2, "reading the size data")
    try:
        ld = read_ld_summary(path)
    except (AttributeError, ValueError) as e:
        raise ValueError("No percentile table found in this size export.") from e
    st = {f"LD_{k}": v for k, v in ld.items()}
    if "LD_D50" not in st:
        raise ValueError("No percentile table found in this size export.")
    if "LD_D10" in st and "LD_D90" in st:
        st["LD_span"] = (st["LD_D90"] - st["LD_D10"]) / st["LD_D50"]
    st["n_counted"] = 0
    text = open(path, encoding="latin1").read()
    m = re.search(r"Date:\s*,\s*([\d-]+)", text)
    when = m.group(1) if m else ""
    rows = [("D10", st.get("LD_D10")), ("D50", st.get("LD_D50")), ("D90", st.get("LD_D90")),
            ("Span", st.get("LD_span")), ("MV", st.get("LD_MV")), ("MA", st.get("LD_MA")), ("MN", st.get("LD_MN"))]
    img = size_card(sample_name(path), when, rows)
    note = ("Microtrac laser diffraction size export (.csv)" + (f", measured {when}" if when else "") +
            ". Percentiles are by volume. A size export has no particle shapes, so it gets no shape classes.")
    step(1.0, "done")
    return _result(path, "size", img, img, [], st, note, settings, run=run_date(path, when))


def size_card(batch, when, rows):
    """Picture of the size numbers, shown in the viewer for a .csv size export."""
    im = Image.new("RGB", (1400, 760), "white")
    d = ImageDraw.Draw(im)
    d.text((60, 40), f"{batch}: laser diffraction size", fill=(31, 78, 121), font=_font(46))
    d.text((60, 108), "by volume" + (f", measured {when}" if when else ""), fill=(87, 96, 106), font=_font(28))
    for k, (name, v) in enumerate(rows):
        y = 190 + k * 76
        d.text((80, y), name, fill=(87, 96, 106), font=_font(34))
        txt = "n/a" if v is None else (f"{v:.2f}" if name == "Span" else f"{v:.1f} um")
        d.text((420, y), txt, fill=(31, 35, 40), font=_font(38))
    return np.array(im)


# ---------------------------------------------------------------- dispatcher

def analyse(path, settings, step):
    """Result for one Microtrac file, or None when the file is an SEM image for the usual route."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv" and is_size_csv(path):
        return size_csv_result(path, settings, step)
    if ext in (".xls", ".xlsx", ".csv"):
        return table_result(path, settings, step)
    if ext == ".pdf":
        kind = pdf_kind(path)
        if kind == "gallery":
            r = gallery_pdf_result(path, settings, step)
        elif kind == "size":
            r = size_result(path, settings, step)
        else:
            raise ValueError("This PDF is neither a Microtrac particle report nor a Microtrac size report.")
        step(1.0, "done")
        return r
    rgb = np.array(Image.open(path).convert("RGB"))
    if looks_like_gallery(rgb):
        r = gallery_image_result(path, rgb, settings, step)
        step(1.0, "done")
        return r
    return None
