"""Five shape classes (C1-C5) from the k-means model fitted on Microtrac SYNC data.

The model is frozen in models/shape_classes_k5.json: a scaler and five centroids
over Circularity, Solidity, Convexity, W/L Aspect Ratio and Extent. The same
classes can be given to a Microtrac particle table or to SEM particles, which is
what lets the two instruments be compared particle class by particle class.
"""
import glob
import io
import json
import os
import re
import numpy as np
from scipy.spatial import ConvexHull
from skimage import measure

from .resources import SHAPE_MODEL as MODEL
FEAT = ["Circularity", "Solidity", "Convexity", "W/L Aspect Ratio", "Extent"]
SHORT = ["C1", "C2", "C3", "C4", "C5"]
SHAPE_RGB = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164)]   # as in the study figures
SHAPE_NAMES = ["C1 compact", "C2 elongated smooth", "C3 blocky rough", "C4 elongated concave",
               "C5 branched / agglomerate"]
BAND = (30.0, 100.0)          # um, area-equivalent diameter: the band shares are reported for


class ShapeClasses:
    def __init__(self, path=MODEL):
        m = json.load(open(path))
        assert m["features"] == FEAT, m["features"]
        self.mean = np.array(m["scaler_mean"]); self.std = np.array(m["scaler_std"])
        self.cent = np.array([c["centroid_standardized"] for c in m["classes"]])
        self.names = [c["name"] for c in m["classes"]]
        self.min_da = m["min_Da_um"]

    def predict(self, X):
        """Class index 0-4 for each row of X (n, 5) in FEAT order; -1 where a value is missing."""
        X = np.asarray(X, float)
        z = (X - self.mean) / self.std
        d = ((z[:, None, :] - self.cent[None]) ** 2).sum(-1)
        out = d.argmin(1)
        out[~np.isfinite(X).all(1)] = -1
        return out


def shape_stats(rows):
    """Class shares by number, by volume and in the band, for particles with Da, Volume and shape_class."""
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
    for k in FEAT:
        out[f"median_{k}"] = float(np.median([r[k] for r in rows]))
    return out


def number_stats(rows):
    """Class shares by number only, for particles without a size (report silhouettes)."""
    rows = [r for r in rows if r.get("shape_class", 0) > 0]
    out = {"n_counted": len(rows)}
    if not rows:
        return out
    cls = np.array([r["shape_class"] for r in rows])
    for k in range(1, 6):
        out[f"num_pct_C{k}"] = 100 * float((cls == k).mean())
    for k in FEAT:
        out[f"median_{k}"] = float(np.median([r[k] for r in rows]))
    return out


# ---- Microtrac SYNC exports --------------------------------------------------

def read_sync_table(path):
    """One '(All particles).xls' export (tab-separated text, units row skipped) as a DataFrame."""
    import pandas as pd
    txt = open(path, encoding="latin1").read().splitlines()
    df = pd.read_csv(io.StringIO("\n".join([txt[0]] + txt[2:])), sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df


def read_ld_summary(path):
    """Laser diffraction summary .csv: D10...D95, MV, SD."""
    t = open(path, encoding="latin1").read().replace("\r", "").replace("\n", "")
    m = re.search(r"Percentiles,+%Tile,Size\(um\),(.*?)Size", t).group(1).strip(", ").split(",")
    r = {k: float(re.search(k + r"\(um\):,([\d.]+)", t).group(1)) for k in ("MV", "MN", "MA")}
    r["SD"] = float(re.search(r"SD:,([\d.]+)", t).group(1))
    for p, v in zip(m[0::2], m[1::2]):
        try:
            r[f"D{int(p)}"] = float(v)
        except ValueError:
            pass
    return r


def sync_folder(root):
    """Every batch sub-folder of root: particle table plus laser diffraction summary if present."""
    import pandas as pd
    parts, ld = [], {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        xls = glob.glob(os.path.join(p, "*.xls"))
        if not os.path.isdir(p) or not xls:
            continue
        df = read_sync_table(xls[0]); df["sample"] = d; parts.append(df)
        csvs = glob.glob(os.path.join(p, "*.csv"))
        if csvs:
            ld[d] = read_ld_summary(csvs[0])
    return pd.concat(parts, ignore_index=True), ld


# ---- SEM particles -----------------------------------------------------------

def _min_area_rect(pts):
    """Side lengths (short, long) of the minimum-area rectangle around points (rotating calipers on the hull)."""
    h = pts[ConvexHull(pts).vertices]
    best = (np.inf, 0, 0)
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        n = np.hypot(*e)
        if n == 0:
            continue
        u = e / n; v = np.array([-u[1], u[0]])
        a, b = h @ u, h @ v
        w, l = a.max() - a.min(), b.max() - b.min()
        if w * l < best[0]:
            best = (w * l, min(w, l), max(w, l))
    return best[1], best[2]


def sem_descriptors(lab, um_px, ids=None):
    """Microtrac-style descriptors for SEM particles: {id: dict(Da, Volume, Circularity, ...)}.

    Perimeters come from the sub-pixel contour of each mask, so they are not
    inflated by the pixel staircase.
    """
    out = {}
    for rp in measure.regionprops(lab):
        if ids is not None and rp.label not in ids:
            continue
        m = np.pad(rp.image, 1).astype(float)
        cs = measure.find_contours(m, 0.5)
        if not cs:
            continue
        c = max(cs, key=len)
        if len(c) < 8:
            continue
        P = np.hypot(*np.diff(c, axis=0).T).sum()
        x, y = c[:, 1], c[:, 0]
        A = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        hull = ConvexHull(c)
        hp = hull.points[hull.vertices]
        HP = np.hypot(*np.diff(np.vstack([hp, hp[:1]]), axis=0).T).sum()
        w, l = _min_area_rect(c)
        da = 2 * np.sqrt(A / np.pi) * um_px
        out[rp.label] = {"Da": da, "Volume": np.pi / 6 * da ** 3,
                         "Circularity": 4 * np.pi * A / P ** 2, "Solidity": A / hull.volume,
                         "Convexity": HP / P, "W/L Aspect Ratio": w / l, "Extent": A / (w * l)}
    return out
