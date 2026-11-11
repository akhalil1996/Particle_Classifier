"""Checks that tell the user when a result should not be trusted as it stands.

Three questions are asked of every image:

1. Was it taken the way the model was validated?  (SEM settings from the metadata)
2. Is the field itself fit to measure?            (crowding, focus, particle count)
3. How firm were the calls?                        (share of area near the cut-off)

and one of every sample: how far could the pooled result move if other fields
had been imaged (interval over fields).

The limits come from the model card that travels with the model file, so a
retrained model brings its own validated settings and limits.
"""
import re

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation


def _number(text):
    m = re.search(r"-?\d+(\.\d+)?", str(text))
    return float(m.group()) if m else None


def settings_check(meta, image_info, card):
    """Warnings for SEM settings that differ from those the model was validated on."""
    v = card.get("validated", {})
    out = []
    if not meta:
        return ["No SEM metadata found, so the imaging settings could not be checked."]
    kv = _number(meta.get("AcceleratingVoltage", ""))
    if kv is not None and v.get("kv") and abs(kv / 1000.0 - v["kv"]) > 0.5:
        out.append(f"Accelerating voltage {kv / 1000:.0f} kV; the model was validated at {v['kv']:.0f} kV.")
    mag = _number(meta.get("Magnification", ""))
    if mag is not None and v.get("magnification") and abs(mag / v["magnification"] - 1) > 0.05:
        out.append(f"Magnification x{mag:.0f}; the model was validated at x{v['magnification']:.0f}.")
    spot = _number(meta.get("Condencer1", ""))
    if spot is not None and v.get("spot") and abs(spot / 1000.0 - v["spot"]) > 1:
        out.append(f"Spot intensity {spot / 1000:.0f}; the model was validated at {v['spot']:.0f}.")
    scan = meta.get("ScanSpeed", "")
    if scan and v.get("scan") and v["scan"].lower() not in scan.lower():
        out.append(f"Scan speed {scan}; the model was validated at {v['scan']}.")
    sig = meta.get("SignalName", "")
    if sig and v.get("signal") and sig.upper() != v["signal"].upper():
        out.append(f"Detector signal {sig}; the model was validated on {v['signal']} images.")
    if abs(image_info.get("scale", 1.0) - 1.0) > 0.05:
        out.append(f"Image resampled from {image_info['native_um_px']:.3f} to {image_info['um_px']:.3f} µm/px; "
                   "results on resampled images read low on fine-nodule cauliflower.")
    return out


def field_metrics(gray, lab, um_px):
    """Crowding and sharpness of one field."""
    fg = lab > 0
    blobs = measure.label(fg)
    n = int(lab.max())
    area = np.bincount(lab.ravel(), minlength=n + 1).astype(float)
    # blob of each particle, and how many particles share it
    first = ndi.maximum(blobs, lab, index=np.arange(1, n + 1)).astype(int) if n else np.array([], int)
    per_blob = np.bincount(first, minlength=int(blobs.max()) + 1)
    big = np.sqrt(4 * area[1:] / np.pi) * um_px >= 30
    shared = per_blob[first] > 1
    touching = 100 * area[1:][big & shared].sum() / max(area[1:][big].sum(), 1)
    s = filters.gaussian(gray.astype(float), 1, preserve_range=True)
    edge = float(filters.sobel(s / 255.0)[segmentation.find_boundaries(fg)].mean()) if fg.any() else 0.0
    return dict(coverage_pct=100 * float(fg.mean()), touching_area_pct=float(touching), edge_sharpness=edge,
                saturated_pct=100 * float((gray >= 254).mean()))


def field_check(m, n_counted, card):
    lim = card.get("limits", {})
    out = []
    if m["coverage_pct"] > lim.get("coverage_max_pct", 55):
        out.append(f"Crowded field: particles cover {m['coverage_pct']:.0f} % of the image "
                   f"(validated up to {lim.get('coverage_max_pct', 55):.0f} %). Touching particles are harder to separate.")
    if m["edge_sharpness"] < lim.get("edge_sharpness_min", 0.08):
        out.append("Particle edges look soft; the image may be out of focus.")
    if m["saturated_pct"] > lim.get("saturated_max_pct", 5):
        out.append(f"{m['saturated_pct']:.0f} % of pixels are saturated; surface texture is lost there.")
    if n_counted < lim.get("min_particles", 50):
        out.append(f"Only {n_counted} particles counted; one field like this gives a rough number.")
    return out


def field_interval(results, n_boot=2000, seed=0):
    """95 % interval of the pooled cauliflower area % of one sample, resampling its fields.

    Fields are the sampling unit: particles within a field are not independent
    (segregation on the stub), so the interval is built over fields, not particles.
    Returns (low, high) or None when there are fewer than three fields.
    """
    a_c = np.array([r["stats"]["cauliflower_area_um2"] for r in results], float)
    a_d = np.array([r["stats"]["dense_area_um2"] for r in results], float)
    k = len(a_c)
    if k < 3 or (a_c + a_d).sum() <= 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, (n_boot, k))
    c, d = a_c[idx].sum(1), a_d[idx].sum(1)
    pct = 100 * c / np.maximum(c + d, 1e-9)
    return float(np.percentile(pct, 2.5)), float(np.percentile(pct, 97.5))


def fields_needed(field_sd_pct, half_width_pct=5.0):
    """Fields needed for a 95 % interval of +/- half_width, from the field-to-field SD."""
    if not np.isfinite(field_sd_pct) or field_sd_pct <= 0:
        return None
    return int(np.ceil((1.96 * field_sd_pct / half_width_pct) ** 2))
