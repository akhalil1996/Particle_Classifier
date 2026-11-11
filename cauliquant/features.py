"""Per-particle shape and texture features.

Two groups of columns come out of particle_table():

* the original ones (solidity, roughness, elong, grad_mean, relief_mean), kept
  so that older tables stay comparable;
* the ones the particle classifier uses (see classifier.py). Their texture
  measures are taken on a grey scale tied to the image itself (0 = background
  level, 1 = typical particle brightness), so that a change of brightness or
  contrast between SEM sessions does not move them, and away from the outline,
  where a merged or cut outline would contaminate them.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, filters

RELIEF_SCALES = (1.5, 3, 6, 12)          # px; 0.7 to 6 um at 0.496 um/px
TEXTURE_KEYS = [f"rel{s}" for s in RELIEF_SCALES] + ["grad", "peaks", "dark", "bright", "lap", "gn", "gn2"]


def texture_maps(gray):
    """Per-pixel texture measures on the image-normalised grey scale."""
    g = gray.astype(np.float32)
    s = filters.gaussian(g, 1, preserve_range=True)
    fg = s > filters.threshold_otsu(s)
    lo = float(np.median(g[~fg])) if (~fg).any() else 0.0
    hi = float(np.median(g[fg])) if fg.any() else 255.0
    gn = (g - lo) / max(hi - lo, 1.0)
    m = {f"rel{sc}": np.abs(gn - ndi.gaussian_filter(gn, sc)) for sc in RELIEF_SCALES}
    m["grad"] = filters.sobel(gn)
    dog = ndi.gaussian_filter(gn, 1.5) - ndi.gaussian_filter(gn, 4)
    m["peaks"] = ((dog == ndi.maximum_filter(dog, 7)) & (dog > 0.04)).astype(np.float32)   # bright nodules a few um across
    m["dark"] = (gn < 0.45).astype(np.float32)                                              # shadowed gaps inside a particle
    m["bright"] = (gn > 1.35).astype(np.float32)
    m["lap"] = np.abs(ndi.gaussian_laplace(gn, 2))
    m["gn"] = gn
    m["gn2"] = gn * gn
    return m


def particle_table(gray, lab, um_px):
    """One dict per particle: size, shape and surface texture."""
    g = gray.astype(float)
    grad = filters.sobel(g / 255.0)
    # fine-scale relief: difference between the image and a smoothed copy
    relief = np.abs(g - filters.gaussian(g, 3, preserve_range=True))
    n = int(lab.max())
    maps = texture_maps(gray)
    inner = lab * ndi.binary_erosion(lab > 0, iterations=3)
    full = np.bincount(lab.ravel(), minlength=n + 1).astype(float)
    cnt = np.bincount(inner.ravel(), minlength=n + 1).astype(float)
    use = np.where(cnt >= 10, 1, 0)                       # too thin for an inner region: measure the whole particle
    mean = {}
    for k, v in maps.items():
        a = np.bincount(inner.ravel(), weights=v.ravel().astype(float), minlength=n + 1) / np.maximum(cnt, 1)
        b = np.bincount(lab.ravel(), weights=v.ravel().astype(float), minlength=n + 1) / np.maximum(full, 1)
        mean[k] = np.where(use == 1, a, b)
    rows = []
    H, W = lab.shape
    for rp in measure.regionprops(lab):
        y0, x0, y1, x1 = rp.bbox
        sl = lab[y0:y1, x0:x1] == rp.label
        inn = ndi.binary_erosion(sl, iterations=2)
        if inn.sum() < 10:
            inn = sl
        perim = max(rp.perimeter, 1.0)
        i = rp.label
        r = dict(
            id=int(i),
            area_um2=rp.area * um_px ** 2,
            ecd_um=rp.equivalent_diameter_area * um_px,
            solidity=rp.solidity,
            roughness=perim ** 2 / (4 * np.pi * rp.area),
            elong=rp.major_axis_length / max(rp.minor_axis_length, 1.0),
            grad_mean=float(grad[y0:y1, x0:x1][inn].mean()),
            relief_mean=float(relief[y0:y1, x0:x1][inn].mean()),
            edge=int(y0 == 0 or x0 == 0 or y1 >= H or x1 >= W),
            cx=float(rp.centroid[1]), cy=float(rp.centroid[0]),
            holes=1 - rp.area / max(rp.area_filled, 1),
            extent=rp.extent,
        )
        for k in TEXTURE_KEYS:
            r[k] = float(mean[k][i])
        r["istd"] = float(np.sqrt(max(r["gn2"] - r["gn"] ** 2, 0.0)))
        r["peak_density"] = r.pop("peaks") * 100.0 / um_px ** 2          # nodule-sized bright spots per 100 um2
        del r["gn2"]
        rows.append(r)
    return rows
