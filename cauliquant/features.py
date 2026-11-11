"""Per-particle shape and texture features."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, filters


def particle_table(gray, lab, um_px):
    """One dict per particle: size, shape and surface texture."""
    g = gray.astype(float)
    grad = filters.sobel(g / 255.0)
    # fine-scale relief: difference between the image and a smoothed copy
    relief = np.abs(g - filters.gaussian(g, 3, preserve_range=True))
    rows = []
    H, W = lab.shape
    for rp in measure.regionprops(lab):
        y0, x0, y1, x1 = rp.bbox
        sl = lab[y0:y1, x0:x1] == rp.label
        inner = ndi.binary_erosion(sl, iterations=2)
        if inner.sum() < 10:
            inner = sl
        perim = max(rp.perimeter, 1.0)
        rows.append(dict(
            id=int(rp.label),
            area_um2=rp.area * um_px ** 2,
            ecd_um=rp.equivalent_diameter_area * um_px,
            solidity=rp.solidity,
            roughness=perim ** 2 / (4 * np.pi * rp.area),
            elong=rp.major_axis_length / max(rp.minor_axis_length, 1.0),
            grad_mean=float(grad[y0:y1, x0:x1][inner].mean()),
            relief_mean=float(relief[y0:y1, x0:x1][inner].mean()),
            edge=int(y0 == 0 or x0 == 0 or y1 >= H or x1 >= W),
            cx=float(rp.centroid[1]), cy=float(rp.centroid[0]),
        ))
    return rows
