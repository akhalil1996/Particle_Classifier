"""Border (guard-frame) rules: which particles count toward a field's area %.

A particle cut by the image border cannot be classified or measured whole, and
large particles are cut more often than small ones. Each rule below decides how
much of each particle's area enters the totals, inside a counting frame that is
inset from the usable image by `inset_um` on every side.

    edge      count a particle only if it lies wholly inside the frame
              (inset 0 = drop every particle touching the image border;
              this was the pipeline's rule up to Nov 2026)
    half      count the whole particle if more than 50% of its area lies
              inside the frame and it does not touch the image border
    centroid  count the whole particle if its centroid lies inside the frame
              and it does not touch the image border (associated-point rule)
    ml        `edge`, with each kept particle weighted by the Miles-Lantuejoul
              factor Fw*Fh / ((Fw - w)(Fh - h)), w and h its bounding box
    clip      count only the pixels of every particle that fall inside the
              frame, whole or cut (area-fraction estimate)

`half` and `centroid` only stay unbiased while the inset is at least about half
the largest particle; with a smaller inset some particles they want are cut by
the image border and are lost (reported as `lost`).
"""
import numpy as np
from skimage import measure

RULES = ("edge", "half", "centroid", "ml", "clip")


def frame_box(shape, um_px, inset_um=0.0, usable_h=None):
    """(y0, x0, y1, x1) of the counting frame in pixels, end-exclusive."""
    H, W = shape
    H = min(H, usable_h) if usable_h else H
    d = int(round(inset_um / um_px))
    if 2 * d >= min(H, W):
        raise ValueError(f"inset {inset_um} um leaves no frame")
    return d, d, H - d, W - d


def counted_area(lab, um_px, rule="edge", inset_um=0.0, usable_h=None):
    """Per particle id: area that counts (um2), and whether a wanted particle was lost.

    Returns {id: (counted_um2, lost)}; counted 0 means the particle does not count.
    `usable_h` is the first image row that is not real image (a text overlay).
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    H, W = lab.shape
    UH = min(H, usable_h) if usable_h else H
    fy0, fx0, fy1, fx1 = frame_box(lab.shape, um_px, inset_um, usable_h)
    a2 = um_px ** 2
    out = {}
    for rp in measure.regionprops(lab):
        y0, x0, y1, x1 = rp.bbox
        border = y0 == 0 or x0 == 0 or x1 >= W or y1 >= UH
        whole_in = not border and y0 >= fy0 and x0 >= fx0 and y1 <= fy1 and x1 <= fx1
        if rp.area == 0:
            continue
        if rule == "edge":
            out[rp.label] = (rp.area * a2 if whole_in else 0.0, False)
        elif rule == "ml":
            if whole_in:
                fh, fw = fy1 - fy0, fx1 - fx0
                h, w = y1 - y0, x1 - x0
                out[rp.label] = (rp.area * a2 * fh * fw / max((fh - h) * (fw - w), 1), False)
            else:
                out[rp.label] = (0.0, False)
        elif rule == "clip":
            sl = lab[max(y0, fy0):min(y1, fy1), max(x0, fx0):min(x1, fx1)] == rp.label
            out[rp.label] = (float(sl.sum()) * a2, False)
        else:
            if rule == "half":
                sl = lab[max(y0, fy0):min(y1, fy1), max(x0, fx0):min(x1, fx1)] == rp.label
                want = sl.sum() > 0.5 * rp.area
            else:
                cy, cx = rp.centroid
                want = fy0 <= cy < fy1 and fx0 <= cx < fx1
            out[rp.label] = (rp.area * a2 if want and not border else 0.0, bool(want and border))
    return out
