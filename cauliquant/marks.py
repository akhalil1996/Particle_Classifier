"""Folder B: the marked version of each raw image in folder A.

For an image A/<name>.tif, folder B holds
  <name>_mask.png     class per pixel: 0 background, 1 dense, 2 cauliflower,
                      255 ignore (uncertain, bad outline, below 10 um, unlabeled)
  <name>_objects.png  particle id per pixel (16-bit), so labels map exactly
  <name>_objects.csv  one row per particle: features, class, source, score
  <name>_overlay.jpg  the image with classes coloured, for checking by eye
The source column says who decided the class: "human" (from the labeling
bench) or "draft" (workflow 1). Human rows are never overwritten.
"""
import csv
import os
import numpy as np
from PIL import Image

BG, DENSE, CAULI, IGNORE = 0, 1, 2, 255
CODE = {"dense": DENSE, "cauliflower": CAULI, "uncertain": IGNORE, "bad": IGNORE,
        "small": IGNORE, "unlabeled": IGNORE, "overlay": IGNORE}
COLOURS = {DENSE: (47, 127, 168), CAULI: (217, 119, 31)}
FIELDS = ["id", "class", "source", "score", "area_um2", "ecd_um", "solidity",
          "roughness", "elong", "grad_mean", "relief_mean", "edge", "cx", "cy"]


def paths(b_dir, name):
    base = os.path.join(b_dir, name)
    return dict(mask=base + "_mask.png", objects=base + "_objects.png",
                csv=base + "_objects.csv", overlay=base + "_overlay.jpg")


def read_rows(b_dir, name):
    p = paths(b_dir, name)["csv"]
    if not os.path.exists(p):
        return None
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def write(b_dir, name, gray, lab, rows, ignore_below=None):
    os.makedirs(b_dir, exist_ok=True)
    p = paths(b_dir, name)
    lut = np.full(int(lab.max()) + 1, IGNORE, np.uint8)
    lut[0] = BG
    for r in rows:
        lut[int(r["id"])] = CODE[r["class"]]
    mask = lut[lab]
    if ignore_below is not None:
        mask[ignore_below:] = IGNORE
    Image.fromarray(mask).save(p["mask"])
    Image.fromarray(lab.astype(np.uint16)).save(p["objects"])
    with open(p["csv"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    rgb = np.stack([gray] * 3, -1).astype(float)
    for k, c in COLOURS.items():
        sel = mask == k
        rgb[sel] = 0.55 * rgb[sel] + 0.45 * np.array(c)
    Image.fromarray(rgb.astype(np.uint8)).save(p["overlay"], quality=85)
    return mask


def load_pair(a_path, b_dir):
    """(grey image, class mask, rows) for one A/B pair, or None if B is missing."""
    from .segment import load_sem
    name = os.path.splitext(os.path.basename(a_path))[0]
    p = paths(b_dir, name)
    if not os.path.exists(p["mask"]):
        return None
    gray, _ = load_sem(a_path)
    mask = np.array(Image.open(p["mask"]))
    return gray, mask, read_rows(b_dir, name)
