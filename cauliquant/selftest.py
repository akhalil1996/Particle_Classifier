"""Quick end-to-end check on a synthetic image: segmentation, U-Net, shape classes, PDF report.

Used by the build (CauliQuant.exe --self-test) to prove the packaged app works.
It checks the machinery only, not accuracy.
"""
import os
import tempfile
import time

import numpy as np
from PIL import Image


def synthetic_image(path, seed=0):
    rng = np.random.default_rng(seed)
    H, W = 960, 1280
    yy, xx = np.mgrid[:H, :W]
    img = np.full((H, W), 70.0)
    for _ in range(45):
        cy, cx = rng.uniform(80, H - 80), rng.uniform(80, W - 80)
        r = rng.uniform(15, 55)
        if rng.random() < 0.4:            # knobbly cluster
            for _ in range(12):
                a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, r)
                rr = rng.uniform(6, 14)
                img[(yy - cy - d * np.sin(a)) ** 2 + (xx - cx - d * np.cos(a)) ** 2 < rr ** 2] = 190
        else:
            e = rng.uniform(0.6, 1.0)
            img[((yy - cy) / (r * e)) ** 2 + ((xx - cx) / r) ** 2 < 1] = 175
    img += rng.normal(0, 8, img.shape)
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(path)
    return path


def run(out_dir=None, log=print):
    from .analysis import Settings, classify_image, morphology_image
    from .report import build_report
    out_dir = out_dir or tempfile.mkdtemp(prefix="cauliquant_selftest_")
    img = synthetic_image(os.path.join(out_dir, "synthetic_x100_1.png"))
    s = Settings()
    t = time.time()
    rc = classify_image(img, s)
    log(f"classification: {rc['stats']['n_counted']} particles, "
        f"{rc['stats']['cauliflower_area_pct']:.1f} % cauliflower ({time.time() - t:.1f} s)")
    t = time.time()
    rm = morphology_image(img, s)
    log(f"morphology: {rm['stats']['n_counted']} particles, C5 {rm['stats'].get('vol_pct_C5', float('nan')):.1f} % "
        f"by volume ({time.time() - t:.1f} s)")
    assert rc["stats"]["n_counted"] > 10 and rm["stats"]["n_counted"] > 10
    for name, r in (("classification", rc), ("morphology", rm)):
        p = build_report([r, r], os.path.join(out_dir, f"selftest_{name}.pdf"))
        assert os.path.getsize(p) > 10000
    log(f"reports written to {out_dir}")
    return out_dir
