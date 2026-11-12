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


def synthetic_gallery(png_path, pdf_path, seed=1):
    """A page of particle silhouettes like a Microtrac viewer report, as a picture and as a PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    rng = np.random.default_rng(seed)
    page = np.full((1056, 816, 3), 255, np.uint8)
    yy, xx = np.mgrid[:50, :50]
    c = canvas.Canvas(pdf_path, pagesize=A4)
    for k in range(40):
        cell = np.full((50, 50, 3), 255, np.uint8)
        a, b = rng.uniform(12, 24), rng.uniform(8, 24)
        m = ((yy - 25) / b) ** 2 + ((xx - 25) / a) ** 2 < 1
        if k % 3 == 0:
            for _ in range(6):
                cy, cx = rng.uniform(8, 42, 2)
                m |= (yy - cy) ** 2 + (xx - cx) ** 2 < rng.uniform(20, 60)
        ys, xs = np.nonzero(m)                      # the report scales every particle to fill its cell
        crop = Image.fromarray((m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255).astype(np.uint8))
        f = 48 / max(crop.size)
        crop = np.array(crop.resize((max(1, round(crop.width * f)), max(1, round(crop.height * f))))) > 127
        oy, ox = (50 - crop.shape[0]) // 2, (50 - crop.shape[1]) // 2
        cell[oy:oy + crop.shape[0], ox:ox + crop.shape[1]][crop] = 30
        y0, x0 = 200 + (k // 10) * 66, 40 + (k % 10) * 66
        page[y0:y0 + 50, x0:x0 + 50] = cell
        c.drawImage(ImageReader(Image.fromarray(cell)), 30 + (k % 10) * 46, 700 - (k // 10) * 46, 37, 37)
    c.save()
    Image.fromarray(page).save(png_path)
    return png_path, pdf_path


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
    from .analysis import morphology_input
    png, pdf = synthetic_gallery(os.path.join(out_dir, "gallery_page1.png"), os.path.join(out_dir, "gallery.pdf"))
    rp, rd = morphology_input(png, s), morphology_input(pdf, s)
    log(f"Microtrac report: picture {rp['stats']['n_counted']} silhouettes, PDF {rd['stats']['n_counted']} silhouettes")
    assert rp["source"] == "gallery" and rd["source"] == "gallery"
    assert rp["stats"]["n_counted"] == 40 and rd["stats"]["n_counted"] == 40
    csvp = os.path.join(out_dir, "S1.csv")
    with open(csvp, "w") as fh:
        fh.write(",, Title: ,-RTIT-Ti, Date: ,2026-05-06, Summary Data ,,,MV(um):,178.5,MN(um):,77.04,MA(um):,141.0,"
                 "SD:,83.33,Percentiles,,,%Tile,Size(um),10,82.82,50,168.3,90,298.8,95,321.3 Size Percent ,,\n")
    rs = morphology_input(csvp, s)
    assert rs["source"] == "size" and abs(rs["stats"]["LD_D50"] - 168.3) < 1e-6
    from openpyxl import Workbook
    wb = Workbook(); ws = wb.active
    feats = ["Id", "Da", "Volume", "Circularity", "Solidity", "Convexity", "W/L Aspect Ratio", "Extent", "Transparency"]
    ws.append(feats)
    rng = np.random.default_rng(0)
    for i in range(300):
        da = rng.uniform(12, 150)
        ws.append([i + 1, da, np.pi / 6 * da ** 3] + list(rng.uniform([0.4, 0.8, 0.9, 0.3, 0.5, 0.15],
                                                                       [0.95, 1.0, 1.0, 0.95, 0.85, 0.25])))
    xp = os.path.join(out_dir, "S1_table.xlsx")
    wb.save(xp)
    rt = morphology_input(xp, s)
    assert rt["source"] == "table" and rt["stats"]["n_counted"] == 300
    from .figures import figures
    from .analysis import summarise_morphology
    figs = figures([rm, rp, rd, rs, rt], summarise_morphology([rm, rp, rd, rs, rt]))
    log("figures drawn: " + ", ".join(str(n) for n, _, png, _ in figs if png))
    assert sum(1 for f in figs if f[2]) >= 4
    p = build_report([rm, rp, rd, rs, rt], os.path.join(out_dir, "selftest_microtrac.pdf"))
    assert os.path.getsize(p) > 10000
    log(f"reports written to {out_dir}")
    return out_dir
