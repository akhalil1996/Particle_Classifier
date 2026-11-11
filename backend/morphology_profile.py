r"""Workflow 3: batch morphology from Microtrac SYNC exports and from classified SEM images,
on the same five shape classes (C1-C5), so the two can be correlated batch by batch.

    python morphology_profile.py --sync <folder with one sub-folder per batch> --out sync_batches.csv
    python morphology_profile.py --sem ..\B_marked --a ..\A_raw --out sem_samples.csv
    python morphology_profile.py --join sync_batches.csv sem_samples.csv --out paired.csv

--sync reads each batch's '(All particles).xls' (and its laser diffraction .csv
if present) and gives the share of each class by number and by volume, the
same shares in the 30-100 um band, band medians of the shape descriptors,
fines and D50. Classes come from the frozen model in models/shape_classes_k5.json;
nothing is refitted.

--sem takes folder B after classify_folder.py and gives, per sample (the part of
the image name before _x<mag>), the cauliflower share by area and by volume and
the C1-C5 shares computed from the SEM outlines. SEM outlines are sharper than
Microtrac's, so the SEM C1-C5 shares are not yet on the Microtrac scale; see
the correlation plan before comparing them directly.

--join pairs the two tables on the batch name and adds Spearman correlations.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))  # repo root, for cauliquant
import argparse
import csv
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image

from cauliquant.shape_classes import ShapeClasses, FEAT, SHORT, sync_folder, sem_descriptors
from cauliquant.segment import load_sem, overlay_top
from cauliquant.frame import counted_area

BAND = (30, 100)
BAND_DESC = ["Circularity", "Solidity", "Convexity", "W/L Aspect Ratio", "Extent"]


def class_shares(df, prefix, out):
    v = df["Volume"]
    for k, s in enumerate(SHORT, 1):
        out[f"{prefix}num%_{s}"] = 100 * (df.cls == k).mean()
        out[f"{prefix}vol%_{s}"] = 100 * v[df.cls == k].sum() / v.sum()


def sync_profile(root):
    sc = ShapeClasses()
    parts, ld = sync_folder(root)
    rows = []
    for s, g in parts.groupby("sample"):
        r = {"sample": s, "n_all": len(g), "fines_lt20um_num%": 100 * (g.Da < 20).mean()}
        g = g[g.Da >= sc.min_da].copy()
        g["cls"] = sc.predict(g[FEAT].values) + 1
        r["n"] = len(g); r["D50_num_um"] = g.Da.median()
        class_shares(g, "", r)
        band = g[(g.Da >= BAND[0]) & (g.Da < BAND[1])]
        class_shares(band, "band_", r)
        for k in BAND_DESC:
            r[f"band_med_{k}"] = band[k].median()
        for k, v in ld.get(s, {}).items():
            r[f"LD_{k}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def sem_profile(b_dir, a_dir, rule="half", inset_um=50.0):
    sc = ShapeClasses()
    parts = []
    for f in sorted(glob.glob(os.path.join(b_dir, "*_objects.csv"))):
        name = os.path.basename(f)[: -len("_objects.csv")]
        a_path = os.path.join(a_dir, name + ".tif")
        gray, um = load_sem(a_path)
        lab = np.array(Image.open(f[: -len(".csv")] + ".png")).astype(np.int32)[: gray.shape[0]]
        counted = counted_area(lab, um, rule, inset_um, overlay_top(a_path))
        rows = {int(r["id"]): r for r in csv.DictReader(open(f))
                if r["class"] in ("dense", "cauliflower") and counted.get(int(r["id"]), (0,))[0] > 0}
        for i, d in sem_descriptors(lab, um, set(rows)).items():
            parts.append(dict(field=name, sample=name.split("_x")[0], id=i, cauli=rows[i]["class"] == "cauliflower", **d))
    df = pd.DataFrame(parts)
    df = df[df.Da >= sc.min_da].copy()
    df["cls"] = sc.predict(df[FEAT].values) + 1
    out = []
    for s, g in df.groupby("sample"):
        area = g.Da ** 2
        r = {"sample": s, "fields": g.field.nunique(), "n": len(g),
             "cauli_area%": 100 * area[g.cauli].sum() / area.sum(),
             "cauli_vol%": 100 * g.Volume[g.cauli].sum() / g.Volume.sum(),
             "cauli_in_C5_vol%": 100 * g.Volume[g.cauli & (g.cls == 5)].sum() / max(g.Volume[g.cauli].sum(), 1e-9)}
        class_shares(g, "sem_", r)
        out.append(r)
    return pd.DataFrame(out)


def join(sync_csv, sem_csv):
    from scipy.stats import spearmanr
    a = pd.read_csv(sync_csv, dtype={"sample": str}); b = pd.read_csv(sem_csv, dtype={"sample": str})
    p = a.merge(b, on="sample", how="inner", suffixes=("_sync", "_sem"))
    if len(p) >= 4:
        for x in ("cauli_vol%", "cauli_area%"):
            for y in ("vol%_C5", "band_vol%_C5", "band_med_Solidity"):
                rho, pv = spearmanr(p[x], p[y])
                print(f"{x:12s} vs {y:18s} rho {rho:+.2f}  p {pv:.3f}  (n={len(p)})")
    else:
        print(f"only {len(p)} paired batches; correlations need at least 4")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync"); ap.add_argument("--sem"); ap.add_argument("--a")
    ap.add_argument("--join", nargs=2, metavar=("SYNC_CSV", "SEM_CSV"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-rule", default="half", choices=("edge", "half", "centroid"),
                    help="border rule for SEM particles (whole-particle rules only; default half = 50%% rule)")
    ap.add_argument("--inset-um", type=float, default=50.0)
    args = ap.parse_args()
    if args.sync:
        t = sync_profile(args.sync)
    elif args.sem:
        t = sem_profile(args.sem, args.a or os.path.join(os.path.dirname(os.path.abspath(args.sem)), "A_raw"), args.frame_rule, args.inset_um)
    elif args.join:
        t = join(*args.join)
    else:
        ap.error("give --sync, --sem or --join")
    t.round(4).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(t)} rows)")


if __name__ == "__main__":
    main()
