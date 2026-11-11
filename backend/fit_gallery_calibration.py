r"""Fit the correction that puts report-silhouette descriptors on the Microtrac scale.

    python fit_gallery_calibration.py --pair report1.pdf table1.xls --pair report2.pdf table2.xls ...

Each pair is one SYNC "Particles" viewer report (PDF) and the "(All particles).xls"
table exported from the same measurement. The report prints the first particles
of the viewer list, sorted by FWidth; the table holds the same particles with
Microtrac's own descriptors. For every descriptor a straight line is fitted from
the value measured on the silhouette to Microtrac's value. The result is written
to models/microtrac_gallery.json together with a check on reports left out of the
fit (share of particles given the same class as from the table).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
import argparse
import json
import os

import numpy as np

from cauliquant import microtrac as mt
from cauliquant.shape_classes import ShapeClasses, FEAT


def pair(pdf, xls):
    thumbs, head, _ = mt.gallery_from_pdf(pdf)
    t = mt.read_table(xls)
    order = np.argsort(-t["FWidth"], kind="stable")
    ref = np.column_stack([t[f][order] for f in FEAT])[:len(thumbs)]
    raw = []
    for a, _ in thumbs:
        m = mt._largest_filled(mt._ink(a))
        d = mt.outline_descriptors(mt._to_cell(m)) if m is not None else None
        raw.append([d[f] for f in FEAT] if d else [np.nan] * len(FEAT))
    raw = np.array(raw)
    ok = np.isfinite(raw).all(1) & np.isfinite(ref).all(1)
    r = [float(np.corrcoef(raw[ok, j], ref[ok, j])[0, 1]) for j in range(len(FEAT))]
    if min(r) < 0.9:
        raise SystemExit(f"{os.path.basename(pdf)} and {os.path.basename(xls)} do not line up "
                         f"(correlations {np.round(r, 2).tolist()}); are they the same measurement?")
    return raw[ok], ref[ok]


def fit(raws, refs):
    X, Y = np.vstack(raws), np.vstack(refs)
    co = [np.polyfit(X[:, j], Y[:, j], 1) for j in range(len(FEAT))]
    return np.array([c[0] for c in co]), np.array([c[1] for c in co])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("REPORT_PDF", "TABLE_XLS"))
    ap.add_argument("--out", default=mt.GALLERY_CAL)
    args = ap.parse_args()
    data = [pair(p, x) for p, x in args.pair]
    sc = ShapeClasses()
    agree, share_err = [], []
    if len(data) > 1:
        for k, (raw, ref) in enumerate(data):
            a, b = fit([d[0] for i, d in enumerate(data) if i != k], [d[1] for i, d in enumerate(data) if i != k])
            c1 = sc.predict(np.clip(a * raw + b, 0.01, 1.0))
            c2 = sc.predict(ref)
            agree.append(float((c1 == c2).mean()))
            share_err.append(max(abs(100 * float((c1 == j).mean()) - 100 * float((c2 == j).mean())) for j in range(5)))
            print(f"left out {os.path.basename(args.pair[k][0])[:50]}: same class {100 * agree[-1]:.1f} %, "
                  f"largest class-share difference {share_err[-1]:.1f} points")
    a, b = fit([d[0] for d in data], [d[1] for d in data])
    card = {"features": FEAT, "slope": [round(float(v), 5) for v in a], "intercept": [round(float(v), 5) for v in b],
            "cell_px": mt.CELL_PX, "reports": len(data), "particles": int(sum(len(d[0]) for d in data)),
            "held_out_class_agreement": round(float(np.mean(agree)), 3) if agree else None,
            "held_out_max_share_difference_pp": round(float(max(share_err)), 2) if share_err else None}
    json.dump(card, open(args.out, "w"), indent=1)
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
