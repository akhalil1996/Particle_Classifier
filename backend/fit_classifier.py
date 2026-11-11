"""Fit the particle classifier that sits on top of the U-Net, check it on samples it has not seen,
and write the model card (.json) next to the network file.

    python backend/fit_classifier.py --raw <folder of SEM images> --labels <folder of label files>
                                     --model models/cauliflower_unet.onnx

Run it after new fields have been labelled, or after the network itself was retrained and
exported with export_onnx.py. The app reads the card automatically.

Label files, one pair per image <name>:
    <name>_objects.png   16-bit label map of the particles (as written by classify_folder.py / the labelling bench)
    <name>_objects.csv   one row per labelled particle with columns
                           id, class (cauliflower | dense; anything else is skipped)
                         and optionally
                           fraction        cauliflower share of the object's area, 0..1 (touching groups)
                           touching_group  1 if the object is several touching particles
                           first_label     'uncertain' marks a particle that was hard to call

The check is leave-one-sample-out: every sample (the part of the file name before _x) is scored by a
classifier fitted on the other samples. That is the error to expect on a new lot.
"""
import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cauliquant.segment import load_sem                                         # noqa: E402
from cauliquant.features import particle_table                                  # noqa: E402
from cauliquant.classifier import network_evidence, feature_row, probabilities, FEATURES, card_path, load_card  # noqa: E402
from cauliquant.infer import UNetONNX                                           # noqa: E402
from cauliquant.analysis import sample_of                                       # noqa: E402
from cauliquant import qc                                                       # noqa: E402


def auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", required=True, help=".onnx network file; the card is written next to it")
    ap.add_argument("--min-ecd", type=float, default=30.0, help="smallest labelled particle used, um")
    ap.add_argument("--name", default=f"CauliQuant classification model {dt.date.today():%Y-%m}")
    ap.add_argument("--kv", type=float, default=15)
    ap.add_argument("--spot", type=float, default=50)
    ap.add_argument("--scan", default="Slow(64)")
    ap.add_argument("--magnification", type=float, default=100)
    ap.add_argument("--C", type=float, default=0.3, help="regularisation of the logistic model")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    net = UNetONNX(a.model)
    rows, meta, fields = [], [], []
    for cs in sorted(glob.glob(os.path.join(a.labels, "*_objects.csv"))):
        name = os.path.basename(cs)[: -len("_objects.csv")]
        img = next((p for e in (".tif", ".tiff", ".png", ".jpg") for p in [os.path.join(a.raw, name + e)] if os.path.exists(p)), None)
        if img is None:
            print("no image for", name)
            continue
        gray, um = load_sem(img)
        lab = np.array(Image.open(os.path.join(a.labels, name + "_objects.png"))).astype(np.int32)
        gray = gray[: lab.shape[0]]
        lab_of = {}
        for r in csv.DictReader(open(cs)):
            if r.get("source", "human") != "human" or r["class"] not in ("cauliflower", "dense"):
                continue
            fr = float(r["fraction"]) if r.get("fraction") not in (None, "") else (1.0 if r["class"] == "cauliflower" else 0.0)
            lab_of[int(r["id"])] = (fr, r.get("touching_group", "0") in ("1", "True"), r.get("first_label", ""))
        score, p3, p5 = network_evidence(net.predict_probs(gray), lab)
        fields.append(qc.field_metrics(gray, lab, um))
        for r in particle_table(gray, lab, um):
            i = r["id"]
            if i not in lab_of or r["ecd_um"] < a.min_ecd or r["edge"]:
                continue
            r.update({"score": float(score[i]), "pix0.3": float(p3[i]), "pix0.5": float(p5[i])})
            fr, grp, first = lab_of[i]
            rows.append(r)
            meta.append(dict(field=name, sample=sample_of(name), frac=fr, group=grp, first=first, area=r["area_um2"]))
        print(f"{name}: {len(rows)} labelled particles so far", flush=True)

    X = np.array([feature_row(r) for r in rows])
    frac = np.array([m["frac"] for m in meta])
    area = np.array([m["area"] for m in meta])
    y = (frac >= 0.5).astype(int)
    sure = np.isin(frac, (0.0, 1.0))
    single = ~np.array([m["group"] for m in meta])
    samples = np.array([m["sample"] for m in meta])
    fld = np.array([m["field"] for m in meta])

    def fit(tr):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=a.C, max_iter=3000).fit(np.clip(sc.transform(X[tr]), -6, 6), y[tr])
        return dict(features=FEATURES, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=m.coef_[0].tolist(),
                    intercept=float(m.intercept_[0]))

    val = {}
    uniq = sorted(set(samples))
    if len(uniq) >= 2:
        pr = np.zeros(len(rows))
        for s in uniq:
            te = samples == s
            pr[te] = probabilities([rows[i] for i in np.where(te)[0]], {"classifier": fit(~te & sure)})
        sp = pr[single] > 0.5
        ys = y[single]
        tp = (sp & (ys == 1)).sum()
        P, R = tp / max(sp.sum(), 1), tp / max(ys.sum(), 1)
        est = lambda m: 100 * (pr[m] * area[m]).sum() / area[m].sum()
        ref = lambda m: 100 * (frac[m] * area[m]).sum() / area[m].sum()
        fe = np.array([est(fld == f) - ref(fld == f) for f in sorted(set(fld))])
        be = np.array([est(samples == s) - ref(samples == s) for s in uniq])
        first = np.array([m["first"] for m in meta])
        val = dict(basis=f"{len(set(fld))} fields of {len(uniq)} samples; each sample scored by a classifier fitted on the "
                         f"others; particles of {a.min_ecd:.0f} um or larger",
                   particles=int(single.sum()), auc=round(auc(ys, pr[single]), 3), precision=round(float(P), 2),
                   recall=round(float(R), 2), f1=round(float(2 * P * R / max(P + R, 1e-9)), 2),
                   field_mae_pp=round(float(np.abs(fe).mean()), 1), field_bias_pp=round(float(fe.mean()), 1),
                   batch_rms_error_pp=round(float(np.sqrt((be ** 2).mean())), 1),
                   batch_max_error_pp=round(float(np.abs(be).max()), 1))
        b = single & (first == "uncertain")
        c = single & ~(first == "uncertain")
        if b.any():
            val.update(accuracy_clear_particles=round(float(((pr[c] > 0.5) == (y[c] == 1)).mean()), 3),
                       accuracy_borderline_particles=round(float(((pr[b] > 0.5) == (y[b] == 1)).mean()), 3),
                       borderline_share_of_particles=round(float(b.sum() / single.sum()), 2))
        print(json.dumps(val, indent=1))
        for s, e in zip(uniq, be):
            print(f"  sample {s}: reference {ref(samples == s):.1f} %, estimate {ref(samples == s) + e:.1f} %")
    else:
        print("Only one sample: no held-out check possible. Label fields from at least two samples.")

    old = load_card(a.model)
    card = dict(name=a.name, created=f"{dt.date.today():%Y-%m-%d}",
                network=dict(old.get("network", {}), file=os.path.basename(a.model),
                             sha256=hashlib.sha256(open(a.model, "rb").read()).hexdigest()),
                classifier=dict(fit(sure), type="logistic regression on standardised features",
                                fitted_on=f"{int(sure.sum())} reference-labelled particles ({a.min_ecd:.0f} um or larger) "
                                          f"from {len(set(fld))} fields of {len(uniq)} samples"),
                validated=dict(kv=a.kv, magnification=a.magnification, spot=a.spot, scan=a.scan, signal="SE", um_px=0.496),
                limits=dict(coverage_max_pct=float(np.ceil(max(f["coverage_pct"] for f in fields) + 1)),
                            edge_sharpness_min=round(0.85 * min(f["edge_sharpness"] for f in fields), 3),
                            saturated_max_pct=float(np.ceil(max(5.0, max(f["saturated_pct"] for f in fields) + 2))),
                            min_particles=50),
                borderline=[0.35, 0.65], validation=val)
    json.dump(card, open(card_path(a.model), "w"), indent=1)
    print("model card written:", card_path(a.model))


if __name__ == "__main__":
    main()
