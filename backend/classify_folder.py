"""Workflow 1: classify every raw image in folder A into a marked image in folder B.

    python classify_folder.py --a A_raw --b B_marked [--model runs/<run>/model.pt]

Without --model, particles get a draft class from a simple shape and texture
rule. With a trained model, each particle takes the class the network gives
most of its pixels. Classes a person set in the labeling bench (source
"human") are kept as they are.

The area % of each field counts particles by a border rule inside a counting
frame (cauliquant/frame.py). The default is the 50% rule with a 50 um inset:
a particle counts if more than half of it lies inside the frame and it is not
cut by the image border. The frame is drawn on the overlay in yellow, and
B/summary.csv lists every field's result.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))  # repo root, for cauliquant
import argparse
import csv
import glob
import os
import numpy as np

from cauliquant.segment import load_sem, segment, min_area_px, overlay_top
from cauliquant.features import particle_table
from cauliquant import marks
from cauliquant.frame import counted_area, frame_box, RULES
from PIL import Image, ImageDraw


def rule_class(r):
    """Draft rule: cauliflower particles are rough-edged with a busy surface."""
    score = (r["grad_mean"] - 0.044) / 0.012 + (r["roughness"] - 1.7) / 0.5 - (r["elong"] - 1.6) / 0.6
    return ("cauliflower" if score > 1.5 else "dense"), score


def model_classes(model_path, gray, lab):
    import torch
    from cauliquant.unet import load_model, predict_probs
    net = load_model(model_path)
    probs = predict_probs(net, gray)            # (3, H, W): bg, dense, cauliflower
    cls = probs.argmax(0)
    out = {}
    for i in range(1, int(lab.max()) + 1):
        sel = lab == i
        if not sel.any():
            continue
        p_c = float(probs[2][sel].mean())
        p_d = float(probs[1][sel].mean())
        out[i] = ("cauliflower" if p_c > p_d else "dense", p_c / max(p_c + p_d, 1e-6))
    return out


def classify_image(a_path, b_dir, model=None, rule="half", inset_um=50.0):
    name = os.path.splitext(os.path.basename(a_path))[0]
    gray, um_px = load_sem(a_path)
    lab = segment(gray, um_px)
    rows = particle_table(gray, lab, um_px)
    old = {r["id"]: r for r in (marks.read_rows(b_dir, name) or []) if r["source"] == "human"}
    by_model = model_classes(model, gray, lab) if model else {}
    amin = min_area_px(um_px) * um_px ** 2
    top = overlay_top(a_path)
    hidden = set(np.unique(lab[top:]).tolist()) - {0} if top is not None else set()
    for r in rows:
        if r["id"] in hidden:
            r["class"], r["source"], r["score"] = "overlay", "overlay", ""
        elif r["area_um2"] < amin:
            r["class"], r["source"], r["score"] = "small", "size", ""
        elif str(r["id"]) in old:
            r["class"], r["source"], r["score"] = old[str(r["id"])]["class"], "human", ""
        elif model:
            r["class"], s = by_model[r["id"]]
            r["source"], r["score"] = "model", round(s, 3)
        else:
            r["class"], s = rule_class(r)
            r["source"], r["score"] = "draft", round(s, 3)
        for k in ("area_um2", "ecd_um", "solidity", "roughness", "elong", "grad_mean", "relief_mean", "cx", "cy"):
            r[k] = round(r[k], 4)
    mask = marks.write(b_dir, name, gray, lab, rows, ignore_below=top)
    counted = counted_area(lab, um_px, rule, inset_um, top)
    big = [r for r in rows if r["class"] in ("dense", "cauliflower") and counted.get(r["id"], (0,))[0] > 0]
    area = lambda c: sum(counted[r["id"]][0] for r in big if r["class"] == c)
    tot = area("dense") + area("cauliflower")
    fy0, fx0, fy1, fx1 = frame_box(lab.shape, um_px, inset_um, top)
    ov = marks.paths(b_dir, name)["overlay"]
    im = Image.open(ov).convert("RGB")
    ImageDraw.Draw(im).rectangle([fx0, fy0, fx1 - 1, fy1 - 1], outline=(255, 215, 0), width=3)
    im.save(ov, quality=90)
    return dict(image=name, particles=len(big), cauliflower_area_pct=round(100 * area("cauliflower") / tot, 1) if tot else None,
                human=sum(r["source"] == "human" for r in rows), frame_rule=rule, inset_um=inset_um,
                lost=sum(v[1] for v in counted.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="folder of raw SEM images (.tif with .txt metadata)")
    ap.add_argument("--b", required=True, help="output folder for marked images")
    ap.add_argument("--model", help="trained model.pt from workflow 2 (optional)")
    ap.add_argument("--frame-rule", default="half", choices=RULES, help="border rule for the area %% (default half = 50%% rule)")
    ap.add_argument("--inset-um", type=float, default=50.0, help="counting-frame inset from each image side, um (default 50)")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.a, "*.tif")))
    print(f"{len(files)} images in {args.a}")
    out = []
    for f in files:
        s = classify_image(f, args.b, args.model, args.frame_rule, args.inset_um)
        out.append(s)
        print(f"{s['image']:32s} {s['particles']:4d} particles  cauliflower {s['cauliflower_area_pct']}% of area  ({s['human']} set by hand)")
    if out:
        with open(os.path.join(args.b, "summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
