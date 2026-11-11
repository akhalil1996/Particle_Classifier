"""Copy labels from the labeling bench into folder B as human labels.

    python import_labels.py --labels cauliflower_labels.csv --a A_raw --b B_marked

The CSV is the file from the bench's "Download labels" button. Run
classify_folder.py first so folder B has the particle ids the bench used.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))  # repo root, for cauliquant
import argparse
import csv
import os
import numpy as np
from PIL import Image

from cauliquant import marks
from cauliquant.segment import load_sem

BENCH = {"Cauliflower": "cauliflower", "Dense": "dense", "Uncertain": "uncertain", "Bad outline": "bad"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    by_field = {}
    with open(args.labels, newline="") as fh:
        for r in csv.DictReader(fh):
            by_field.setdefault(r["field"], {})[r["object"]] = BENCH[r["class"]]
    for name, labels in sorted(by_field.items()):
        rows = marks.read_rows(args.b, name)
        a_path = os.path.join(args.a, name + ".tif")
        if rows is None or not os.path.exists(a_path):
            print(f"skip {name}: run classify_folder.py on it first")
            continue
        n = 0
        for r in rows:
            if r["id"] in labels:
                r["class"], r["source"], r["score"] = labels[r["id"]], "human", ""
                n += 1
        gray, _ = load_sem(a_path)
        lab = np.array(Image.open(marks.paths(args.b, name)["objects"])).astype(np.int32)
        marks.write(args.b, name, gray, lab, rows)
        print(f"{name}: {n} particles set by hand")


if __name__ == "__main__":
    main()
