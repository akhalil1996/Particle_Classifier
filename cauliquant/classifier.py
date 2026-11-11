"""Particle classifier: network evidence plus shape and surface texture -> cauliflower probability.

The U-Net looks at pixels. Its per-particle score ranks particles well but is
not a probability, and on a new SEM session it tends to sit low, so a fixed
cut-off on it under-calls cauliflower. This module combines the score with
measures a person uses when sorting particles (how knobbly the surface is at
several scales, shadowed gaps, outline shape) in a logistic model fitted on
reference-labelled particles. The fitted model travels with the network as a
"model card" (a .json file next to the .onnx file), together with the SEM
settings it was validated on and its measured error on batches it never saw.

A network file without a card still works: the raw network score is used.
"""
import json
import os

import numpy as np

FEATURES = ["lg", "pix0.3", "pix0.5", "solidity", "roughness", "elong", "lecd", "holes", "extent",
            "rel1.5", "rel3", "rel6", "rel12", "grad", "peak_density", "dark", "bright", "lap", "gn", "istd"]

DEFAULT_CARD = {
    "name": "no model card", "classifier": None,
    "validated": {"kv": 15, "magnification": 100, "spot": 50, "scan": "Slow(64)", "signal": "SE", "um_px": 0.496},
    "limits": {"coverage_max_pct": 55, "edge_sharpness_min": 0.08, "saturated_max_pct": 5, "min_particles": 50},
    "borderline": [0.35, 0.65], "validation": {},
}


def card_path(model_path):
    return os.path.splitext(model_path)[0] + ".json"


def load_card(model_path):
    """Model card of a network file; defaults (and no classifier) when there is none."""
    card = dict(DEFAULT_CARD)
    p = card_path(model_path or "")
    if os.path.exists(p):
        try:
            got = json.load(open(p, encoding="utf-8"))
            for k, v in got.items():
                card[k] = {**card[k], **v} if isinstance(v, dict) and isinstance(card.get(k), dict) else v
        except (ValueError, OSError):
            pass
    return card


def network_evidence(probs, lab):
    """Per particle id: mean score, and share of its pixels above 0.3 and 0.5.

    Returns three arrays indexed by particle id.
    """
    n = int(lab.max())
    l = lab.ravel()
    cnt = np.maximum(np.bincount(l, minlength=n + 1), 1).astype(float)
    pc = np.bincount(l, weights=probs[2].ravel().astype(float), minlength=n + 1) / cnt
    pd_ = np.bincount(l, weights=probs[1].ravel().astype(float), minlength=n + 1) / cnt
    score = pc / np.maximum(pc + pd_, 1e-6)
    ratio = (probs[2] / np.maximum(probs[1] + probs[2], 1e-6)).ravel()
    pix3 = np.bincount(l, weights=(ratio > 0.3).astype(float), minlength=n + 1) / cnt
    pix5 = np.bincount(l, weights=(ratio > 0.5).astype(float), minlength=n + 1) / cnt
    return score, pix3, pix5


def feature_row(r):
    """Classifier inputs of one particle row (a dict from particle_table plus score, pix0.3, pix0.5)."""
    s = min(max(r["score"], 1e-4), 1 - 1e-4)
    d = dict(r)
    d["lg"] = float(np.log(s / (1 - s)))
    d["lecd"] = float(np.log(max(r["ecd_um"], 1e-3)))
    return [d[k] for k in FEATURES]


def probabilities(rows, card):
    """Cauliflower probability for each particle row."""
    clf = card.get("classifier")
    if not rows:
        return np.zeros(0)
    if not clf:
        return np.array([r["score"] for r in rows], float)
    assert clf["features"] == FEATURES, "model card was made for another feature set"
    X = np.array([feature_row(r) for r in rows], float)
    z = (X - np.array(clf["mean"])) / np.array(clf["scale"])
    z = np.clip(z, -6, 6)                      # an odd particle cannot run away with the result
    t = z @ np.array(clf["coef"]) + clf["intercept"]
    return 1.0 / (1.0 + np.exp(-t))
