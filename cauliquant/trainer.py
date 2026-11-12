"""Refine the particle classifier with your own judgements, on this PC.

The U-Net stays as it is. What gets refitted is the logistic classifier on top
of it (models/cauliflower_unet.json), which turns the network score plus shape
and surface-texture measures into a cauliflower probability. Refitting takes a
second and needs no extra software.

Training set = the reference particles the shipped classifier was fitted on
(models/classifier_base.csv) + particles judged in earlier published rounds
(models/classifier_user.csv) + particles judged on this PC since then. A
photo submitted twice replaces its earlier judgements.

The refined card is written to the user folder and used in place of the
shipped one while the network file is the same. Publishing it (github_publish)
puts the card and the judged particles into the repository, so the next
installer carries them.
"""
import csv
import datetime as dt
import hashlib
import json
import os

import numpy as np

from .classifier import FEATURES, feature_row, load_card, card_path
from .resources import resource, DEFAULT_UNET, user_dir, local_card_path

C_REG = 0.3                 # same regularisation as the shipped classifier
BASE_CSV = resource("models", "classifier_base.csv")
SHIPPED_USER_CSV = resource("models", "classifier_user.csv")
COLUMNS = ["image", "image_sha1", "particle", "cauliflower", "ecd_um", "model_probability", "changed", "date"] + FEATURES


def local_csv():
    return os.path.join(user_dir(), "classifier_user.csv")


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ fitting

def fit_logistic(X, y, w=None, C=C_REG, iters=50):
    """L2 logistic regression on standardised, clipped features (the shipped classifier's recipe).

    Minimises 0.5*|coef|^2 + C * sum(w * logloss) with Newton steps; the intercept is not penalised.
    Gives the same coefficients as scikit-learn's LogisticRegression(C=C) without needing it.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    mean = np.average(X, axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = np.clip((X - mean) / scale, -6, 6)
    A = np.hstack([Z, np.ones((len(Z), 1))])
    b = np.zeros(A.shape[1])
    reg = np.ones(A.shape[1]) / C
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A @ b)))
        g = A.T @ (w * (p - y)) + reg * b
        H = (A * (w * p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-10:
            break
    return dict(features=list(FEATURES), mean=mean.tolist(), scale=scale.tolist(), coef=b[:-1].tolist(),
                intercept=float(b[-1]))


def predict(clf, X):
    z = np.clip((np.asarray(X, float) - np.array(clf["mean"])) / np.array(clf["scale"]), -6, 6)
    return 1 / (1 + np.exp(-(z @ np.array(clf["coef"]) + clf["intercept"])))


# ------------------------------------------------------------- training set

def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def judged_rows():
    """Your judged particles: published rounds plus this PC, newest submission of each photo wins."""
    local = _read(local_csv())
    mine = {r["image_sha1"] for r in local}
    return [r for r in _read(SHIPPED_USER_CSV) if r["image_sha1"] not in mine] + local


def training_set():
    base = _read(BASE_CSV)
    user = judged_rows()
    Xb = np.array([[float(r[k]) for k in FEATURES] for r in base]).reshape(-1, len(FEATURES))
    yb = np.array([int(r["cauliflower"]) for r in base])
    Xu = np.array([[float(r[k]) for k in FEATURES] for r in user]).reshape(-1, len(FEATURES))
    yu = np.array([int(r["cauliflower"]) for r in user])
    iu = np.array([r["image_sha1"] for r in user])
    return Xb, yb, Xu, yu, iu


def judgement_rows(result, judged, path_sha1):
    """Training rows from one analysed photo and your judgement of each particle.

    judged: {particle id: "cauliflower" | "dense"}; particles left out (skipped, bad
    outline) are not used. Particles that touch the image edge are not used either,
    since their outline is cut.
    """
    today = f"{dt.date.today():%Y-%m-%d}"
    out = []
    for r in result["rows"]:
        lab = judged.get(r["id"])
        if lab not in ("cauliflower", "dense") or r.get("edge"):
            continue
        model = r["class"]
        row = dict(image=result["name"], image_sha1=path_sha1, particle=r["id"], cauliflower=int(lab == "cauliflower"),
                   ecd_um=round(r["ecd_um"], 2), model_probability=r.get("probability", ""),
                   changed=int(lab != model), date=today)
        row.update({k: f"{v:.6g}" for k, v in zip(FEATURES, feature_row(r))})
        out.append(row)
    return out


# ------------------------------------------------------------------- refit

def active_card(model_path=DEFAULT_UNET):
    return load_card(model_path)


def refit(model_path=DEFAULT_UNET, weight=2.0):
    """Refit on base + all judged particles; write the refined card to the user folder. Returns a report dict."""
    old = load_card(model_path)
    Xb, yb, Xu, yu, iu = training_set()
    X = np.vstack([Xb, Xu]) if len(Xu) else Xb
    y = np.concatenate([yb, yu]) if len(yu) else yb
    w = np.concatenate([np.ones(len(yb)), np.full(len(yu), float(weight))])
    new = fit_logistic(X, y, w)
    rep = dict(base_particles=int(len(yb)), your_particles=int(len(yu)), your_photos=int(len(set(iu.tolist()))),
               weight=float(weight))
    if old.get("classifier"):
        acc = lambda clf, XX, yy: float(((predict(clf, XX) > 0.5) == (yy == 1)).mean()) if len(yy) else float("nan")
        rep.update(base_agree_before=acc(old["classifier"], Xb, yb), base_agree_after=acc(new, Xb, yb),
                   your_agree_before=acc(old["classifier"], Xu, yu), your_agree_after=acc(new, Xu, yu))
        # each of your photos scored by a classifier that never saw it
        photos = sorted(set(iu.tolist()))
        if len(photos) >= 2:
            hit = []
            for ph in photos:
                te = iu == ph
                clf = fit_logistic(np.vstack([Xb, Xu[~te]]), np.concatenate([yb, yu[~te]]),
                                   np.concatenate([np.ones(len(yb)), np.full(int((~te).sum()), float(weight))]))
                hit.append((predict(clf, Xu[te]) > 0.5) == (yu[te] == 1))
            rep["your_agree_unseen"] = float(np.concatenate(hit).mean())
    card = json.loads(json.dumps(old))
    rev = int(card.get("revision", 0)) + 1
    card.update(revision=rev, created=f"{dt.date.today():%Y-%m-%d}")
    card["classifier"] = dict(new, type="logistic regression on standardised features",
                              fitted_on=f"{len(yb)} reference-labelled particles + {len(yu)} particles judged in the "
                                        f"developer tab on {rep['your_photos']} photos (weight {weight:g})")
    card["refined"] = dict(revision=rev, date=card["created"], base_particles=len(yb), judged_particles=len(yu),
                           judged_photos=rep["your_photos"], weight=float(weight))
    if len(yu):
        card.setdefault("validation", {})["note"] = (
            "Held-out figures above were measured on the reference fields before the developer-tab refinement.")
    tmp = local_card_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=1)
    os.replace(tmp, local_card_path())
    rep["revision"] = rev
    return rep


def submit(result, judged, image_path, model_path=DEFAULT_UNET, weight=2.0):
    """Store your judgement of one photo (replacing an earlier one of the same photo) and refit."""
    sha = file_sha1(image_path)
    rows = judgement_rows(result, judged, sha)
    keep = [r for r in _read(local_csv()) if r["image_sha1"] != sha]
    _write(local_csv(), keep + rows)
    rep = refit(model_path, weight)
    ch = [r for r in rows if int(r["changed"])]
    rep.update(photo_particles=len(rows), photo_changed=len(ch))
    if ch:
        clf = load_card(model_path)["classifier"]
        Xc = np.array([[float(r[k]) for k in FEATURES] for r in ch])
        yc = np.array([int(r["cauliflower"]) for r in ch])
        rep["changed_now_agree"] = int(((predict(clf, Xc) > 0.5) == (yc == 1)).sum())
    return rep


def revert():
    """Back to the shipped classifier: removes the refined card (your judged particles stay stored)."""
    if os.path.exists(local_card_path()):
        os.remove(local_card_path())


def shipped_card_path(model_path=DEFAULT_UNET):
    return card_path(model_path)
