"""Figures of the morphology report, drawn from whatever files were analysed.

    1  example particles of each shape class        Microtrac report silhouettes
    2  shape class against particle size            Microtrac table particles (else SEM particles)
    3  laser diffraction D10 / D50 / D90 per batch  size reports, at least two batches
    4  class shares per sample, by number / volume  Microtrac tables (else reports, else SEM)
    5  shape at fixed size, 30-100 um, per batch    Microtrac tables, at least two batches
    6  correlation between descriptors              Microtrac table particles

A figure is left out when the files it needs were not opened; figures() says which.
"""
import io
import re

import numpy as np

from .shape_classes import FEAT, BAND

COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NAMES = ["C1 Compact", "C2 Elongated smooth", "C3 Blocky rough", "C4 Elongated concave", "C5 Branched/agglomerate"]
STYLE = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
         "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
         "savefig.dpi": 170, "font.family": "DejaVu Sans"}
SIZE_BINS = [10, 20, 30, 50, 75, 100, 150, 500]
DESC5 = ["Solidity", "Convexity", "Circularity", "Extent", "W/L Aspect Ratio", "Transparency"]
CORR = ["Circularity", "Sphericity", "Compactness", "Roundness", "Ellipse Ratio", "W/L Aspect Ratio", "Ellipticity",
        "Solidity", "Concavity", "Convexity", "Extent", "Transparency", "Da"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    return plt


def _png(fig):
    plt = _plt()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


RUN_STYLE = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D")]
RUN_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]


def _axis(batches):
    """x position of each batch: numbered batches at their number, the others after them, one step apart."""
    from .analysis import _natural
    names = sorted(set(batches), key=_natural)
    num = [b for b in names if re.fullmatch(r"\d+", b)]
    other = [b for b in names if b not in num]
    if len(num) >= 2:
        pos = {b: float(b) for b in num}
        top = max(pos.values())
        span = max(pos.values()) - min(pos.values())
        gap = max(1.0, span / 12)
        pos.update({b: top + gap * (i + 1.5) for i, b in enumerate(other)})
        return pos, num
    return {b: float(i) for i, b in enumerate(names)}, []


def _runs(items):
    """Measurement runs to draw as separate series: the dates, when some batch was measured more than once."""
    by = {}
    for b, r in items:
        by.setdefault(b, set()).add(r)
    if any(len(v) > 1 for v in by.values()):
        runs = sorted({r for _, r in items})
        if len(runs) <= len(RUN_STYLE):
            return runs
    return [None]


def _ticks(a, pos, rot_long=True):
    names = sorted(pos, key=pos.get)
    a.set_xticks([pos[n] for n in names])
    long = max(map(len, names)) > 4
    a.set_xticklabels(names, rotation=60 if long or len(names) > 16 else 0, ha="right" if long else "center",
                      fontsize=7)


def _by_sample(results, source):
    out = {}
    for r in results:
        if r.get("source", "sem") == source:
            out.setdefault(r["label"], []).extend(r["rows"])
    return out


def _bootmed_se(x, n=300, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    return float(np.std([np.median(rng.choice(x, len(x))) for _ in range(n)]))


# ---------------------------------------------------------------- figures

def fig_exemplars(results):
    pools = {k: [] for k in range(1, 6)}
    for r in results:
        for k, ims in (r.get("exemplars") or {}).items():
            pools[int(k)] += ims
    if not any(pools.values()):
        return None
    rng = np.random.default_rng(3)
    plt = _plt()
    fig, ax = plt.subplots(5, 12, figsize=(10, 4.6))
    for c in range(5):
        ims = pools[c + 1]
        pick = rng.choice(len(ims), size=min(12, len(ims)), replace=False) if ims else []
        for j in range(12):
            a = ax[c, j]
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            for sp in a.spines.values():
                sp.set_visible(False)
            if j < len(pick):
                a.imshow(ims[pick[j]])
        ax[c, 0].set_ylabel(NAMES[c].replace(" ", "\n", 1), rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Example particles from each class (from the Microtrac particle reports)", fontsize=10, x=0.02,
                 ha="left")
    fig.tight_layout()
    return _png(fig)


def fig_class_vs_size(results):
    rows, src = [], "Microtrac particle tables"
    for s in ("table", "sem"):
        rows = [p for r in results if r.get("source", "sem") == s for p in r["rows"] if p.get("shape_class")]
        if rows:
            src = "Microtrac particle tables" if s == "table" else "SEM images"
            break
    if len(rows) < 50:
        return None
    da = np.array([p["Da"] for p in rows]); cls = np.array([p["shape_class"] for p in rows])
    labs, share = [], []
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        m = (da >= lo) & (da < hi)
        labs.append(f"{lo}–{hi}")
        share.append([100 * float((cls[m] == k).mean()) if m.any() else 0 for k in range(1, 6)])
    share = np.array(share)
    plt = _plt()
    fig, a = plt.subplots(figsize=(7, 3.4))
    bottom = np.zeros(len(labs))
    for c in range(5):
        a.bar(labs, share[:, c], bottom=bottom, color=COL[c], edgecolor="white", linewidth=1.5, label=NAMES[c],
              width=0.75)
        bottom += share[:, c]
    a.set_xlabel("Area-equivalent diameter Da (µm)"); a.set_ylabel("% of particles")
    a.set_title(f"Shape class against particle size ({len(rows):,} particles, {src})", loc="left", fontsize=10)
    a.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    a.grid(axis="x", visible=False); a.set_ylim(0, 100)
    fig.tight_layout()
    return _png(fig)


def fig_psd(summary):
    rows = [s for s in summary if s["source"] == "size" and np.isfinite(s.get("LD_D50", np.nan))]
    if len(rows) < 2:
        return None
    pos, num = _axis([s["batch"] for s in rows])
    runs = _runs([(s["batch"], s["run"]) for s in rows])
    plt = _plt()
    fig, a = plt.subplots(figsize=(7.4, 3.4))
    for ri, run in enumerate(runs):
        rr = sorted([s for s in rows if run is None or s["run"] == run], key=lambda s: pos[s["batch"]])
        ls, _ = RUN_STYLE[ri]
        for k, c, m in (("LD_D90", "#6250d6", "^"), ("LD_D50", "#2a78d6", "o"), ("LD_D10", "#1baf7a", "s")):
            xs = np.array([pos[s["batch"]] for s in rr]); ys = np.array([s.get(k, np.nan) for s in rr])
            line = np.array([s["batch"] in num for s in rr]) if num else np.ones(len(rr), bool)
            a.plot(xs[line], ys[line], ls, color=c, lw=2, marker=m, ms=5, mfc=c if ri == 0 else "white",
                   label=k[3:] if ri == 0 else None)
            a.plot(xs[~line], ys[~line], ls="none", color=c, marker=m, ms=6, mfc=c if ri == 0 else "white")
    if runs != [None]:
        for ri, run in enumerate(runs):
            a.plot([], [], RUN_STYLE[ri][0], color="#52514e", marker="o", mfc="#52514e" if ri == 0 else "white",
                   label=f"measured {run}" if run else "undated")
    a.set_xlabel("Batch"); a.set_ylabel("Laser diffraction size (µm)")
    a.set_title("Particle size by batch (volume distribution, laser diffraction)", loc="left", fontsize=10)
    _ticks(a, pos)
    h, l = a.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, ncol=len(l), fontsize=7.5, frameon=False, loc="lower center")
    return _png(fig)


def fig_shares(summary):
    for src, title in (("table", "Microtrac particle tables"), ("gallery", "Microtrac particle reports (particles shown)"),
                       ("sem", "SEM images")):
        rows = [s for s in summary if s["source"] == src and s.get("n_counted")]
        if rows:
            break
    else:
        return None
    labels = [s["sample"] for s in rows]
    panels = [("num_pct_C", "By number")] + ([] if src == "gallery" else [("vol_pct_C", "By volume (mass-relevant)")])
    plt = _plt()
    stack = len(rows) > 14 and len(panels) == 2           # many samples: one panel above the other
    if stack:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8.2), squeeze=False)
        axs = axs.T
    else:
        fig, axs = plt.subplots(1, len(panels), figsize=(10 if len(panels) == 2 else 6, 4.4), squeeze=False)
    for a, (pref, tt) in zip(axs[0], panels):
        bottom = np.zeros(len(rows))
        for c in range(5):
            v = np.array([s.get(f"{pref}{c + 1}", 0.0) for s in rows])
            a.bar(labels, v, bottom=bottom, color=COL[c], edgecolor="white", linewidth=1.2, width=0.8, label=NAMES[c])
            bottom += v
        a.set_title(tt, loc="left", fontsize=10); a.tick_params(axis="x", labelsize=7, rotation=90)
        a.grid(axis="x", visible=False); a.set_ylim(0, 100)
    axs[0][0].set_ylabel("% of particles ≥10 µm")
    h, l = axs[0][-1].get_legend_handles_labels()
    fig.suptitle(f"Shape class shares per sample ({title})", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05 if stack else 0.08, 1, 1))
    fig.legend(h, l, ncol=5, fontsize=7, frameon=False, loc="lower center")
    return _png(fig)


def fig_band_descriptors(results):
    groups = {}
    for r in results:
        if r.get("source") == "table":
            groups.setdefault((r["sample"], r.get("run", "")), []).extend(
                p for p in r["rows"] if BAND[0] <= p["Da"] < BAND[1])
    if len({b for b, _ in groups}) < 2:
        return None
    pos, num = _axis([b for b, _ in groups])
    runs = _runs(list(groups))
    plt = _plt()
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.4))
    for a, k in zip(axs.flat, DESC5):
        drawn = False
        for ri, run in enumerate(runs):
            keys = sorted([g for g in groups if run is None or g[1] == run], key=lambda g: pos[g[0]])
            vals = [[p[k] for p in groups[g] if k in p] for g in keys]
            ok = [i for i, v in enumerate(vals) if len(v) >= 5]
            if not ok:
                continue
            drawn = True
            xs = np.array([pos[keys[i][0]] for i in ok]); ys = np.array([np.median(vals[i]) for i in ok])
            es = np.array([1.96 * _bootmed_se(vals[i]) for i in ok])
            line = np.array([keys[i][0] in num for i in ok]) if num else np.ones(len(ok), bool)
            c = RUN_COL[ri]
            lab = (f"measured {run}" if run else "undated") if run is not None else None
            a.errorbar(xs[line], ys[line], yerr=es[line], fmt="o" + RUN_STYLE[ri][0], color=c, lw=1.5, ms=4.5,
                       capsize=2, ecolor="#8a8984", label=lab)
            a.errorbar(xs[~line], ys[~line], yerr=es[~line], fmt="D", color=c, ms=5, capsize=2, ecolor="#8a8984")
        if not drawn:
            a.set_visible(False)
            continue
        a.set_title(k, loc="left", fontsize=9); a.tick_params(labelsize=7)
        _ticks(a, pos)
    if runs != [None]:
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, ncol=len(runs), fontsize=7.5, frameon=False, loc="lower center")
    fig.suptitle(f"Shape at fixed size ({BAND[0]:.0f}–{BAND[1]:.0f} µm particles): median ±95 % CI per batch",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05 if runs != [None] else 0, 1, 1))
    return _png(fig)


def fig_redundancy(results):
    rows = [p for r in results if r.get("source") == "table" for p in r["rows"]]
    cols = [c for c in CORR if rows and c in rows[0]]
    if len(rows) < 50 or len(cols) < 4:
        return None
    from scipy.stats import spearmanr
    M = np.array([[p.get(c, np.nan) for c in cols] for p in rows], float)
    M = M[np.isfinite(M).all(1)]
    C = spearmanr(M).correlation
    plt = _plt()
    fig, a = plt.subplots(figsize=(6.4, 5.4))
    im = a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1); a.grid(False)
    a.set_xticks(range(len(cols))); a.set_xticklabels(cols, rotation=60, ha="right", fontsize=7)
    a.set_yticks(range(len(cols))); a.set_yticklabels(cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            a.text(j, i, f"{C[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=5.5,
                   color="#0b0b0b" if abs(C[i, j]) < 0.7 else "white")
    a.set_title(f"Spearman correlation between descriptors ({len(M):,} particles)", loc="left", fontsize=10)
    fig.colorbar(im, shrink=0.7)
    fig.tight_layout()
    return _png(fig)


def figures(results, summary):
    """[(number, title, png bytes or None, what it needs)] for the morphology report."""
    from .analysis import sample_labels
    sample_labels(results)
    spec = [
        (1, "Example particles of each shape class", lambda: fig_exemplars(results),
         "needs a Microtrac particle report (PDF or page pictures)"),
        (2, "Shape class against particle size", lambda: fig_class_vs_size(results),
         "needs a Microtrac particle table or SEM images"),
        (3, "Particle size by batch", lambda: fig_psd(summary), "needs size reports of at least two batches"),
        (4, "Shape class shares per sample", lambda: fig_shares(summary), "needs particles with shape classes"),
        (5, "Shape at fixed size per batch", lambda: fig_band_descriptors(results),
         "needs Microtrac particle tables of at least two batches"),
        (6, "Correlation between descriptors", lambda: fig_redundancy(results), "needs a Microtrac particle table"),
    ]
    out = []
    for n, title, fn, need in spec:
        try:
            png = fn()
        except Exception as e:                       # a figure never stops the report
            png, need = None, f"could not be drawn ({type(e).__name__}: {e})"
        out.append((n, title, png, need))
    return out
