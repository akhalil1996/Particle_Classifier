"""PDF reports for one image or a batch of images (ReportLab)."""
import datetime as dt
import io
import math

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
                                PageBreak, KeepTogether)
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart

from . import __version__, APP_NAME
from .analysis import (CAULI_RGB, DENSE_RGB, SHAPE_RGB, SHAPE_NAMES, FRAME_RGB,
                       summarise_classification, summarise_morphology)

ACCENT = colors.HexColor("#1F4E79")
_ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_ss["Heading1"], fontSize=17, textColor=ACCENT, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontSize=12.5, textColor=ACCENT, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("body", parent=_ss["BodyText"], fontSize=9, leading=12)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#555555"))

RULE_TEXT = {
    "half": "a particle counts if more than half of its area lies inside the frame and it is not cut by the image border",
    "edge": "a particle counts only if it lies wholly inside the frame",
    "centroid": "a particle counts if its centre lies inside the frame and it is not cut by the image border",
    "ml": "particles wholly inside the frame, weighted by the Miles-Lantuejoul factor",
    "clip": "only the part of each particle inside the frame counts",
}


def _f(v, d=1, unit=""):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return f"{float(v):.{d}f}{unit}"
    except (TypeError, ValueError):
        return str(v)


def _img(arr, width_mm):
    im = Image.open(io.BytesIO(arr)) if isinstance(arr, (bytes, bytearray)) else Image.fromarray(arr)
    im.thumbnail((1400, 1400))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    buf.seek(0)
    w = width_mm * mm
    return RLImage(buf, width=w, height=w * im.height / im.width)


def _table(data, col_widths=None, header=True, align_right_from=1):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#999999")),
          ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT")]
    if header:
        st += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
               ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F9")])]
    t.setStyle(TableStyle(st))
    return t


def _legend(items, label_mm=30):
    cells, widths = [], []
    for rgb, label in items:
        sw = Table([[""]], colWidths=3.5 * mm, rowHeights=3.5 * mm)
        sw.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.Color(*[c / 255 for c in rgb]))]))
        cells += [sw, Paragraph(label, SMALL)]
        widths += [5 * mm, label_mm * mm]
    t = Table([cells], colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
    return t


def _bar(labels, values, title, colour, ymax=None):
    d = Drawing(170 * mm, 60 * mm)
    ch = VerticalBarChart()
    ch.x, ch.y, ch.width, ch.height = 12 * mm, 14 * mm, 150 * mm, 38 * mm
    ch.data = [[0 if (v is None or math.isnan(v)) else v for v in values]]
    ch.categoryAxis.categoryNames = [l[:14] for l in labels]
    ch.categoryAxis.labels.fontSize = 6.5
    ch.categoryAxis.labels.angle = 30 if len(labels) > 6 else 0
    ch.categoryAxis.labels.boxAnchor = "ne" if len(labels) > 6 else "n"
    ch.valueAxis.valueMin = 0
    ch.valueAxis.valueMax = ymax or max(5, (max(ch.data[0]) if ch.data[0] else 5) * 1.2)
    ch.valueAxis.labels.fontSize = 7
    ch.categoryAxis.labels.fontName = ch.valueAxis.labels.fontName = "Helvetica"
    ch.bars[0].fillColor = colors.Color(*[c / 255 for c in colour])
    ch.bars[0].strokeColor = None
    d.add(ch)
    d.add(String(12 * mm, 56 * mm, title, fontSize=8.5, fontName="Helvetica-Bold", fillColor=ACCENT))
    return d


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(15 * mm, 10 * mm, f"{APP_NAME} {__version__}  |  {doc.title}")
    canvas.drawRightString(195 * mm, 10 * mm, f"page {doc.page}")
    canvas.restoreState()


def _settings_rows(r):
    s = r["settings"]
    rows = [["Frame rule", f"{s['frame_rule']} ({RULE_TEXT.get(s['frame_rule'], '')}), inset {s['inset_um']:.0f} um"]]
    if r["kind"] == "classification":
        rows += [["Model", s["model_path"]], ["Cauliflower cut-off", f"score > {s['cutoff']:.2f}"]]
    rows += [["Pixel size", "entered by user: %.4f um/px" % s["um_px_override"] if s["um_px_override"] else "from each image's SEM metadata"],
             ["Resampling", "to 0.496 um/px when the image pixel size differs by more than 5%" if s["resample"] else "off"]]
    return [[Paragraph(f"<b>{a}</b>", BODY), Paragraph(b, BODY)] for a, b in rows]


def _image_section(r):
    st, info = r["stats"], r["image_info"]
    out = [Paragraph(r["name"], H2)]
    meta = f"Sample <b>{r['sample']}</b> &nbsp; | &nbsp; {info['native_size'][0]} x {info['native_size'][1]} px"
    if info.get("magnification"):
        meta += f" &nbsp; | &nbsp; x{info['magnification']}"
    meta += f" &nbsp; | &nbsp; {info['native_um_px']:.3f} um/px ({info['um_source']})"
    if info["scale"] != 1.0:
        meta += f", resampled x{info['scale']:.2f}"
    out.append(Paragraph(meta, SMALL))
    out.append(Spacer(1, 2 * mm))
    imgs = Table([[_img(r["original"], 88), _img(r["analysed"], 88)],
                  [Paragraph("Original", SMALL), Paragraph("Analysed (yellow = counting frame)", SMALL)]],
                 colWidths=[90 * mm, 90 * mm])
    imgs.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    out.append(imgs)
    if r["kind"] == "classification":
        out.append(_legend([(CAULI_RGB, "cauliflower"), (DENSE_RGB, "dense"), (FRAME_RGB, "counting frame")], 28))
        rows = [["Result", "Value"],
                ["Cauliflower, % of particle area", _f(st["cauliflower_area_pct"], 1, " %")],
                ["Cauliflower, % of volume (sphere-equivalent)", _f(st["cauliflower_vol_pct"], 1, " %")],
                ["Particles counted (cauliflower / dense)", f"{st['n_counted']} ({st['n_cauliflower']} / {st['n_dense']})"],
                ["Median size, cauliflower / dense (ECD)", f"{_f(st['median_ecd_cauliflower_um'], 0)} / {_f(st['median_ecd_dense_um'], 0)} um"],
                ["Not counted: below 10 um / outside frame", f"{st['n_small']} / {st['n_outside_frame']}"],
                ["Counting frame", f"{st['frame_um'][0]:.0f} x {st['frame_um'][1]:.0f} um"]]
    else:
        out.append(_legend([(SHAPE_RGB[k], SHAPE_NAMES[k]) for k in range(5)]))
        rows = [["Shape class", "% by number", "% by volume", "% by volume, 30-100 um"]]
        for k in range(1, 6):
            rows.append([SHAPE_NAMES[k - 1], _f(st.get(f"num_pct_C{k}")), _f(st.get(f"vol_pct_C{k}")),
                         _f(st.get(f"band_vol_pct_C{k}"))])
        rows.append(["Particles classified", str(st.get("n_counted", 0)), "", f"{st.get('n_band', 0)} in band"])
        out.append(Spacer(1, 2 * mm))
        out.append(_table(rows, [70 * mm, 30 * mm, 30 * mm, 45 * mm]))
        rows = [["Descriptor (median)", "Value"], ["D50 by number (area-equivalent diameter)", _f(st.get("D50_num_um"), 1, " um")]]
        for k in ("Circularity", "Solidity", "Convexity", "W/L Aspect Ratio", "Extent"):
            rows.append([k, _f(st.get(f"median_{k}"), 3)])
    out.append(Spacer(1, 2 * mm))
    out.append(_table(rows, [100 * mm, 60 * mm]))
    return out


def build_report(results, path, title=None):
    """Write a PDF for one result or a list of results (a batch)."""
    if isinstance(results, dict):
        results = [results]
    kind = results[0]["kind"]
    what = "Particle classification" if kind == "classification" else "Morphology profile"
    title = title or f"{what} report"
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=16 * mm, title=title, author=APP_NAME)
    el = [Paragraph(title, H1),
          Paragraph(f"{APP_NAME} {__version__} &nbsp; | &nbsp; {dt.datetime.now():%Y-%m-%d %H:%M} &nbsp; | &nbsp; "
                    f"{len(results)} image{'s' if len(results) > 1 else ''}", SMALL),
          Spacer(1, 3 * mm)]

    if kind == "classification":
        el.append(Paragraph("Particles above 10 um equivalent diameter are segmented and each is scored by a "
                            "U-Net as porous cauliflower agglomerate or dense particle. The cauliflower share is "
                            "the projected area of cauliflower particles over the area of all counted particles "
                            "inside the counting frame.", BODY))
    else:
        el.append(Paragraph("Particles above 10 um are segmented and their outlines measured with Microtrac-style "
                            "descriptors (circularity, solidity, convexity, W/L aspect ratio, extent). Each particle "
                            "is assigned to one of five shape classes (C1-C5) by a fixed k-means model fitted on "
                            "Microtrac SYNC data. SEM outlines are sharper than dynamic image analysis outlines, so "
                            "the SEM shares indicate trends between samples and are not on the Microtrac scale.", BODY))
    el.append(Spacer(1, 2 * mm))
    el.append(_table(_settings_rows(results[0]), [38 * mm, 142 * mm], header=False, align_right_from=9))

    if len(results) > 1:
        el.append(Paragraph("Summary by sample", H2))
        if kind == "classification":
            summ = summarise_classification(results)
            rows = [["Sample", "Fields", "Particles", "Cauliflower area %", "Field mean %", "Field SD"]]
            for s in summ:
                rows.append([s["sample"], s["fields"], s["particles"], _f(s["cauliflower_area_pct"]),
                             _f(s["field_mean_pct"]), _f(s["field_sd_pct"])])
            el.append(_table(rows, [50 * mm, 18 * mm, 22 * mm, 34 * mm, 28 * mm, 24 * mm]))
            el.append(Paragraph("Cauliflower area % pools all counted particles of a sample's fields.", SMALL))
            el.append(_bar([s["sample"] for s in summ], [s["cauliflower_area_pct"] for s in summ],
                           "Cauliflower area % by sample", CAULI_RGB))
            el.append(Paragraph("Per image", H2))
            rows = [["Image", "Sample", "Particles", "Cauliflower", "Dense", "Cauliflower area %"]]
            for r in results:
                st = r["stats"]
                rows.append([r["name"], r["sample"], st["n_counted"], st["n_cauliflower"], st["n_dense"],
                             _f(st["cauliflower_area_pct"])])
            el.append(_table(rows, [58 * mm, 30 * mm, 20 * mm, 22 * mm, 18 * mm, 30 * mm]))
        else:
            summ = summarise_morphology(results)
            rows = [["Sample", "Fields", "Particles"] + [f"C{k} vol%" for k in range(1, 6)] + ["C5 band vol%", "D50 um"]]
            for s in summ:
                rows.append([s["sample"], s["fields"], s.get("n_counted", 0)] +
                            [_f(s.get(f"vol_pct_C{k}")) for k in range(1, 6)] +
                            [_f(s.get("band_vol_pct_C5")), _f(s.get("D50_num_um"), 0)])
            el.append(_table(rows, [36 * mm, 12 * mm, 16 * mm] + [16 * mm] * 5 + [20 * mm, 16 * mm]))
            el.append(Paragraph("Shares pool all classified particles of a sample's fields. "
                                "Band = particles of 30-100 um area-equivalent diameter.", SMALL))
            el.append(_bar([s["sample"] for s in summ], [s.get("band_vol_pct_C5", float("nan")) for s in summ],
                           "C5 (branched / agglomerate), % of volume in the 30-100 um band", SHAPE_RGB[4]))
    for r in results:
        el.append(PageBreak())
        el.extend(_image_section(r))
    doc.build(el, onFirstPage=_footer, onLaterPages=_footer)
    return path


def export_csv(results, path):
    """Per-particle table of every image in one CSV."""
    import csv
    keys, rows = [], []
    for r in results:
        for p in r["rows"]:
            row = dict(image=r["name"], sample=r["sample"], **p)
            for k in row:
                if k not in keys:
                    keys.append(k)
            rows.append(row)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    return path


def export_summary_csv(results, path):
    import csv
    if results[0]["kind"] == "classification":
        per = [dict(image=r["name"], sample=r["sample"], **{k: v for k, v in r["stats"].items() if k != "frame_um"})
               for r in results]
    else:
        per = [dict(image=r["name"], sample=r["sample"], **{k: v for k, v in r["stats"].items() if k != "frame_um"})
               for r in results]
    keys = []
    for p in per:
        for k in p:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for p in per:
            w.writerow({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in p.items()})
    return path
