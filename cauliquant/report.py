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

from . import __version__, APP_NAME, COPYRIGHT
from .analysis import (CAULI_RGB, DENSE_RGB, CAULI_SOFT_RGB, DENSE_SOFT_RGB, SHAPE_RGB, SHAPE_NAMES, FRAME_RGB,
                       summarise_classification, summarise_morphology, batch_table)
from .resources import resource

ACCENT = colors.HexColor("#1F4E79")
_ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_ss["Heading1"], fontSize=17, textColor=ACCENT, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontSize=12.5, textColor=ACCENT, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("body", parent=_ss["BodyText"], fontSize=9, leading=12)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#555555"))
WARN = ParagraphStyle("warn", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor("#8A4B08"),
                      backColor=colors.HexColor("#FFF4E0"), borderPadding=4, spaceBefore=4, spaceAfter=4)

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
    im.thumbnail((1200, 1200))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=80)
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
    canvas.drawCentredString(105 * mm, 10 * mm, COPYRIGHT)
    canvas.drawRightString(195 * mm, 10 * mm, f"page {doc.page}")
    canvas.restoreState()


def _logo(name, height_mm):
    import os
    p = resource("assets", name)
    if not os.path.exists(p):
        return Spacer(1, height_mm * mm)
    im = Image.open(p)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        im = bg
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    h = height_mm * mm
    return RLImage(buf, width=h * im.width / im.height, height=h)


def _letterhead():
    """McGill University and P2[AM]2 logos across the top of the first page."""
    t = Table([[_logo("mcgill_logo.png", 11), _logo("p2am2_logo.png", 11)]], colWidths=[90 * mm, 90 * mm],
              hAlign="LEFT")
    t.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "LEFT"), ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#C00000"))]))
    return t


def _validated_text(v):
    if not v:
        return ""
    return (f"x{v.get('magnification', 100):.0f}, {v.get('kv', 15):.0f} kV, spot intensity {v.get('spot', 50):.0f}, "
            f"scan {v.get('scan', '')}, {v.get('signal', 'SE')} detector, {v.get('um_px', 0.496):.3f} um/px")


def _settings_rows(r):
    s = r["settings"]
    rows = [["Frame rule", f"{s['frame_rule']} ({RULE_TEXT.get(s['frame_rule'], '')}), inset {s['inset_um']:.0f} um"]]
    if r["kind"] == "classification":
        m = r.get("model", {})
        model = s["model_path"] + (f" ({m['name']})" if m.get("name") else "")
        if m.get("sha256"):
            model += f", file fingerprint {m['sha256'][:12]}"
        rows += [["Model", model],
                 ["Result", f"{m.get('estimator', 'particle calls')}; a particle is drawn as cauliflower when its "
                            f"probability is above {s['cutoff']:.2f}"]]
        if m.get("validated"):
            rows.append(["Validated for", _validated_text(m["validated"])])
        v = m.get("validation") or {}
        if v:
            txt = (f"on batches the model never saw: one field within about {v.get('field_mae_pp', float('nan')):.0f} "
                   f"points of the reference on average, a pooled batch within about "
                   f"{v.get('batch_rms_error_pp', float('nan')):.0f} points (largest {v.get('batch_max_error_pp', float('nan')):.0f}).")
            if "accuracy_borderline_particles" in v:
                txt += (f" Clear particles are called correctly {100 * v['accuracy_clear_particles']:.0f} % of the time, "
                        f"borderline ones {100 * v['accuracy_borderline_particles']:.0f} %.")
            rows.append(["Measured error", txt])
    rows += [["Pixel size", "entered by user: %.4f um/px" % s["um_px_override"] if s["um_px_override"] else "from each image's SEM metadata"],
             ["Resampling", "to 0.496 um/px when the image pixel size differs by more than 5%" if s["resample"] else "off"]]
    return [[Paragraph(f"<b>{a}</b>", BODY), Paragraph(b, BODY)] for a, b in rows]


def _warnings(r):
    w = r.get("warnings") or []
    if not w:
        return []
    what = "image" if r.get("source", "sem") == "sem" else "file"
    return [Paragraph(f"<b>Check this {what}:</b> " + " ".join(w), WARN)]


def _plain(t):
    """Text for the built-in PDF fonts, which have no micro sign or thin dashes."""
    return t.replace("µ", "u").replace("–", "-").replace("−", "-").replace("·", "-")


def _microtrac_section(r):
    """Page for one Microtrac file: particle report (silhouettes), particle table or size report."""
    st, src = r["stats"], r["source"]
    out = [Paragraph(_plain(r["name"]), H2),
           Paragraph(f"Sample <b>{_plain(r.get('label', r['sample']))}</b> &nbsp; | &nbsp; "
                     f"{_plain(r['path'])}", SMALL), Spacer(1, 2 * mm)]
    if src == "gallery":
        imgs = Table([[_img(r["original"], 88), _img(r["analysed"], 88)],
                      [Paragraph("Particles as printed in the report", SMALL),
                       Paragraph("Coloured by shape class", SMALL)]], colWidths=[90 * mm, 90 * mm])
        imgs.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        out += [imgs, _legend([(SHAPE_RGB[k], SHAPE_NAMES[k]) for k in range(5)]), Spacer(1, 2 * mm)]
        rows = [["Shape class", "% by number"]] + [[SHAPE_NAMES[k - 1], _f(st.get(f"num_pct_C{k}"))] for k in range(1, 6)]
        rows.append(["Particles classified", str(st.get("n_counted", 0))])
        out.append(_table(rows, [118 * mm, 52 * mm]))
        rows = [["Descriptor (median, on the Microtrac scale)", "Value"]]
        rows += [[k, _f(st.get(f"median_{k}"), 3)] for k in ("Circularity", "Solidity", "Convexity", "W/L Aspect Ratio", "Extent")]
    elif src == "table":
        out += [_img(r["analysed"], 175), Spacer(1, 2 * mm)]
        rows = [["Shape class", "% by number", "% by volume", "% by volume, 30-100 um"]]
        for k in range(1, 6):
            rows.append([SHAPE_NAMES[k - 1], _f(st.get(f"num_pct_C{k}")), _f(st.get(f"vol_pct_C{k}")),
                         _f(st.get(f"band_vol_pct_C{k}"))])
        rows.append(["Particles classified", str(st.get("n_counted", 0)), "", f"{st.get('n_band', 0)} in band"])
        out.append(_table(rows, [70 * mm, 30 * mm, 30 * mm, 45 * mm]))
        rows = [["Descriptor (median)", "Value"],
                ["D50 by number (area-equivalent diameter)", _f(st.get("D50_num_um"), 1, " um")],
                ["Particles below 20 um, % by number", _f(st.get("fines_lt20um_num_pct"), 1, " %")]]
        rows += [[k, _f(st.get(f"median_{k}"), 3)] for k in ("Circularity", "Solidity", "Convexity", "W/L Aspect Ratio", "Extent")]
    else:
        out += [_img(r["original"], 150), Spacer(1, 2 * mm)]
        rows = [["Size by volume (laser diffraction)", "Value"]]
        rows += [[f"D{p}", _f(st.get(f"LD_D{p}"), 1, " um")] for p in (10, 50, 90)]
        rows += [["Span, (D90 - D10) / D50", _f(st.get("LD_span"), 2)],
                 ["Mean by volume (MV) / by area (MA) / by number (MN)",
                  f"{_f(st.get('LD_MV'), 1)} / {_f(st.get('LD_MA'), 1)} / {_f(st.get('LD_MN'), 1)} um"]]
    out.append(Spacer(1, 2 * mm))
    out.append(_table(rows, [118 * mm, 52 * mm]))
    out.append(Spacer(1, 2 * mm))
    out.append(Paragraph(_plain(r.get("note", "")), SMALL))
    out.extend(_warnings(r))
    return out


def _image_section(r):
    if r.get("source", "sem") != "sem":
        return _microtrac_section(r)
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
        out.append(_legend([(CAULI_RGB, "cauliflower"), (CAULI_SOFT_RGB, "borderline, cauliflower"),
                            (DENSE_SOFT_RGB, "borderline, dense"), (DENSE_RGB, "dense"), (FRAME_RGB, "frame")], 30))
        rows = [["Result", "Value"],
                ["Cauliflower, % of particle area", _f(st["cauliflower_area_pct"], 1, " %")],
                ["Cauliflower, % of volume (sphere-equivalent)", _f(st["cauliflower_vol_pct"], 1, " %")],
                ["Counting only particles called cauliflower, % of area", _f(st.get("calls_area_pct"), 1, " %")],
                ["Area held by borderline particles", _f(st.get("borderline_area_pct"), 1, " %")],
                ["Particles counted (cauliflower / dense / of which borderline)",
                 f"{st['n_counted']} ({st['n_cauliflower']} / {st['n_dense']} / {st.get('n_borderline', 0)})"],
                ["Median size, cauliflower / dense (ECD)", f"{_f(st['median_ecd_cauliflower_um'], 0)} / {_f(st['median_ecd_dense_um'], 0)} um"],
                ["Not counted: below 10 um / outside frame", f"{st['n_small']} / {st['n_outside_frame']}"],
                ["Counting frame", f"{st['frame_um'][0]:.0f} x {st['frame_um'][1]:.0f} um"],
                ["Image covered by particles / large-particle area in touching clusters",
                 f"{_f(st.get('coverage_pct'), 0)} % / {_f(st.get('touching_area_pct'), 0)} %"]]
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
    out.append(_table(rows, [118 * mm, 52 * mm]))
    out.extend(_warnings(r))
    return out


def _morphology_summary(results, single=False):
    from .figures import figures
    out = []
    summ = summarise_morphology(results)
    if not single:
        bt = batch_table(results)
        has_ld = any("LD_D50" in b for b in bt)
        rows = [["Sample", "Shapes from", "Particles", "C5 num %", "C5 vol %", "C5 vol % 30-100"] +
                (["D10 um", "D50 um", "D90 um"] if has_ld else [])]
        src = {"table": "MT table", "gallery": "MT report", "sem": "SEM"}
        for b in bt:
            shp = b.get("shape_source")
            num_only = shp == "gallery"
            rows.append([_plain(b["sample"]), src.get(shp, "-"), b.get("n_counted", "-") if shp else "-",
                         _f(b.get("num_pct_C5")) if shp else "-",
                         "-" if num_only or not shp else _f(b.get("vol_pct_C5")),
                         "-" if num_only or not shp else _f(b.get("band_vol_pct_C5"))] +
                        ([_f(b.get("LD_D10")), _f(b.get("LD_D50")), _f(b.get("LD_D90"))] if has_ld else []))
        w = [38 * mm, 19 * mm, 16 * mm, 16 * mm, 16 * mm, 25 * mm] + ([16 * mm] * 3 if has_ld else [])
        scale = 180 * mm / sum(w)
        out.append(_table(rows, [x * scale for x in w]))
        out.append(Paragraph("Shapes from: the source of the shape classes for that sample, in this order of "
                             "preference: the Microtrac particle table (every particle, by number and volume), the "
                             "Microtrac particle report (the particles it prints, by number only), SEM images. "
                             "D10, D50 and D90 are laser diffraction sizes by volume. When a batch was measured on "
                             "more than one day, each day is its own sample, named with the date.", SMALL))
        sem_rows = [x for x in summ if x["source"] == "sem"]
        if sem_rows and any(x["source"] in ("table", "gallery") for x in summ):
            out.append(Paragraph("SEM shares are reported per image and in the figures only where no Microtrac "
                                 "data was opened; SEM outlines are not on the Microtrac scale.", SMALL))
    figs = figures(results, summ)
    have = [f for f in figs if f[2]]
    if have:
        out.append(PageBreak())
        out.append(Paragraph("Figures", H2))
        for n, title, png, _ in figs:
            if not png:
                continue
            im = Image.open(io.BytesIO(png))
            w = 180 * mm
            h = w * im.height / im.width
            if h > 210 * mm:
                w, h = w * 210 * mm / h, 210 * mm
            out.append(KeepTogether([Paragraph(f"<b>Figure {n}.</b> {_plain(title)}", BODY), Spacer(1, 1 * mm),
                                     RLImage(io.BytesIO(png), width=w, height=h), Spacer(1, 4 * mm)]))
        miss = [f"Figure {n} ({title.lower()}) {need}" for n, title, png, need in figs if not png]
        if miss:
            out.append(Paragraph("Not drawn for these files: " + "; ".join(_plain(m) for m in miss) + ".", SMALL))
    return out


def _gallery_text():
    from .microtrac import load_calibration
    c = load_calibration()[2]
    if not c.get("held_out_class_agreement"):
        return ""
    return (f" A correction fitted on {c.get('reports', 0)} reports paired with their tables puts the silhouette "
            f"descriptors on the Microtrac scale; on reports left out of the fit, "
            f"{100 * c['held_out_class_agreement']:.0f} % of particles got the same class as from the table and "
            f"class shares differed by at most {c.get('held_out_max_share_difference_pp', 0):.1f} points.")


def build_report(results, path, title=None):
    """Write a PDF for one result or a list of results (a batch)."""
    if isinstance(results, dict):
        results = [results]
    kind = results[0]["kind"]
    what = "Particle classification" if kind == "classification" else "Morphology profile"
    title = title or f"{what} report"
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=16 * mm, title=title, author=APP_NAME)
    if kind == "morphology":
        from .analysis import sample_labels
        sample_labels(results)
    sem = [r for r in results if r.get("source", "sem") == "sem"]
    mt = [r for r in results if r.get("source", "sem") != "sem"]
    what_n = "image" if not mt else "file"
    el = [_letterhead(), Spacer(1, 4 * mm), Paragraph(title, H1),
          Paragraph(f"{APP_NAME} {__version__} &nbsp; | &nbsp; {dt.datetime.now():%Y-%m-%d %H:%M} &nbsp; | &nbsp; "
                    f"{len(results)} {what_n}{'s' if len(results) > 1 else ''} &nbsp; | &nbsp; {COPYRIGHT}", SMALL),
          Spacer(1, 3 * mm)]

    if kind == "classification":
        el.append(Paragraph("Particles above 10 um equivalent diameter are segmented. Each one gets a probability of "
                            "being a porous cauliflower agglomerate, from a U-Net's reading of its surface combined "
                            "with its outline shape and surface relief. The cauliflower share is the "
                            "probability-weighted projected area of all counted particles inside the counting frame: "
                            "every particle adds its area times its probability. Particles with a probability between "
                            "0.35 and 0.65 are reported as borderline and drawn in a lighter shade.", BODY))
    else:
        if sem:
            el.append(Paragraph("<b>SEM images.</b> Particles above 10 um are segmented and their outlines measured "
                                "with Microtrac-style descriptors (circularity, solidity, convexity, W/L aspect "
                                "ratio, extent). Each particle is assigned to one of five shape classes (C1-C5) by a "
                                "fixed k-means model fitted on Microtrac SYNC data. SEM outlines are sharper than "
                                "dynamic image analysis outlines and touching particles are split by the software, so "
                                "the SEM shares rank samples against each other and are not on the Microtrac scale. "
                                "The C5 share moves by several points with how strongly touching particles are "
                                "split; compare samples imaged and analysed the same way.", BODY))
        if mt:
            cal = _gallery_text() if any(r["source"] == "gallery" for r in mt) else ""
            el.append(Paragraph("<b>Microtrac files.</b> A particle table (.xls) is classified from Microtrac's own "
                                "descriptors for every particle and gives shares by number, by volume and in the "
                                "30-100 um band. A particle report (PDF or page pictures) prints each particle as a "
                                "silhouette; the silhouettes are measured again and given the same five classes. "
                                "A report holds only part of the sample and no sizes, so its shares are by number "
                                "of the particles shown and are not comparable with shares by volume." + cal +
                                " A size report gives the laser diffraction percentiles and no shape classes.", BODY))
    el.append(Spacer(1, 2 * mm))
    if sem or kind == "classification":
        el.append(_table(_settings_rows((sem or results)[0]), [38 * mm, 142 * mm], header=False, align_right_from=9))

    if len(results) > 1:
        el.append(Paragraph("Summary by sample", H2))
        if kind == "classification":
            summ = summarise_classification(results)
            rows = [["Sample", "Fields", "Particles", "Cauliflower area %", "95 % interval", "Field SD", "By calls %"]]
            for s in summ:
                ci = f"{_f(s['ci_low_pct'])} to {_f(s['ci_high_pct'])}" if s["fields"] >= 3 else "needs 3 fields"
                rows.append([s["sample"], s["fields"], s["particles"], _f(s["cauliflower_area_pct"]), ci,
                             _f(s["field_sd_pct"]), _f(s["calls_area_pct"])])
            el.append(_table(rows, [40 * mm, 14 * mm, 20 * mm, 32 * mm, 30 * mm, 18 * mm, 22 * mm]))
            note = ("Cauliflower area % pools all counted particles of a sample's fields. The 95 % interval shows how "
                    "far the pooled value could move if other fields of the same stub had been imaged (fields "
                    "resampled 2000 times); it does not include the method error given under Measured error above. "
                    "By calls = counting only particles drawn as cauliflower.")
            need = [f"{s['sample']}: about {s['fields_for_5pp']} fields" for s in summ
                    if s.get("fields_for_5pp") and s["fields"] >= 3 and s["fields_for_5pp"] > s["fields"]]
            if need:
                note += " For an interval of +/- 5 points, from the spread seen here: " + "; ".join(need) + "."
            flagged = sum(s["warnings"] for s in summ)
            if flagged:
                note += f" {flagged} image{'s' if flagged > 1 else ''} carry a warning (see the image pages)."
            el.append(Paragraph(note, SMALL))
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
            el.extend(_morphology_summary(results))
    elif kind == "morphology":
        el.extend(_morphology_summary(results, single=True))
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
