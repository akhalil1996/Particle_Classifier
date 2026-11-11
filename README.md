# CauliQuant

Desktop software that measures porous **cauliflower** agglomerates and particle shape in titanium powder from SEM micrographs.

| Tab | What it does |
|---|---|
| **Home** | What the software does, version, image requirements, which classification model is loaded |
| **Particle classification** | Segments every particle above 10 µm, gives it a cauliflower probability (U-Net reading of the surface combined with outline shape and surface relief), and reports the cauliflower share of particle area (and volume) inside a counting frame, with borderline particles, quality warnings and, per sample, a 95 % interval over fields |
| **Morphology profiling** | Assigns every particle to one of five shape classes (C1 compact … C5 branched / agglomerate) with the k-means model fitted on Microtrac SYNC data, on SEM images and on Microtrac exports (particle report PDF or page pictures, particle table .xls, size report PDF) |

Each analysis tab takes **one image** or **a whole folder** (batch). The viewer switches between the original and the analysed image, the results sit on the right, and **Export PDF report** / **Export CSV** save the results. In batch mode fields are pooled into samples by file name: everything before `_x` is the sample (`384_x100_0031` → `384`).

## Installing (users)

Download `CauliQuant-Setup-<version>.exe` from the repository's **Releases** page and run it. No Python or scripts needed. Windows may show a SmartScreen warning because the installer is not code-signed: click *More info → Run anyway*.

## Images

Secondary-electron images at x100, ideally 2560 × 1920 px (0.496 µm/px), particles spread as a monolayer. The pixel size is read from the Hitachi SU3500 metadata (`.txt` next to the image, or TIFF tag 40092); otherwise enter it in the Settings bar. Images at another pixel size are resampled to 0.496 µm/px before analysis. The data bar is detected and left out.

## Microtrac files (morphology tab)

Open them with **Open files…** or **Open folder…** next to, or instead of, SEM images. Each file is recognised by its content:

| File | What the tab reads | Shares |
|---|---|---|
| Particle report, `…(All particles).pdf` (SYNC viewer, "Particles") | every printed silhouette, measured again and given the C1-C5 class | by number, of the particles shown |
| Pictures of report pages (`…ViewerReport_page1.jpg`, PNG) | same, from the page picture | by number, of the particles shown |
| Particle table, `…(All particles).xls` | Microtrac's own descriptors for every particle | by number, by volume, and by volume in the 30-100 µm band |
| Size report, `<batch>.pdf` ("Particle Size Analysis") | D10, D50, D90, span, MV / MA / MN | none (no shapes) |

The report prints each particle scaled to the same small cell, with no size, and only the first particles of the viewer list (about 1,300 of several thousand, the widest when sorted by FWidth), so its shares are by number of the particles shown and do not compare with shares by volume. The particle table is the complete record and should be preferred when it is available. A silhouette's outline is a little longer than Microtrac's own, so a linear correction per descriptor (`models/microtrac_gallery.json`, refit with `backend/fit_gallery_calibration.py --pair report.pdf table.xls …`) puts the silhouettes on the Microtrac scale: fitted on three reports with their tables, 93 % of particles of a report left out of the fit got the same class as from the table, and class shares differed by at most 1.7 points. Pictures should be page exports or straight scans; a phone photo at an angle distorts the shapes. Microtrac exports are named after the SOP and a time stamp, so their batch is taken from the folder they sit in (`382/H0131…pdf` → `382`).

## Reading a result

- **The headline is a probability-weighted area share.** Every counted particle adds its area times its cauliflower probability. Counting only the particles called cauliflower is also reported; it reads lower, because a classifier misses more of a minority class than it over-calls.
- **Borderline particles** (probability 0.35 to 0.65) are drawn in a lighter shade and their share of the area is reported. About a third of particles sit between the two classes, and that is where most of the remaining error is.
- **One image is a rough number.** Fields of the same stub differ by about 10 points. In batch mode each sample gets a 95 % interval over its fields (three fields or more) and an estimate of how many fields a ±5 point interval would take.
- **Warnings** flag images whose SEM settings differ from the ones the model was validated on (x100, 15 kV, spot intensity 50, slow scan 64, SE detector), crowded, soft or saturated fields, and fields with few particles.
- **Measured error.** Every report prints the model's error on batches it never saw during fitting. For the bundled model: one field within about 8 points of the reference on average, a pooled batch within about 6 points.
- **Morphology shares are relative.** C1 to C5 on SEM outlines rank samples against each other; they are not on the Microtrac scale and move by several points with how strongly touching particles are split.

## Repository layout

```
run_app.py              start the app (python run_app.py)
gui/                    the window (PySide6)
cauliquant/             analysis package: segmentation, features, counting frame, U-Net, shape classes, reports
models/                 cauliflower_unet.onnx (network), cauliflower_unet.json (its model card: particle
                        classifier, validated SEM settings, measured error), shape_classes_k5.json (C1-C5 model),
                        microtrac_gallery.json (silhouette correction for Microtrac particle reports)
backend/                command-line tools, not part of the installer:
  classify_folder.py      batch classification folder A -> folder B (masks, overlays, summary.csv)
  train.py                train the U-Net on labelled A/B pairs, with the training monitor
  export_onnx.py          convert a trained model.pt into the .onnx the app loads
  fit_classifier.py       fit the particle classifier on labelled fields, check it on samples it has not
                          seen, and write the model card
  morphology_profile.py   Microtrac SYNC exports and SEM results per batch, and their correlation
  fit_gallery_calibration.py  refit the silhouette correction from report PDFs paired with their .xls tables
  import_labels.py        bring labelling-bench labels into folder B
packaging/              PyInstaller spec and Inno Setup installer script
.github/workflows/      builds the Windows installer on GitHub
tests/smoke_test.py     end-to-end check on a synthetic image
```

## Running from source (developers)

```
pip install -r requirements.txt
python run_app.py
```

The back-end tools also need `pip install -r requirements-backend.txt` (adds PyTorch, pandas and scikit-learn).

## Updating the classification model

The model has two parts: the U-Net (`.onnx`) and its model card (`.json` with the same name), which holds the particle classifier, the SEM settings it was validated on and its measured error.

**New labelled fields, same network** (the usual case):

1. `python backend/fit_classifier.py --raw <images> --labels <label folder> --model models/cauliflower_unet.onnx`
   It scores every sample with a classifier fitted on the other samples, prints the error, and rewrites the card.
2. Commit the new `.json`, bump `__version__`, push.

**Retrained network:**

1. Train: `python backend/train.py --a A_raw --b B_marked` → `runs/<time>/model.pt`
2. Convert: `python backend/export_onnx.py runs/<time>/model.pt models/cauliflower_unet.onnx`
3. Refit the classifier and card with `fit_classifier.py` as above (the classifier depends on the network's scores).
4. Either pick the new `.onnx` in the app (Home → *Choose model file…*; keep the `.json` next to it) or commit both so the next installer ships them.

A network file without a card still runs, on the raw network score, and every result carries a warning that it is not validated.

## Building the installer

GitHub builds it: every push to `main` builds the installer and publishes it on the Release named after the version in `cauliquant/__init__.py` (for example `v0.2.0`). To ship a new version, bump `__version__` and push; pushing again without a bump refreshes the installer on the existing release.

To build on your own Windows PC instead: `pip install -r requirements.txt pyinstaller`, then `pyinstaller packaging/cauliquant.spec --noconfirm`, then compile `packaging/installer.iss` with Inno Setup 6. The installer appears in `installer/`.
