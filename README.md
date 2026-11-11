# CauliQuant

Desktop software that measures porous **cauliflower** agglomerates and particle shape in titanium powder from SEM micrographs.

| Tab | What it does |
|---|---|
| **Home** | What the software does, version, image requirements, which classification model is loaded |
| **Particle classification** | Segments every particle above 10 µm, scores it with a U-Net as cauliflower or dense, and reports the cauliflower share of particle area (and volume) inside a counting frame |
| **Morphology profiling** | Assigns every particle to one of five shape classes (C1 compact … C5 branched / agglomerate) with the k-means model fitted on Microtrac SYNC data |

Each analysis tab takes **one image** or **a whole folder** (batch). The viewer switches between the original and the analysed image, the results sit on the right, and **Export PDF report** / **Export CSV** save the results. In batch mode fields are pooled into samples by file name: everything before `_x` is the sample (`384_x100_0031` → `384`).

## Installing (users)

Download `CauliQuant-Setup-<version>.exe` from the repository's **Releases** page and run it. No Python or scripts needed. Windows may show a SmartScreen warning because the installer is not code-signed: click *More info → Run anyway*.

## Images

Secondary-electron images at x100, ideally 2560 × 1920 px (0.496 µm/px), particles spread as a monolayer. The pixel size is read from the Hitachi SU3500 metadata (`.txt` next to the image, or TIFF tag 40092); otherwise enter it in the Settings bar. Images at another pixel size are resampled to 0.496 µm/px before analysis. The data bar is detected and left out.

## Repository layout

```
run_app.py              start the app (python run_app.py)
gui/                    the window (PySide6)
cauliquant/             analysis package: segmentation, features, counting frame, U-Net, shape classes, reports
models/                 cauliflower_unet.onnx (classifier), shape_classes_k5.json (C1-C5 model)
backend/                command-line tools, not part of the installer:
  classify_folder.py      batch classification folder A -> folder B (masks, overlays, summary.csv)
  train.py                train the U-Net on labelled A/B pairs, with the training monitor
  export_onnx.py          convert a trained model.pt into the .onnx the app loads
  morphology_profile.py   Microtrac SYNC exports and SEM results per batch, and their correlation
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

The back-end tools also need `pip install -r requirements-backend.txt` (adds PyTorch and pandas).

## Updating the classification model

1. Train: `python backend/train.py --a A_raw --b B_marked` → `runs/<time>/model.pt`
2. Convert: `python backend/export_onnx.py runs/<time>/model.pt models/cauliflower_unet.onnx`
3. Either pick the new `.onnx` in the app (Home → *Choose model file…*) or commit it so the next installer ships it.

## Building the installer

GitHub builds it: every push to `main` produces the installer as a workflow artifact (Actions tab), and pushing a version tag publishes a Release:

```
# bump __version__ in cauliquant/__init__.py first
git tag v0.1.0
git push origin v0.1.0
```

To build on your own Windows PC instead: `pip install -r requirements.txt pyinstaller`, then `pyinstaller packaging/cauliquant.spec --noconfirm`, then compile `packaging/installer.iss` with Inno Setup 6. The installer appears in `installer/`.
