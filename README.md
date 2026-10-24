# MogScan

Real-time, multi-person, explainable facial-aesthetics ("PSL") scoring from any USB webcam.
Everything runs in the browser, on-device.

Final project (Startup/Product track) by Ben Blair and Cyril Sokolov.

```
USB cam → MediaPipe FaceLandmarker (≤6 faces, 478 landmarks, head pose)
        → IoU tracker + pose gate (apparent |yaw| < 14°) + EMA smoothing
        → Procrustes frontalization → 143 features: 58 geometry/3D-depth/symmetry (canthal tilt, FWHR,
          orbital vector…), 14 whole-face shape (averageness, dimorphism, PCs), 20 pixel features
          (skin, colouring, hair, leanness), 51 expression blendshapes
        → sign-constrained ridge model (every weight shown) ┐
        → MobileNetV3 CNN via onnxruntime-web (score+sex)   ┴→ fused score → PSL 1–8 → tier
                                                              + Harmony / Dimorphism / Angularity / Softmax pillars
```

## Repo layout
| Path | What |
|---|---|
| `app/` | Vite + React + TypeScript web app (the product) |
| `training/` | Python pipeline: landmark extraction, metrics, geometric + CNN models, ONNX export, evaluation |
| `study/` | User-study protocol, consent script, survey, analysis scripts |
| `writeup/` | LaTeX source for the written deliverable (`application.pdf`) |
| `slides/` | Demo-day slide deck (`slides.pdf`) |
| `video/` | Demo video script, shot list, ffmpeg assembly script |

## Run the app
```bash
cd app
npm install        # also copies the MediaPipe wasm into public/ (works offline)
npm run dev        # open http://localhost:5173, then pick your USB camera from the dropdown
npm test           # TypeScript ↔ Python feature parity tests (geometry, shape, pixels)
npm run build      # static build in app/dist (deployable anywhere, e.g. Vercel)
```
The app has two inputs: the live camera (pick a USB camera from the dropdown) or **Upload photo**.
There are no battles, rankings or leaderboards; each face is scanned for ~2 s, then the score locks.
The card shows a glow-up plan (non-surgical actions, each with a predicted score) and your potential score.
Add `?nocam` to the URL to skip the webcam, and `?study` to show the **Record** button used for the user study.
Set `VITE_FEEDBACK_URL` (for example in `app/.env.local`) to show a Feedback button that links to your Google Form.

## Reproduce the models
```bash
cd training
python3 -m venv .venv && .venv/bin/pip install torch torchvision mediapipe scikit-learn onnx onnxruntime onnxscript gdown pandas matplotlib scipy pytest
mkdir -p data && cd data && ../.venv/bin/gdown 1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf && unzip SCUT-FBP5500_v2.1.zip && cd ..
.venv/bin/python extract_landmarks.py   # MediaPipe on 5,500 faces (landmarks + blendshapes)
.venv/bin/python build_features.py      # frontal template, shape model, all 143 features
.venv/bin/python train_geometric.py     # ridge models (free + sign-constrained) + per-group ablations
.venv/bin/python train_cnn.py           # MobileNetV3 fine-tune (~35 min on Apple-silicon MPS)
.venv/bin/python fuse_calibrate.py      # fusion weights, PSL calibration, headline results
.venv/bin/python export_onnx.py         # app/public/models/mognet.onnx
.venv/bin/python eval_bias.py           # subgroup accuracy and bias
.venv/bin/python make_figures.py        # figures + LaTeX number macros for the write-up
.venv/bin/python make_parity_fixture.py # fixtures for the TS parity test
```
All reported numbers come from `training/out/results*.json`. The write-up pulls them in automatically from
`writeup/generated/`.

Dataset: [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) (Liang et al., ICPR 2018).
It is licensed for non-commercial research. Its images are not redistributed here.

## Build the documents
```bash
cd writeup && latexmk -pdf application.tex
cd slides && latexmk -pdf slides.tex
```
