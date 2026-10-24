"""Does following the glow-up plan actually raise the MogScan score? Measure it.

For held-out *test* faces, each cosmetic action is simulated as an edit of the photo (contour shadow
under the jaw, darker brows, smoother skin, ...). The edited photo then goes through the complete app
pipeline: MediaPipe landmarks + blendshapes -> every feature -> feature model, CNN on the crop -> fused
score -> PSL. The change against the unedited photo is the action's real effect on the score.

Outputs:
  ../app/public/models/glowup.json   per action and sex: mean feature change, CNN effect as a function of
                                     the person's main measurement, success rate (the app's predictions)
  out/results_glowup.json            the same, plus summary numbers for the write-up
"""
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from PIL import Image
from torchvision import transforms

from extract_landmarks import DATA, MODEL, load_labels
from mogscan_metrics import (appearance_crop, compute_appearance, compute_metrics, compute_shape, crop_box,
                             frontalize, to_3d)
from train_cnn import DEVICE, MEAN, SIZE, STD, MogNet

ROOT = Path(__file__).parent
OUT = ROOT / "out"
APP_MODELS = ROOT.parent / "app" / "public" / "models"
N_PER_SEX = 150
MIN_SUCCESS = 0.6   # an action is offered only if it raised the score for at least this share of faces

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
BROW_R = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_L = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LID_R = [33, 246, 161, 160, 159, 158, 157, 173, 133]
LID_L = [263, 466, 388, 387, 386, 385, 384, 398, 362]
LOWER_R = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LOWER_L = [263, 249, 390, 373, 374, 380, 381, 382, 362]
JAW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]

# Action id -> the feature its prediction is personalised on (must match app/src/scoring/glowup.ts).
MAIN_FEATURE = {"skincare": "skin_texture", "redness": "skin_redness", "undereye": "under_eye", "glow": "skin_yellow",
                "jaw": "jaw_definition", "cheeks": "cheek_hollow", "brows": "brow_contrast", "lashes": "lash_line",
                "lips": "lip_contrast", "facialhair": "facial_hair", "eyes": "sclera_bright"}


# ---- masks and edits (img: float32 HxWx3 RGB in 0..255, p: (478, 2) pixel landmarks) ----------------------

def poly(shape, p, idx, blur=0.0):
    m = np.zeros(shape[:2], np.float32)
    cv2.fillPoly(m, [np.round(p[idx]).astype(np.int32)], 1.0)
    return cv2.GaussianBlur(m, (0, 0), blur) if blur > 0 else m


def line(shape, pts, thick, blur):
    m = np.zeros(shape[:2], np.float32)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1.0, max(1, int(round(thick))), cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), blur)


def skin_mask(img, p, ipd):
    m = poly(img.shape, p, FACE_OVAL)
    for idx in (EYE_R, EYE_L, BROW_R, BROW_L, LIPS_OUT):
        m -= cv2.dilate(poly(img.shape, p, idx), np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 0.03 * ipd)


def apply(img, m, edited):
    return img + m[..., None] * (edited - img)


def e_skincare(img, p, ipd, sex):
    m = skin_mask(img, p, ipd)
    smooth = cv2.bilateralFilter(img, 0, 18, 0.05 * ipd)
    out = apply(img, m, 0.25 * img + 0.75 * smooth)
    big = cv2.GaussianBlur(out, (0, 0), 0.25 * ipd)
    mean = (big * m[..., None]).sum((0, 1)) / (m.sum() + 1e-6)
    return apply(out, m * 0.35, out + (mean - big))


def e_redness(img, p, ipd, sex):
    m = skin_mask(img, p, ipd)
    r, g = img[..., 0], img[..., 1]
    out = img.copy()
    out[..., 0] = r - 0.2 * np.maximum(r - g - 25, 0)
    return apply(img, m, out)


def e_undereye(img, p, ipd, sex):
    m = np.zeros(img.shape[:2], np.float32)
    for idx in (LOWER_R, LOWER_L):
        lo = p[idx]
        m = np.maximum(m, poly(img.shape, np.vstack([lo, lo[::-1] + [0, 0.22 * ipd]]), list(range(2 * len(idx)))))
    m = cv2.GaussianBlur(m, (0, 0), 0.04 * ipd)
    return apply(img, m, img * 1.12)


def e_glow(img, p, ipd, sex):
    return apply(img, skin_mask(img, p, ipd), img * np.array([1.04, 1.01, 0.92], np.float32))


def e_jaw(img, p, ipd, sex):
    c = p.mean(axis=0)
    jaw = p[JAW]
    u = (jaw - c) / (np.linalg.norm(jaw - c, axis=1, keepdims=True) + 1e-6)
    m = line(img.shape, jaw + 0.05 * ipd * u, 0.08 * ipd, 0.035 * ipd)
    return apply(img, m, img * 0.75)


def e_cheeks(img, p, ipd, sex):
    m = np.zeros(img.shape[:2], np.float32)
    for ear, mouth in ((234, 61), (454, 291)):
        a = p[ear] + [0, 0.05 * ipd]
        m = np.maximum(m, line(img.shape, np.array([a, a + 0.55 * (p[mouth] - p[ear])]), 0.1 * ipd, 0.045 * ipd))
    return apply(img, m, img * 0.85)


def e_brows(img, p, ipd, sex):
    m = np.maximum(poly(img.shape, p, BROW_R), poly(img.shape, p, BROW_L))
    k = max(1, int(round(0.02 * ipd)))
    m = cv2.GaussianBlur(cv2.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8)), (0, 0), 0.012 * ipd)
    return apply(img, m, img * 0.7)


def e_lashes(img, p, ipd, sex):
    m = np.maximum(line(img.shape, p[LID_R], 0.025 * ipd, 0.006 * ipd), line(img.shape, p[LID_L], 0.025 * ipd, 0.006 * ipd))
    return apply(img, m, img * 0.45)


def e_lips(img, p, ipd, sex):
    m = np.clip(poly(img.shape, p, LIPS_OUT) - poly(img.shape, p, LIPS_IN), 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 0.01 * ipd)
    return apply(img, m * 0.3, np.broadcast_to(np.array([165, 55, 75], np.float32), img.shape))


def e_facialhair(img, p, ipd, sex):
    m = poly(img.shape, p, FACE_OVAL)
    m[: int(p[2][1] + 0.05 * ipd)] = 0
    m -= cv2.dilate(poly(img.shape, p, LIPS_OUT), np.ones((7, 7), np.uint8))
    m = cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 0.04 * ipd)
    grain = cv2.GaussianBlur(np.random.default_rng(0).random(img.shape[:2]).astype(np.float32), (0, 0), 0.6)
    grain = (grain - grain.min()) / (grain.max() - grain.min() + 1e-6)
    return img * (1 - 0.2 * m * grain ** 1.5 * 2)[..., None]


def e_eyes(img, p, ipd, sex):
    m = np.maximum(poly(img.shape, p, EYE_R), poly(img.shape, p, EYE_L))
    for c, ring in ((468, [469, 470, 471, 472]), (473, [474, 475, 476, 477])):
        r = 1.1 * np.mean([np.linalg.norm(p[i] - p[c]) for i in ring])
        cv2.circle(m, tuple(np.round(p[c]).astype(int)), int(round(r)), 0.0, -1)
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    m = m * (lum > 90)
    return apply(img, cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.5), img + 0.25 * (255 - img))


EDITS = {"skincare": e_skincare, "redness": e_redness, "undereye": e_undereye, "glow": e_glow, "jaw": e_jaw,
         "cheeks": e_cheeks, "brows": e_brows, "lashes": e_lashes, "lips": e_lips, "facialhair": e_facialhair,
         "eyes": e_eyes}
MALE_ONLY = {"facialhair"}


# ---- full app pipeline -------------------------------------------------------------------------------------

class Scorer:
    def __init__(self):
        self.geo = json.load(open(APP_MODELS / "geometric.json"))
        self.fusion = json.load(open(APP_MODELS / "fusion.json"))
        self.template = np.array(json.load(open(APP_MODELS / "face_template.json")))
        self.shape = json.load(open(APP_MODELS / "shape.json"))
        self.cnn = MogNet().to(DEVICE)
        self.cnn.load_state_dict(torch.load(OUT / "cnn.pt", map_location=DEVICE))
        self.cnn.eval()
        self.tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        opts = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=MODEL.read_bytes()),
                                     running_mode=RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True)
        self.lmk = FaceLandmarker.create_from_options(opts)

    def features(self, img: Image.Image):
        rgb = np.asarray(img)
        res = self.lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None, None
        w, h = img.size
        lm = np.array([[l.x, l.y, l.z] for l in res.face_landmarks[0]])
        f3 = frontalize(to_3d(lm, w, h), self.template)
        px = lm[:, :2] * [w, h]
        crop, crop_pts = appearance_crop(img, px, w, h)
        feats = {**compute_metrics(f3), **compute_shape(f3, self.template, self.shape),
                 **compute_appearance(crop, crop_pts),
                 **{f"bs_{c.category_name}": c.score for c in res.face_blendshapes[0] if c.category_name != "_neutral"}}
        return feats, px

    def geo_score(self, feats, sex):
        s = self.geo["sexes"][sex]
        v = np.array([feats[m] for m in self.geo["metrics"]])
        z = np.clip((v - np.array(s["mean"])) / np.array(s["std"]), -4, 4)
        return s["intercept"] + float(np.sum(np.array(s["lin"]) * z + np.array(s["quad"]) * (z ** 2 - 1)))

    @torch.no_grad()
    def cnn_score(self, img: Image.Image, px):
        x, y, side = crop_box(px, *img.size)
        box = tuple(int(round(v)) for v in (x, y, x + side, y + side))
        t = self.tf(img.crop(box)).unsqueeze(0).to(DEVICE)
        return float(self.cnn(t)[0].cpu())

    def score(self, img: Image.Image, sex):
        feats, px = self.features(img)
        if feats is None:
            return None
        g, c = self.geo_score(feats, sex), self.cnn_score(img, px)
        f = self.fusion
        fused = f["wCnn"] * c + f["wGeo"] * g + f["bias"]
        return {"feats": feats, "px": px, "geo": g, "cnn": c, "psl": 4 + (fused - f["scoreMean"]) / f["scoreStd"]}


def main():
    labels = load_labels()
    test = labels[labels.split == "test"]
    faces = pd.concat([test[test.sex == s].sample(N_PER_SEX, random_state=0) for s in ("M", "F")])
    scorer = Scorer()
    names = scorer.geo["metrics"]
    rows = []
    for k, r in enumerate(faces.itertuples()):
        img = Image.open(DATA / "Images" / r.name).convert("RGB")
        base = scorer.score(img, r.sex)
        if base is None:
            continue
        p = base["px"]
        ipd = float(np.linalg.norm(p[468] - p[473]))
        arr = np.asarray(img, np.float32)
        combined = arr.copy()
        variants = {}
        for a, edit in EDITS.items():
            if a in MALE_ONLY and r.sex != "M":
                continue
            variants[a] = edit(arr, p, ipd, r.sex)
            combined = edit(combined, p, ipd, r.sex)
        variants["combined"] = combined
        for a, out in variants.items():
            res = scorer.score(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), r.sex)
            if res is None:
                continue
            row = {"name": r.name, "sex": r.sex, "action": a, "dpsl": res["psl"] - base["psl"],
                   "dcnn": res["cnn"] - base["cnn"], "dgeo": res["geo"] - base["geo"]}
            if a in MAIN_FEATURE:
                main_f = MAIN_FEATURE[a]
                s = scorer.geo["sexes"][r.sex]
                i = names.index(main_f)
                row["z_main"] = (base["feats"][main_f] - s["mean"][i]) / s["std"][i]
            row.update({f"d_{m}": res["feats"][m] - base["feats"][m] for m in names})
            rows.append(row)
        if k % 50 == 0:
            print(f"{k}/{len(faces)}", flush=True)
    df = pd.DataFrame(rows)

    app, summary = {"minSuccess": MIN_SUCCESS, "actions": {}}, {}
    for a in [*EDITS, "combined"]:
        for sex in ("M", "F"):
            d = df[(df.action == a) & (df.sex == sex)]
            if d.empty:
                continue
            stats = {"n": int(len(d)), "meanPsl": float(d.dpsl.mean()), "success": float((d.dpsl > 0).mean()),
                     "meanCnn": float(d.dcnn.mean()), "meanGeo": float(d.dgeo.mean())}
            summary.setdefault(a, {})[sex] = stats
            if a == "combined":
                continue
            # CNN effect personalised on the main measurement: dcnn ~ a + b * z_main (least squares).
            X = np.stack([np.ones(len(d)), d.z_main.values], axis=1)
            coef, *_ = np.linalg.lstsq(X, d.dcnn.values, rcond=None)
            fd = {m: float(d[f"d_{m}"].mean()) for m in names if not m.startswith("bs_")}
            app["actions"].setdefault(a, {})[sex] = {
                **stats, "cnnA": float(coef[0]), "cnnB": float(coef[1]),
                "featureDelta": {m: v for m, v in fd.items() if abs(v) > 1e-9},
            }
    for sex in ("M", "F"):
        singles = sum(summary[a][sex]["meanPsl"] for a in EDITS if sex in summary[a])
        summary["combined"][sex]["additivity"] = summary["combined"][sex]["meanPsl"] / singles if singles else None
        app.setdefault("additivity", {})[sex] = summary["combined"][sex]["additivity"]
    json.dump(app, open(APP_MODELS / "glowup.json", "w"))
    json.dump(summary, open(OUT / "results_glowup.json", "w"), indent=2)
    print(f"{'action':12s} {'sex':3s} {'mean PSL':>9s} {'raised':>7s} {'CNN part':>9s}")
    for a, by in summary.items():
        for sex, s in by.items():
            print(f"{a:12s} {sex:3s} {s['meanPsl']:+9.3f} {s['success'] * 100:6.0f}% {s['meanCnn']:+9.3f}")


if __name__ == "__main__":
    main()
