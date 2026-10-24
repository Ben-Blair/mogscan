"""Frontalize every SCUT face and compute every feature the model uses.

Features (see mogscan_metrics.py): geometric / depth / symmetry metrics, whole-face shape
features (averageness, dimorphism, 12 shape PCs), pixel features (skin, colouring, hair,
leanness) and the 51 MediaPipe expression blendshapes.

Outputs:
  out/features.csv                         name, score, sex, race, split, yaw, pitch, all features
  out/feature_groups.json                  group -> feature names (order used everywhere)
  ../app/public/models/face_template.json  478x3 mean frontal shape used live in the browser
  ../app/public/models/shape.json          dimorphism axis + shape PCA basis
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from extract_landmarks import DATA, load_labels
from mogscan_metrics import (APPEARANCE_GROUPS, METRIC_GROUPS, SHAPE_NAMES, appearance_crop, build_shape_model,
                             build_template, compute_appearance, compute_metrics, compute_shape, frontalize, to_3d)

ROOT = Path(__file__).parent
OUT = ROOT / "out"
APP_MODELS = ROOT.parent / "app" / "public" / "models"


def feature_groups(blend_names) -> dict:
    return {**METRIC_GROUPS, "shape": SHAPE_NAMES, **APPEARANCE_GROUPS,
            "expression": [f"bs_{b}" for b in blend_names if b != "_neutral"]}


def main():
    d = np.load(OUT / "landmarks.npz")
    labels = load_labels().set_index("name")
    names = list(d["names"])
    shapes = [to_3d(lm, w, h) for lm, (w, h) in zip(d["landmarks"], d["sizes"])]
    yaw, pitch = d["poses"][:, 0], d["poses"][:, 1]
    blend_names = [str(b) for b in d["blend_names"]]

    # Template and shape space from *training* faces only, so test data never leaks in.
    is_train = np.array([labels.loc[n, "split"] == "train" for n in names])
    frontal = is_train & (np.abs(yaw) < 3) & (np.abs(pitch - np.median(pitch)) < 4)
    template = build_template([s for s, f in zip(shapes, frontal) if f])
    print(f"template built from {frontal.sum()} frontal training faces")
    fronts = [frontalize(s, template) for s in shapes]
    is_male = np.array([labels.loc[n, "sex"] == "M" for n in names])
    shape = build_shape_model([f for f, t in zip(fronts, is_train) if t], is_male[is_train], template)

    groups = feature_groups(blend_names)
    rows = []
    for i, n in enumerate(names):
        w, h = d["sizes"][i]
        pts_px = d["landmarks"][i][:, :2].astype(np.float64) * [w, h]
        crop, crop_pts = appearance_crop(Image.open(DATA / "Images" / n), pts_px, w, h)
        rows.append({"name": n, **labels.loc[n].to_dict(), "yaw": yaw[i], "pitch": pitch[i],
                     **compute_metrics(fronts[i]), **compute_shape(fronts[i], template, shape),
                     **compute_appearance(crop, crop_pts),
                     **{f"bs_{b}": float(v) for b, v in zip(blend_names, d["blendshapes"][i]) if b != "_neutral"}})
    all_names = [f for g in groups.values() for f in g]
    df = pd.DataFrame(rows)[["name", "score", "sex", "race", "split", "yaw", "pitch", *all_names]]
    df.to_csv(OUT / "features.csv", index=False)
    json.dump(groups, open(OUT / "feature_groups.json", "w"), indent=1)
    with open(APP_MODELS / "face_template.json", "w") as f:
        json.dump(np.round(template, 6).tolist(), f, separators=(",", ":"))
    with open(APP_MODELS / "shape.json", "w") as f:
        json.dump({k: np.round(np.asarray(v), 7).tolist() for k, v in shape.items()}, f, separators=(",", ":"))
    print(f"{len(all_names)} features: " + ", ".join(f"{g}={len(v)}" for g, v in groups.items()))
    print("corr(|yaw|, symmetry) =", round(np.corrcoef(np.abs(df.yaw), df.symmetry)[0, 1], 3))


if __name__ == "__main__":
    main()
