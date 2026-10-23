"""Frontalize every SCUT face, compute geometry metrics, and export the frontal template.

Outputs:
  out/features.csv                         name, score, sex, race, split, yaw, pitch, metrics
  ../app/public/models/face_template.json  478x3 mean frontal shape used live in the browser
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extract_landmarks import load_labels
from mogscan_metrics import METRIC_NAMES, build_template, compute_metrics, frontalize, to_3d

ROOT = Path(__file__).parent
OUT = ROOT / "out"
APP_MODELS = ROOT.parent / "app" / "public" / "models"


def main():
    d = np.load(OUT / "landmarks.npz")
    labels = load_labels().set_index("name")
    names = list(d["names"])
    shapes = [to_3d(lm, w, h) for lm, (w, h) in zip(d["landmarks"], d["sizes"])]
    yaw, pitch = d["poses"][:, 0], d["poses"][:, 1]

    # Template from near-frontal *training* faces only, so test data never leaks in.
    is_train = np.array([labels.loc[n, "split"] == "train" for n in names])
    frontal = is_train & (np.abs(yaw) < 3) & (np.abs(pitch - np.median(pitch)) < 4)
    template = build_template([s for s, f in zip(shapes, frontal) if f])
    print(f"template built from {frontal.sum()} frontal training faces")

    rows = []
    for n, s, y, p in zip(names, shapes, yaw, pitch):
        m = compute_metrics(frontalize(s, template)[:, :2])
        rows.append({"name": n, **labels.loc[n].to_dict(), "yaw": y, "pitch": p, **m})
    df = pd.DataFrame(rows)[["name", "score", "sex", "race", "split", "yaw", "pitch", *METRIC_NAMES]]
    df.to_csv(OUT / "features.csv", index=False)
    with open(APP_MODELS / "face_template.json", "w") as f:
        json.dump(np.round(template, 6).tolist(), f, separators=(",", ":"))
    print(df[METRIC_NAMES].describe().T[["mean", "std"]].round(3))
    print("corr(|yaw|, symmetry) =", round(np.corrcoef(np.abs(df.yaw), df.symmetry)[0, 1], 3))


if __name__ == "__main__":
    main()
