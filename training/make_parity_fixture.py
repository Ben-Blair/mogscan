"""Dump a few SCUT faces (raw landmarks + Python-computed features) for the TS parity test.

Appearance features are checked on a *synthetic* crop (smooth colour gradients plus seeded noise), not on
the dataset photo: SCUT-FBP5500 images must not be redistributed, and parity only needs both sides to see
identical pixels."""
import base64
import json
from pathlib import Path

import numpy as np
from mogscan_metrics import (APPEAR_SIZE, apparent_pose, compute_appearance, crop_box, compute_metrics, compute_shape,
                             frontalize, to_3d)

ROOT = Path(__file__).parent
MODELS = ROOT.parent / "app" / "public" / "models"
FIXTURE = ROOT.parent / "app" / "src" / "scoring" / "__fixtures__" / "parity.json"


def synthetic_crop(seed: int) -> np.ndarray:
    """Face-like but non-identifying test image: colour gradients, a darker band, seeded noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:APPEAR_SIZE, 0:APPEAR_SIZE] / APPEAR_SIZE
    img = np.stack([180 + 50 * xx, 140 + 40 * yy, 120 + 30 * xx * yy], axis=2)
    img[int(APPEAR_SIZE * 0.35):int(APPEAR_SIZE * 0.45)] *= 0.5
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    d = np.load(ROOT / "out" / "landmarks.npz")
    # Use the rounded template / shape model exactly as shipped to the browser.
    template = np.array(json.load(open(MODELS / "face_template.json")))
    shape = json.load(open(MODELS / "shape.json"))
    faces = []
    # A spread of faces including strongly turned ones (index chosen by |yaw|).
    order = np.argsort(np.abs(d["poses"][:, 0]))
    for i in [int(order[k]) for k in (0, 1000, 3000, 5000, -1)]:
        lm, (w, h) = d["landmarks"][i].astype(np.float64), d["sizes"][i]
        f3 = frontalize(to_3d(lm, w, h), template)
        x, y, side = crop_box(lm[:, :2] * [w, h], w, h)
        x, y, side = round(x), round(y), round(side)
        crop_pts = (lm[:, :2] * [w, h] - [x, y]) * (APPEAR_SIZE / side)
        crop = synthetic_crop(i)
        rgba = np.concatenate([crop, np.full(crop.shape[:2] + (1,), 255, np.uint8)], axis=2)
        faces.append({"name": str(d["names"][i]), "w": int(w), "h": int(h),
                      "landmarks": [[float(x), float(y), float(z)] for x, y, z in lm],
                      "metrics": {**compute_metrics(f3), **compute_shape(f3, template, shape)},
                      "pose": apparent_pose(to_3d(lm, w, h), template),
                      "crop": base64.b64encode(rgba.tobytes()).decode(), "cropSize": int(crop.shape[0]),
                      "appearance": compute_appearance(crop, crop_pts)})
    json.dump(faces, open(FIXTURE, "w"))
    print(f"wrote {len(faces)} faces to {FIXTURE}")


if __name__ == "__main__":
    main()
