"""Dump a few SCUT faces (raw landmarks + Python-computed metrics) for the TS parity test."""
import json
from pathlib import Path

import numpy as np

from mogscan_metrics import apparent_pose, compute_metrics, frontalize, to_3d

ROOT = Path(__file__).parent
FIXTURE = ROOT.parent / "app" / "src" / "scoring" / "__fixtures__" / "parity.json"


def main():
    d = np.load(ROOT / "out" / "landmarks.npz")
    template = np.array(json.load(open(ROOT.parent / "app" / "public" / "models" / "face_template.json")))
    faces = []
    # A spread of faces including strongly turned ones (index chosen by |yaw|).
    order = np.argsort(np.abs(d["poses"][:, 0]))
    for i in [int(order[k]) for k in (0, 1000, 3000, 5000, -1)]:
        lm, (w, h) = d["landmarks"][i].astype(np.float64), d["sizes"][i]
        m = compute_metrics(frontalize(to_3d(lm, w, h), template)[:, :2])
        faces.append({"name": str(d["names"][i]), "w": int(w), "h": int(h),
                      "landmarks": [[float(x), float(y), float(z)] for x, y, z in lm], "metrics": m,
                      "pose": apparent_pose(to_3d(lm, w, h), template)})
    json.dump(faces, open(FIXTURE, "w"))
    print(f"wrote {len(faces)} faces to {FIXTURE}")


if __name__ == "__main__":
    main()
