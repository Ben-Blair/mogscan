"""Run MediaPipe FaceLandmarker over SCUT-FBP5500.

Output: training/out/landmarks.npz with names, (N,478,3) normalised landmarks, image sizes,
head pose (yaw, pitch) and the 52 ARKit-style expression blendshapes. Metrics are computed
afterwards by build_features.py.
"""
import os
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from mogscan_metrics import head_pose

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "SCUT-FBP5500_v2"
OUT = ROOT / "out"
MODEL = ROOT.parent / "app" / "public" / "models" / "face_landmarker.task"


def load_labels() -> pd.DataFrame:
    split_dir = DATA / "train_test_files" / "split_of_60%training and 40%testing"
    rows = []
    for split in ("train", "test"):
        for line in open(split_dir / f"{split}.txt"):
            name, score = line.split()
            rows.append((name, float(score), split))
    df = pd.DataFrame(rows, columns=["name", "score", "split"])
    df["race"] = df.name.str[0].map({"A": "Asian", "C": "Caucasian"})
    df["sex"] = df.name.str[1].map({"F": "F", "M": "M"})
    return df


def main():
    OUT.mkdir(exist_ok=True)
    df = load_labels()
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL.read_bytes()),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
        output_face_blendshapes=True,
    )
    names, lms, sizes, poses, blends = [], [], [], [], []
    blend_names = None
    with FaceLandmarker.create_from_options(opts) as lmk:
        for i, row in df.iterrows():
            rgb = np.asarray(Image.open(DATA / "Images" / row["name"]).convert("RGB"))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = lmk.detect(img)
            if not res.face_landmarks:
                continue
            arr = np.array([[l.x, l.y, l.z] for l in res.face_landmarks[0]], dtype=np.float32)
            pose = head_pose(np.array(res.facial_transformation_matrixes[0]))
            names.append(row["name"])
            lms.append(arr)
            sizes.append((img.width, img.height))
            poses.append((pose["yaw"], pose["pitch"]))
            blend_names = [c.category_name for c in res.face_blendshapes[0]]
            blends.append([c.score for c in res.face_blendshapes[0]])
            if i % 500 == 0:
                print(f"{i}/{len(df)}")
    print(f"detected {len(names)}/{len(df)} faces")
    np.savez_compressed(OUT / "landmarks.npz", names=np.array(names), landmarks=np.stack(lms),
                        sizes=np.array(sizes), poses=np.array(poses),
                        blendshapes=np.array(blends, dtype=np.float32), blend_names=np.array(blend_names))


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
