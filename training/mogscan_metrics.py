"""Facial-geometry metrics computed from MediaPipe FaceLandmarker (478 landmarks).

This file is the reference implementation. app/src/scoring/metrics.ts is a line-by-line
port and is checked against fixtures produced by this module (see make_parity_fixture.py).

Pipeline: MediaPipe landmarks -> to_3d (pixel x/y, z scaled by width) -> frontalize
(rigid Procrustes alignment to a mean frontal template, removing yaw/pitch/roll) ->
compute_metrics on the x/y of the frontalized points. Every length is normalised by the
inter-pupillary distance (IPD) or by face width, so metrics are invariant to camera distance.
"""
import math

import numpy as np

# MediaPipe canonical face-mesh indices. "R" = subject's right (image left).
IDX = dict(
    r_eye_outer=33, r_eye_inner=133, r_lid_up=159, r_lid_low=145,
    l_eye_inner=362, l_eye_outer=263, l_lid_up=386, l_lid_low=374,
    r_iris=468, l_iris=473,
    r_brow=105, l_brow=334, glabella=9,
    forehead=10, subnasale=2, nose_r=129, nose_l=358,
    mouth_r=61, mouth_l=291, lip_up_top=0, lip_up_bot=13, lip_low_top=14, lip_low_bot=17,
    menton=152, zyg_r=234, zyg_l=454, gon_r=172, gon_l=397,
)

MIDLINE = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 0, 13, 14, 17, 152]
SYM_PAIRS = [(33, 263), (133, 362), (159, 386), (145, 374), (61, 291), (234, 454),
             (172, 397), (129, 358), (105, 334), (70, 300), (58, 288), (132, 361), (93, 323)]

METRIC_NAMES = [
    "canthal_tilt", "eye_spacing", "fwhr", "midface_ratio", "lower_third",
    "chin_philtrum", "nose_intercanthal", "mouth_nose", "jaw_cheek", "lip_ratio",
    "eye_aspect", "brow_eye", "face_lw", "symmetry", "jaw_angle",
]


def to_3d(lm_norm: np.ndarray, img_w: float, img_h: float) -> np.ndarray:
    """MediaPipe normalised (x, y, z) -> pixel-space 3D (z shares the x scale)."""
    lm = np.asarray(lm_norm, dtype=np.float64)
    return np.stack([lm[:, 0] * img_w, lm[:, 1] * img_h, lm[:, 2] * img_w], axis=1)


def normalise_shape(p3: np.ndarray) -> np.ndarray:
    """Centre on the mean landmark and scale to unit RMS radius in the image plane."""
    c = p3 - p3.mean(axis=0)
    return c / math.sqrt(float((c[:, :2] ** 2).sum(axis=1).mean()))


def best_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Horn's quaternion method: rotation R minimising ||src @ R.T - dst|| (both centred)."""
    S = src.T @ dst
    sxx, sxy, sxz = S[0]
    syx, syy, syz = S[1]
    szx, szy, szz = S[2]
    N = np.array([
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ])
    w, v = np.linalg.eigh(N)
    q0, q1, q2, q3 = v[:, int(np.argmax(w))]
    return np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3],
    ])


def frontalize(p3: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Rigidly rotate a face onto the frontal template; returns normalised 3D points."""
    q = normalise_shape(p3)
    return q @ best_rotation(q, template).T


def apparent_pose(p3: np.ndarray, template: np.ndarray) -> dict:
    """Yaw/pitch (degrees) of the face *as it appears in the image*, relative to the frontal
    template. Unlike MediaPipe's transformation matrix, this is not affected by where the face sits
    in the frame (off-axis perspective), so it is what the live pose gate uses."""
    r = best_rotation(normalise_shape(p3), template)
    return {"yaw": math.degrees(math.atan2(r[0, 2], r[2, 2])), "pitch": math.degrees(math.atan2(-r[1, 2], r[2, 2]))}


def build_template(shapes: list) -> np.ndarray:
    """Generalised Procrustes mean of (already near-frontal) 3D shapes."""
    t = np.mean([normalise_shape(s) for s in shapes], axis=0)
    for _ in range(3):
        t = normalise_shape(np.mean([frontalize(s, t) for s in shapes], axis=0))
    return t


def _roll_normalise(pts: np.ndarray) -> np.ndarray:
    r, l = pts[IDX["r_iris"]], pts[IDX["l_iris"]]
    ang = math.atan2(l[1] - r[1], l[0] - r[0])
    c, s = math.cos(-ang), math.sin(-ang)
    centre = (r + l) / 2
    d = pts - centre
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], axis=1)


def _dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _angle(a, b, c) -> float:
    """Angle ABC in degrees."""
    v1, v2 = a - b, c - b
    cosv = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def compute_metrics(pts_px: np.ndarray) -> dict:
    """pts_px: (478, 2) array of pixel coordinates. Returns dict of METRIC_NAMES -> float."""
    p = _roll_normalise(np.asarray(pts_px, dtype=np.float64)[:, :2])
    P = {k: p[v] for k, v in IDX.items()}
    ipd = _dist(P["r_iris"], P["l_iris"])
    bizyg = _dist(P["zyg_r"], P["zyg_l"])

    # Canthal tilt: outer canthus above inner canthus => positive (image y points down).
    tilt_r = math.degrees(math.atan2(P["r_eye_inner"][1] - P["r_eye_outer"][1],
                                     abs(P["r_eye_outer"][0] - P["r_eye_inner"][0])))
    tilt_l = math.degrees(math.atan2(P["l_eye_inner"][1] - P["l_eye_outer"][1],
                                     abs(P["l_eye_outer"][0] - P["l_eye_inner"][0])))

    nose_w = _dist(P["nose_r"], P["nose_l"])
    intercanthal = _dist(P["r_eye_inner"], P["l_eye_inner"])
    mouth_w = _dist(P["mouth_r"], P["mouth_l"])
    philtrum = abs(P["lip_up_top"][1] - P["subnasale"][1])
    chin = abs(P["menton"][1] - P["lip_low_bot"][1])
    mid_third = abs(P["subnasale"][1] - P["glabella"][1])
    low_third = abs(P["menton"][1] - P["subnasale"][1])
    upper_face_h = abs(P["lip_up_top"][1] - P["glabella"][1])
    pupil_y = (P["r_iris"][1] + P["l_iris"][1]) / 2
    up_lip = abs(P["lip_up_bot"][1] - P["lip_up_top"][1])
    low_lip = abs(P["lip_low_bot"][1] - P["lip_low_top"][1])
    ear_r = _dist(P["r_lid_up"], P["r_lid_low"]) / _dist(P["r_eye_outer"], P["r_eye_inner"])
    ear_l = _dist(P["l_lid_up"], P["l_lid_low"]) / _dist(P["l_eye_outer"], P["l_eye_inner"])
    brow_eye = ((P["r_lid_up"][1] - P["r_brow"][1]) + (P["l_lid_up"][1] - P["l_brow"][1])) / 2

    mid_x = float(np.mean(p[MIDLINE, 0]))
    sym = 0.0
    for a, b in SYM_PAIRS:
        mirrored = np.array([2 * mid_x - p[b, 0], p[b, 1]])
        sym += _dist(p[a], mirrored)
    sym /= len(SYM_PAIRS) * ipd

    jaw_angle = (_angle(P["zyg_r"], P["gon_r"], P["menton"]) +
                 _angle(P["zyg_l"], P["gon_l"], P["menton"])) / 2

    return {
        "canthal_tilt": (tilt_r + tilt_l) / 2,
        "eye_spacing": ipd / bizyg,
        "fwhr": bizyg / upper_face_h,
        "midface_ratio": abs(P["lip_up_top"][1] - pupil_y) / ipd,
        "lower_third": low_third / (mid_third + low_third),
        "chin_philtrum": chin / philtrum,
        "nose_intercanthal": nose_w / intercanthal,
        "mouth_nose": mouth_w / nose_w,
        "jaw_cheek": _dist(P["gon_r"], P["gon_l"]) / bizyg,
        "lip_ratio": low_lip / max(up_lip, 1e-6),
        "eye_aspect": (ear_r + ear_l) / 2,
        "brow_eye": brow_eye / ipd,
        "face_lw": _dist(P["forehead"], P["menton"]) / bizyg,
        "symmetry": sym,
        "jaw_angle": jaw_angle,
    }


def crop_box(pts_px: np.ndarray, img_w: int, img_h: int, scale: float = 1.35):
    """Square crop around the landmark bbox. Same rule is used live in the browser."""
    x0, y0 = pts_px[:, 0].min(), pts_px[:, 1].min()
    x1, y1 = pts_px[:, 0].max(), pts_px[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * scale
    return cx - side / 2, cy - side / 2, side


def head_pose(matrix: np.ndarray) -> dict:
    """Yaw/pitch in degrees from MediaPipe's 4x4 facial transformation matrix (column-major
    rotation in the upper-left 3x3). 0/0 is a face looking straight into the camera."""
    r = np.asarray(matrix)[:3, :3]
    yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
    pitch = math.degrees(math.atan2(-r[1, 2], r[2, 2]))
    return {"yaw": yaw, "pitch": pitch}
