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
    r_brow=105, l_brow=334, r_brow_low=52, l_brow_low=282, r_brow_in=107, l_brow_in=336,
    r_brow_tail=70, l_brow_tail=300, glabella=9, nasion=168,
    forehead=10, subnasale=2, nose_tip=1, nose_r=129, nose_l=358, bridge_r=193, bridge_l=417,
    mouth_r=61, mouth_l=291, lip_up_top=0, lip_up_bot=13, lip_low_top=14, lip_low_bot=17,
    bow_r=37, bow_l=267, pogonion=175, chin_r=176, chin_l=400,
    menton=152, zyg_r=234, zyg_l=454, gon_r=172, gon_l=397, malar_r=116, malar_l=345,
    temple_r=54, temple_l=284, tip_r=45, tip_l=275, cheek_low_r=58, cheek_low_l=288,
)

R_IRIS_RING, L_IRIS_RING = [469, 470, 471, 472], [474, 475, 476, 477]
MIDLINE = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 0, 13, 14, 17, 152]
SYM_REGIONS = dict(
    eyes=[(33, 263), (133, 362), (159, 386), (145, 374)],
    brows=[(105, 334), (70, 300), (107, 336), (52, 282)],
    nose=[(129, 358), (193, 417), (64, 294)],
    mouth=[(61, 291), (37, 267), (84, 314)],
    jaw=[(234, 454), (172, 397), (58, 288), (132, 361), (93, 323), (176, 400)],
)
SYM_PAIRS = [(33, 263), (133, 362), (159, 386), (145, 374), (61, 291), (234, 454),
             (172, 397), (129, 358), (105, 334), (70, 300), (58, 288), (132, 361), (93, 323)]

# Every geometric measurement, grouped. All lengths are divided by IPD, face width or face
# height; depth (z) comes from the frontalized 3D mesh and is also divided by IPD.
METRIC_GROUPS = dict(
    eyes=["canthal_tilt", "canthal_asym", "eye_spacing", "intercanthal_eye", "eye_width", "eye_aspect",
          "iris_fill", "scleral_show", "upper_scleral"],
    brows=["brow_eye", "brow_tilt", "brow_thickness", "brow_length", "brow_gap"],
    nose=["nose_intercanthal", "nose_face", "nose_length", "nose_bridge", "nose_tip_width"],
    mouth=["mouth_nose", "mouth_ipd", "lip_ratio", "lip_fullness", "cupid_bow", "mouth_tilt", "philtrum_lower"],
    jaw=["chin_philtrum", "chin_height", "chin_width", "jaw_cheek", "jaw_angle", "jaw_frame", "chin_angle",
         "ramus"],
    proportions=["fwhr", "midface_ratio", "lower_third", "upper_third", "face_lw", "temple_cheek",
                 "cheekbone_height", "lower_face_fullness"],
    depth=["nasal_projection", "chin_projection", "lip_projection", "malar_projection", "orbital_depth",
           "forehead_slope", "facial_convexity", "face_depth", "orbital_vector", "midface_projection"],
    symmetry=["symmetry", "sym_eyes", "sym_brows", "sym_nose", "sym_mouth", "sym_jaw"],
)
METRIC_NAMES = [m for g in METRIC_GROUPS.values() for m in g]

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


def _sym(p: np.ndarray, pairs, mid_x: float, ipd: float) -> float:
    """Mean distance between each left point and its mirrored right partner, in IPD units."""
    return sum(_dist(p[a], np.array([2 * mid_x - p[b, 0], p[b, 1]])) for a, b in pairs) / (len(pairs) * ipd)


def compute_metrics(pts: np.ndarray) -> dict:
    """pts: (478, 2) or (478, 3) frontalized landmarks. Returns dict of METRIC_NAMES -> float.
    Depth metrics need z; with 2D input they are 0."""
    pts = np.asarray(pts, dtype=np.float64)
    p = _roll_normalise(pts[:, :2])
    z = pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts))
    P = {k: p[v] for k, v in IDX.items()}
    Z = {k: z[v] for k, v in IDX.items()}
    ipd = _dist(P["r_iris"], P["l_iris"])
    bizyg = _dist(P["zyg_r"], P["zyg_l"])
    bigon = _dist(P["gon_r"], P["gon_l"])

    # Canthal tilt: outer canthus above inner canthus => positive (image y points down).
    tilt_r = math.degrees(math.atan2(P["r_eye_inner"][1] - P["r_eye_outer"][1],
                                     abs(P["r_eye_outer"][0] - P["r_eye_inner"][0])))
    tilt_l = math.degrees(math.atan2(P["l_eye_inner"][1] - P["l_eye_outer"][1],
                                     abs(P["l_eye_outer"][0] - P["l_eye_inner"][0])))
    brow_tilt_r = math.degrees(math.atan2(P["r_brow_in"][1] - P["r_brow_tail"][1],
                                          abs(P["r_brow_tail"][0] - P["r_brow_in"][0])))
    brow_tilt_l = math.degrees(math.atan2(P["l_brow_in"][1] - P["l_brow_tail"][1],
                                          abs(P["l_brow_tail"][0] - P["l_brow_in"][0])))

    nose_w = _dist(P["nose_r"], P["nose_l"])
    intercanthal = _dist(P["r_eye_inner"], P["l_eye_inner"])
    eye_w = (_dist(P["r_eye_outer"], P["r_eye_inner"]) + _dist(P["l_eye_outer"], P["l_eye_inner"])) / 2
    mouth_w = _dist(P["mouth_r"], P["mouth_l"])
    philtrum = abs(P["lip_up_top"][1] - P["subnasale"][1])
    chin = abs(P["menton"][1] - P["lip_low_bot"][1])
    mid_third = abs(P["subnasale"][1] - P["glabella"][1])
    low_third = abs(P["menton"][1] - P["subnasale"][1])
    face_len = _dist(P["forehead"], P["menton"])
    upper_face_h = abs(P["lip_up_top"][1] - P["glabella"][1])
    pupil_y = (P["r_iris"][1] + P["l_iris"][1]) / 2
    up_lip = abs(P["lip_up_bot"][1] - P["lip_up_top"][1])
    low_lip = abs(P["lip_low_bot"][1] - P["lip_low_top"][1])
    ear_r = _dist(P["r_lid_up"], P["r_lid_low"]) / _dist(P["r_eye_outer"], P["r_eye_inner"])
    ear_l = _dist(P["l_lid_up"], P["l_lid_low"]) / _dist(P["l_eye_outer"], P["l_eye_inner"])
    brow_eye = ((P["r_lid_up"][1] - P["r_brow"][1]) + (P["l_lid_up"][1] - P["l_brow"][1])) / 2

    # Iris: diameter = 2 x mean ring radius; sclera visible below / above the iris.
    iris_d, show_low, show_up = [], [], []
    for c, ring, lid_up, lid_low in ((P["r_iris"], R_IRIS_RING, P["r_lid_up"], P["r_lid_low"]),
                                     (P["l_iris"], L_IRIS_RING, P["l_lid_up"], P["l_lid_low"])):
        d = 2 * float(np.mean([_dist(c, p[i]) for i in ring]))
        iris_d.append(d)
        show_low.append((lid_low[1] - (c[1] + d / 2)) / d)
        show_up.append(((c[1] - d / 2) - lid_up[1]) / d)

    mid_x = float(np.mean(p[MIDLINE, 0]))
    jaw_angle = (_angle(P["zyg_r"], P["gon_r"], P["menton"]) +
                 _angle(P["zyg_l"], P["gon_l"], P["menton"])) / 2

    # Pseudo-profile in the sagittal (y, z) plane. MediaPipe z decreases toward the camera.
    def yz(k):
        return np.array([P[k][1], Z[k]])
    t = (P["lip_low_bot"][1] - P["nose_tip"][1]) / (P["pogonion"][1] - P["nose_tip"][1] + 1e-9)
    e_line_z = Z["nose_tip"] + t * (Z["pogonion"] - Z["nose_tip"])

    return {
        "canthal_tilt": (tilt_r + tilt_l) / 2,
        "canthal_asym": abs(tilt_r - tilt_l),
        "eye_spacing": ipd / bizyg,
        "intercanthal_eye": intercanthal / eye_w,
        "eye_width": eye_w / bizyg,
        "eye_aspect": (ear_r + ear_l) / 2,
        "iris_fill": float(np.mean(iris_d)) / eye_w,
        "scleral_show": float(np.mean(show_low)),
        "upper_scleral": float(np.mean(show_up)),
        "brow_eye": brow_eye / ipd,
        "brow_tilt": (brow_tilt_r + brow_tilt_l) / 2,
        "brow_thickness": ((P["r_brow_low"][1] - P["r_brow"][1]) + (P["l_brow_low"][1] - P["l_brow"][1])) / (2 * ipd),
        "brow_length": (_dist(P["r_brow_in"], P["r_brow_tail"]) + _dist(P["l_brow_in"], P["l_brow_tail"])) / (2 * eye_w),
        "brow_gap": _dist(P["r_brow_in"], P["l_brow_in"]) / intercanthal,
        "nose_intercanthal": nose_w / intercanthal,
        "nose_face": nose_w / bizyg,
        "nose_length": abs(P["subnasale"][1] - P["nasion"][1]) / abs(P["menton"][1] - P["nasion"][1]),
        "nose_bridge": _dist(P["bridge_r"], P["bridge_l"]) / nose_w,
        "nose_tip_width": _dist(P["tip_r"], P["tip_l"]) / nose_w,
        "mouth_nose": mouth_w / nose_w,
        "mouth_ipd": mouth_w / ipd,
        "lip_ratio": low_lip / max(up_lip, 1e-6),
        "lip_fullness": (up_lip + low_lip) / low_third,
        "cupid_bow": (P["lip_up_top"][1] - (P["bow_r"][1] + P["bow_l"][1]) / 2) / max(up_lip, 1e-6),
        "mouth_tilt": ((P["lip_up_bot"][1] + P["lip_low_top"][1]) / 2 - (P["mouth_r"][1] + P["mouth_l"][1]) / 2) / mouth_w,
        "philtrum_lower": philtrum / low_third,
        "chin_philtrum": chin / philtrum,
        "chin_height": chin / low_third,
        "chin_width": _dist(P["chin_r"], P["chin_l"]) / bigon,
        "jaw_cheek": bigon / bizyg,
        "jaw_angle": jaw_angle,
        "jaw_frame": bigon / face_len,
        "chin_angle": _angle(P["gon_r"], P["menton"], P["gon_l"]),
        "ramus": (_dist(P["zyg_r"], P["gon_r"]) + _dist(P["zyg_l"], P["gon_l"])) / (2 * face_len),
        "fwhr": bizyg / upper_face_h,
        "midface_ratio": abs(P["lip_up_top"][1] - pupil_y) / ipd,
        "lower_third": low_third / (mid_third + low_third),
        "upper_third": abs(P["glabella"][1] - P["forehead"][1]) / abs(P["menton"][1] - P["forehead"][1]),
        "face_lw": face_len / bizyg,
        "temple_cheek": _dist(P["temple_r"], P["temple_l"]) / bizyg,
        "cheekbone_height": (P["subnasale"][1] - (P["malar_r"][1] + P["malar_l"][1]) / 2) / (P["subnasale"][1] - pupil_y),
        "lower_face_fullness": _dist(P["cheek_low_r"], P["cheek_low_l"]) / bizyg,
        "nasal_projection": ((Z["nose_r"] + Z["nose_l"]) / 2 - Z["nose_tip"]) / ipd,
        "chin_projection": (Z["lip_low_bot"] - Z["pogonion"]) / ipd,
        "lip_projection": (e_line_z - Z["lip_low_bot"]) / ipd,
        "malar_projection": ((Z["zyg_r"] + Z["zyg_l"]) / 2 - (Z["malar_r"] + Z["malar_l"]) / 2) / ipd,
        "orbital_depth": ((Z["r_iris"] + Z["l_iris"]) / 2 - (Z["r_brow"] + Z["l_brow"]) / 2) / ipd,
        "forehead_slope": (Z["forehead"] - Z["glabella"]) / ipd,
        "facial_convexity": _angle(yz("glabella"), yz("subnasale"), yz("pogonion")),
        "face_depth": ((Z["zyg_r"] + Z["zyg_l"]) / 2 - Z["nose_tip"]) / bizyg,
        # Orbital vector: cheekbone front vs. cornea (positive = cheek projects past the eye).
        "orbital_vector": ((Z["r_iris"] + Z["l_iris"]) / 2 - (Z["malar_r"] + Z["malar_l"]) / 2) / ipd,
        # Midface (maxilla) projection vs. the brow; low values = recessed midface.
        "midface_projection": (Z["glabella"] - Z["subnasale"]) / ipd,
        "symmetry": _sym(p, SYM_PAIRS, mid_x, ipd),
        **{f"sym_{k}": _sym(p, v, mid_x, ipd) for k, v in SYM_REGIONS.items()},
    }


SHAPE_PCS = 12
SHAPE_NAMES = ["averageness", "dimorphism"] + [f"shape_pc{k + 1}" for k in range(SHAPE_PCS)]


def build_shape_model(frontal_shapes: list, is_male: np.ndarray, template: np.ndarray) -> dict:
    """Whole-face shape space from frontalized training shapes: the male-minus-female mean
    direction (sexual dimorphism axis) and the top principal components of shape residuals."""
    r = np.stack([(f - template).ravel() for f in frontal_shapes])
    axis = r[is_male].mean(axis=0) - r[~is_male].mean(axis=0)
    mean = r.mean(axis=0)
    _, sv, vt = np.linalg.svd(r - mean, full_matrices=False)
    return {"axis": axis / np.linalg.norm(axis), "mean": mean, "pcs": vt[:SHAPE_PCS],
            "pcStd": sv[:SHAPE_PCS] / math.sqrt(len(r) - 1)}


def compute_shape(f3: np.ndarray, template: np.ndarray, shape: dict) -> dict:
    """f3: frontalized normalised 3D shape (output of frontalize). Averageness is the RMS
    distance to the mean face; lower means more average (known to read as attractive)."""
    r = (np.asarray(f3) - template).ravel()
    out = {"averageness": math.sqrt(float((r ** 2).sum()) / len(f3)),
           "dimorphism": float(r @ np.asarray(shape["axis"]))}
    pc = (np.asarray(shape["pcs"]) @ (r - np.asarray(shape["mean"]))) / np.asarray(shape["pcStd"])
    out.update({f"shape_pc{k + 1}": float(v) for k, v in enumerate(pc)})
    return out


# Appearance is measured on a small square crop (same box as the CNN crop) so it costs ~nothing.
APPEAR_SIZE = 160
SKIN_PTS = [50, 280, 151, 199]   # cheeks, forehead, chin
CHEEK_PTS = [50, 280]
LIP_PTS = [11, 16]
JAW_PTS = [172, 136, 150, 149, 176, 148, 397, 365, 379, 378, 400, 377]
APPEARANCE_GROUPS = dict(
    skin=["skin_lum", "skin_tone_var", "skin_texture", "skin_redness", "skin_yellow", "under_eye"],
    colouring=["lip_contrast", "lip_redness", "eye_contrast", "sclera_bright", "brow_contrast", "lash_line",
               "iris_lightness", "iris_blue"],
    hair=["hair_darkness", "hair_texture", "facial_hair"],
    leanness=["jaw_definition", "cheek_hollow", "neck_width"],
)
APPEARANCE_NAMES = [m for g in APPEARANCE_GROUPS.values() for m in g]


def _patch(img: np.ndarray, cx: float, cy: float, r: int) -> np.ndarray:
    """Pixels in the (2r+1)^2 box around (cx, cy), clamped to the image. img is (S, S, 3) float."""
    s = img.shape[0]
    x, y = int(math.floor(cx)), int(math.floor(cy))
    x0, x1 = min(max(x - r, 0), s - 1), min(max(x + r, 0), s - 1)
    y0, y1 = min(max(y - r, 0), s - 1), min(max(y + r, 0), s - 1)
    return img[y0:y1 + 1, x0:x1 + 1]


def _lum(px: np.ndarray) -> np.ndarray:
    return 0.299 * px[..., 0] + 0.587 * px[..., 1] + 0.114 * px[..., 2]


def _texture(px: np.ndarray) -> float:
    """Mean absolute 4-neighbour Laplacian of luminance (fine skin texture / blemishes)."""
    y = _lum(px)
    if y.shape[0] < 3 or y.shape[1] < 3:
        return 0.0
    lap = 4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    return float(np.abs(lap).mean())


def compute_appearance(rgb: np.ndarray, pts: np.ndarray) -> dict:
    """rgb: (S, S, 3) uint8 crop; pts: (478, 2) landmarks in crop pixel coordinates."""
    img = np.asarray(rgb, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    ipd = _dist(pts[468], pts[473])
    rad = lambda k: max(1, int(math.floor(k * ipd + 0.5)))
    mean_rgb = lambda i, k: _patch(img, *pts[i], rad(k)).reshape(-1, 3).mean(axis=0)
    chroma = lambda c: (c[0] - c[1]) / (c.sum() + 1e-6)

    skin = [mean_rgb(i, 0.08) for i in SKIN_PTS]
    skin_y = np.array([float(_lum(c)) for c in skin])
    sy = float(skin_y.mean()) + 1e-6
    cheeks = [mean_rgb(i, 0.08) for i in CHEEK_PTS]
    lip = np.mean([mean_rgb(i, 0.03) for i in LIP_PTS], axis=0)
    iris = np.mean([mean_rgb(i, 0.04) for i in (468, 473)], axis=0)
    sclera = []
    for c, corners in ((468, (33, 133)), (473, (263, 362))):
        for k in corners:
            q = pts[c] + 0.65 * (pts[k] - pts[c])
            sclera.append(_patch(img, q[0], q[1], rad(0.02)).reshape(-1, 3).mean(axis=0))
    brow = np.mean([_patch(img, *((pts[a] + pts[b]) / 2), rad(0.03)).reshape(-1, 3).mean(axis=0)
                    for a, b in ((105, 52), (334, 282))], axis=0)
    under = np.mean([_patch(img, *(pts[a] + 0.35 * (pts[b] - pts[a])), rad(0.04)).reshape(-1, 3).mean(axis=0)
                     for a, b in ((145, 50), (374, 280))], axis=0)
    cheek_y = float(np.mean([_lum(c) for c in cheeks])) + 1e-6
    lum_at = lambda q, k: float(_lum(_patch(img, q[0], q[1], rad(k)).reshape(-1, 3).mean(axis=0)))

    # Jaw definition: luminance step across the jawline (shadow under a defined jaw).
    centre = pts.mean(axis=0)
    steps = []
    for i in JAW_PTS:
        u = pts[i] - centre
        u = u / (np.linalg.norm(u) + 1e-9) * 0.06 * ipd
        steps.append(abs(lum_at(pts[i] - u, 0.02) - lum_at(pts[i] + u, 0.02)))
    submalar = [pts[a] + 0.5 * (pts[b] - pts[a]) for a, b in ((116, 172), (345, 397))]
    hair_pt = pts[10] + 0.35 * (pts[10] - pts[168])
    hair_px = _patch(img, hair_pt[0], hair_pt[1], rad(0.1))
    beard = np.mean([lum_at((pts[0] + pts[2]) / 2, 0.03), lum_at(pts[199], 0.06)])
    return {
        "skin_lum": sy / 255,
        "skin_tone_var": float(skin_y.std()) / sy,
        "skin_texture": float(np.mean([_texture(_patch(img, *pts[i], rad(0.12))) for i in CHEEK_PTS])) / sy,
        "skin_redness": float(np.mean([chroma(c) for c in cheeks])),
        "skin_yellow": float(np.mean([((c[0] + c[1]) / 2 - c[2]) / (c.sum() + 1e-6) for c in cheeks])),
        "lip_contrast": (sy - float(_lum(lip))) / sy,
        "lip_redness": float(chroma(lip)) - float(np.mean([chroma(c) for c in cheeks])),
        "eye_contrast": (sy - float(_lum(iris))) / sy,
        "sclera_bright": float(np.mean([_lum(c) for c in sclera])) / sy,
        "brow_contrast": (float(skin_y[2]) - float(_lum(brow))) / (float(skin_y[2]) + 1e-6),
        "under_eye": float(_lum(under)) / cheek_y,
        "lash_line": np.mean([lum_at(pts[i], 0.015) for i in (159, 386)]) / sy,
        "iris_lightness": float(_lum(iris)) / 255,
        "iris_blue": float((iris[2] - iris[0]) / (iris.sum() + 1e-6)),
        "hair_darkness": (sy - float(_lum(hair_px.reshape(-1, 3).mean(axis=0)))) / sy,
        "hair_texture": _texture(hair_px) / sy,
        "facial_hair": (cheek_y - float(beard)) / cheek_y,
        "jaw_definition": float(np.mean(steps)) / sy,
        "cheek_hollow": float(np.mean([lum_at(q, 0.04) for q in submalar])) /
                        (float(np.mean([lum_at(pts[i], 0.04) for i in (116, 345)])) + 1e-6),
        "neck_width": _neck_width(img, pts, np.mean(cheeks, axis=0)) / (_dist(pts[234], pts[454]) + 1e-6),
    }


def _neck_width(img: np.ndarray, pts: np.ndarray, skin: np.ndarray) -> float:
    """Width of the skin-coloured run through the neck, a little below the chin (pixels)."""
    s = img.shape[0]
    face_len = _dist(pts[10], pts[152])
    y = min(max(int(math.floor(pts[152][1] + 0.12 * face_len)), 0), s - 1)
    x_mid = min(max(int(math.floor(pts[152][0])), 0), s - 1)
    ref = float(np.linalg.norm(skin)) + 1e-6
    is_skin = lambda x: float(np.linalg.norm(img[y, x] - skin)) / ref < 0.25
    if not is_skin(x_mid):
        return 0.0
    x0 = x_mid
    while x0 > 0 and is_skin(x0 - 1):
        x0 -= 1
    x1 = x_mid
    while x1 < s - 1 and is_skin(x1 + 1):
        x1 += 1
    return float(x1 - x0 + 1)


def appearance_crop(img, pts_px: np.ndarray, img_w: int, img_h: int):
    """PIL image + pixel landmarks -> (S, S, 3) crop and landmarks in crop coordinates.
    The browser does the same with drawImage on an S x S canvas."""
    from PIL import Image
    x, y, side = crop_box(pts_px, img_w, img_h)
    x, y, side = round(x), round(y), round(side)
    crop = img.crop((x, y, x + side, y + side)).resize((APPEAR_SIZE, APPEAR_SIZE), Image.BILINEAR)
    k = APPEAR_SIZE / side
    return np.asarray(crop.convert("RGB")), (pts_px[:, :2] - [x, y]) * k


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
