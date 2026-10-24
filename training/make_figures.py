"""Figures and tables for the write-up (no face images, only numbers).

  writeup/figures/scatter.pdf        fused prediction vs. human rating on the test split
  writeup/figures/metric_corr.pdf    which looksmax features actually track human ratings (top 25)
  writeup/figures/ablation.pdf       what each data source adds to the explainable model
  writeup/generated/*.tex            LaTeX macros/tables filled from results JSON
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).parent
OUT = ROOT / "out"
FIG = ROOT.parent / "writeup" / "figures"
GEN = ROOT.parent / "writeup" / "generated"
LABELS = {"canthal_tilt": "Canthal tilt", "eye_spacing": "Eye spacing (ESR)", "fwhr": "FWHR",
          "midface_ratio": "Midface ratio", "lower_third": "Lower third", "chin_philtrum": "Chin : philtrum",
          "nose_intercanthal": "Nose : intercanthal", "mouth_nose": "Mouth : nose", "jaw_cheek": "Jaw : cheekbone",
          "lip_ratio": "Lower : upper lip", "eye_aspect": "Eye openness", "brow_eye": "Brow-eye distance",
          "face_lw": "Face length : width", "symmetry": "Asymmetry", "jaw_angle": "Gonial angle",
          "averageness": "Averageness", "dimorphism": "Shape dimorphism", "skin_texture": "Skin texture",
          "skin_lum": "Skin brightness", "lip_contrast": "Lip contrast", "eye_contrast": "Eye contrast",
          "sclera_bright": "Sclera brightness", "jaw_frame": "Jaw width : face length", "chin_angle": "Chin taper",
          "orbital_vector": "Orbital vector", "midface_projection": "Midface projection",
          "facial_convexity": "Facial convexity", "cheekbone_height": "Cheekbone height",
          "lower_face_fullness": "Lower-face fullness", "jaw_definition": "Jaw definition",
          "brow_thickness": "Brow thickness", "nose_face": "Nose : face width", "iris_fill": "Iris : eye width",
          "eye_width": "Eye width : face width", "upper_scleral": "Upper scleral show", "temple_cheek": "Temple : cheekbone",
          "upper_third": "Upper third", "orbital_depth": "Brow ridge depth", "intercanthal_eye": "Intercanthal : eye",
          "brow_length": "Brow length", "lash_line": "Lash-line brightness", "ramus": "Ramus length"}
GROUP_LABELS = {"eyes": "Eyes", "brows": "Brows", "nose": "Nose", "mouth": "Mouth & lips", "jaw": "Jaw & chin",
                "proportions": "Proportions", "depth": "Profile depth (3D)", "symmetry": "Symmetry",
                "shape": "Whole-face shape", "skin": "Skin", "colouring": "Colouring & contrast",
                "hair": "Hair & grooming", "leanness": "Leanness", "expression": "Expression"}


def label(m):
    if m in LABELS:
        return LABELS[m]
    if m.startswith("bs_"):
        return "expr: " + m[3:]
    return m.replace("_", " ")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def main():
    FIG.mkdir(parents=True, exist_ok=True); GEN.mkdir(parents=True, exist_ok=True)
    res = json.load(open(OUT / "results.json"))
    t = pd.read_csv(OUT / "test_preds.csv")

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    for (race, sex), g in t.groupby(["race", "sex"]):
        ax.scatter(g.score, g.fused, s=3, alpha=0.45, label=f"{race[0]}{sex}")
    ax.plot([1, 5], [1, 5], color="black", lw=0.7)
    ax.set_xlabel("Mean human rating (60 raters)"); ax.set_ylabel("MogScan fused prediction")
    ax.set_title(f"Test split, n={len(t)}, r={res['fused']['pearson']:.3f}", fontsize=8)
    ax.legend(markerscale=3, fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "scatter.pdf"); fig.savefig(FIG / "scatter.png", dpi=200)

    geo = json.load(open(OUT / "results_geometric.json"))
    groups = json.load(open(OUT / "feature_groups.json"))
    all_corr = pd.DataFrame({s: geo[s]["metric_corr"] for s in ("M", "F")})
    top = all_corr.mean(axis=1).abs().sort_values(ascending=False).index[:25]
    corr = all_corr.loc[top]
    corr = corr.reindex(corr.mean(axis=1).sort_values().index)
    fig, ax = plt.subplots(figsize=(3.3, 4.6))
    y = np.arange(len(corr))
    ax.barh(y - 0.2, corr.M, height=0.4, label="male faces", color="#4a6cf7")
    ax.barh(y + 0.2, corr.F, height=0.4, label="female faces", color="#e4572e")
    ax.set_yticks(y, [label(m) for m in corr.index]); ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Pearson r with human rating (train split)")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "metric_corr.pdf", bbox_inches="tight"); fig.savefig(FIG / "metric_corr.png", dpi=200, bbox_inches="tight")

    ab = geo["ablation"]
    full = geo["all"]["pearson"]
    order = sorted(groups, key=lambda g: ab["only"][g])
    fig, ax = plt.subplots(figsize=(3.3, 3.4))
    y = np.arange(len(order))
    ax.barh(y, [ab["only"][g] for g in order], color="#4a6cf7", label="this group alone")
    ax.scatter([10 * (full - ab["without"][g]) for g in order], y, color="#e4572e", s=10, zorder=3,
               label="drop when removed (x10)")
    ax.set_yticks(y, [f"{GROUP_LABELS[g]} ({len(groups[g])})" for g in order])
    ax.set_xlabel("Test Pearson r of the explainable model")
    ax.axvline(full, color="black", lw=0.6, ls="--")
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "ablation.pdf", bbox_inches="tight"); fig.savefig(FIG / "ablation.png", dpi=200, bbox_inches="tight")

    # LaTeX macros so the write-up never contains hand-typed numbers.
    cnn = json.load(open(OUT / "results_cnn.json"))
    bias = json.load(open(OUT / "results_bias.json"))
    feats = pd.read_csv(OUT / "features.csv")
    m = {
        "nTrain": int((feats.split == "train").sum()), "nTest": res["n_test"],
        "geoR": f"{res['geometric']['pearson']:.3f}", "geoMAE": f"{res['geometric']['mae']:.3f}", "geoRMSE": f"{res['geometric']['rmse']:.3f}",
        "cnnR": f"{res['cnn']['pearson']:.3f}", "cnnMAE": f"{res['cnn']['mae']:.3f}", "cnnRMSE": f"{res['cnn']['rmse']:.3f}",
        "fusedR": f"{res['fused']['pearson']:.3f}", "fusedMAE": f"{res['fused']['mae']:.3f}", "fusedRMSE": f"{res['fused']['rmse']:.3f}",
        "sexAcc": f"{res['sex_accuracy'] * 100:.1f}", "pslMAE": f"{res['psl_mae']:.2f}",
        "tierExact": f"{res['tier_exact'] * 100:.1f}", "tierWithinOne": f"{res['tier_within_one'] * 100:.1f}",
        "wCnn": f"{res['fusion']['wCnn']:.2f}", "wGeo": f"{res['fusion']['wGeo']:.2f}",
        "scoreMean": f"{res['fusion']['scoreMean']:.2f}", "scoreStd": f"{res['fusion']['scoreStd']:.2f}",
        "bestValR": f"{cnn['best_val_pearson']:.3f}",
        "topMetric": label(corr.mean(axis=1).abs().idxmax()),
        "topMetricR": f"{corr.mean(axis=1).loc[corr.mean(axis=1).abs().idxmax()]:.2f}",
        "nFeatures": geo["n_features"], "nGroups": len(groups),
        "nGeometry": sum(len(groups[g]) for g in ("eyes", "brows", "nose", "mouth", "jaw", "proportions", "depth", "symmetry")),
        "nShape": len(groups["shape"]), "nPixel": sum(len(groups[g]) for g in ("skin", "colouring", "hair", "leanness")),
        "nExpression": len(groups["expression"]),
        "origFifteenR": f"{ab['original_15']:.3f}", "freeR": f"{geo['all_free']['pearson']:.3f}",
        "cantTiltR": f"{all_corr.loc['canthal_tilt'].mean():.2f}",
    }
    with open(GEN / "numbers.tex", "w") as f:
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(GEN / "ablation_table.tex", "w") as f:
        for g in sorted(groups, key=lambda g: -ab["only"][g]):
            name = GROUP_LABELS[g].replace("&", "\\&")
            f.write(f"{name} & {len(groups[g])} & {ab['only'][g]:.3f} & {ab['without'][g] - full:+.3f} \\\\\n")
    glow = json.load(open(OUT / "results_glowup.json"))
    glow_names = {"skincare": "Skincare (smoother, even skin)", "redness": "Reduce redness", "undereye": "Brighten under-eyes",
                  "glow": "Self-tanner / bronzer", "jaw": "Jawline contour", "cheeks": "Cheekbone contour",
                  "brows": "Fill in brows", "lashes": "Define lash line", "lips": "Lip colour", "facialhair": "Stubble",
                  "eyes": "Brighter sclera", "combined": "All of the above"}
    cell = lambda a, s: (f"{glow[a][s]['success'] * 100:.0f}\\% & {glow[a][s]['meanPsl']:+.3f}" if s in glow[a] else "-- & --")
    with open(GEN / "glowup_table.tex", "w") as f:
        for a in sorted((a for a in glow if a != "combined"), key=lambda a: -glow[a]["M"]["meanPsl"]) + ["combined"]:
            f.write(f"{glow_names[a]} & {cell(a, 'M')} & {cell(a, 'F')} \\\\\n")
    with open(GEN / "numbers.tex", "a") as f:
        f.write(f"\\newcommand{{\\glowN}}{{{glow['brows']['M']['n'] + glow['brows']['F']['n']}}}\n")
    with open(GEN / "bias_table.tex", "w") as f:
        for b in bias:
            f.write(f"{b['group']} & {b['n']} & {b['pearson']:.3f} & {b['mae']:.3f} & {b['signed_bias']:+.3f} \\\\\n")
    print(m)
    write_study_macros()


def write_study_macros():
    """Macros from the real user study, if it has been run. Otherwise \\studyPending is defined and
    the write-up shows clearly marked placeholders instead of numbers."""
    study = ROOT.parent / "study" / "out"
    lines = []
    survey, logs = study / "survey_summary.json", study / "logs_summary.json"
    if survey.exists():
        s = json.load(open(survey))
        lk = s["likert"]
        pay = s.get("pay", {})
        paying = sum(v for k, v in pay.items() if not str(k).lower().startswith("nothing"))
        vals = {"studyN": s["n"], "susMean": f"{s['sus']['mean']:.1f}", "susSD": f"{s['sus']['sd']:.1f}",
                "npsScore": f"{s['nps']['score']:+.0f}", "payPct": f"{paying / max(1, s['n']) * 100:.0f}"}
        for k in ("ACC", "STABLE", "EXPLAIN", "FEATURES", "PILLARS", "PRIVACY", "BETTER", "FEELBAD"):
            if k in lk:
                vals[f"agree{k.title()}"] = f"{lk[k]['agree_pct']:.0f}"
                vals[f"mean{k.title()}"] = f"{lk[k]['mean']:.2f}"
        lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()]
    else:
        lines.append("\\newcommand{\\studyPending}{}")
    if logs.exists():
        l = json.load(open(logs))
        if l["tracks"]:
            lines += [f"\\newcommand{{\\logSessions}}{{{l['sessions']}}}", f"\\newcommand{{\\logTracks}}{{{l['tracks']}}}",
                      f"\\newcommand{{\\pslStd}}{{{l['psl_std_median']:.3f}}}",
                      f"\\newcommand{{\\fpsMedian}}{{{l['fps_median']:.0f}}}"]
    if not any("logTracks" in x for x in lines):
        lines.append("\\newcommand{\\logsPending}{}")
    (GEN / "study.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
