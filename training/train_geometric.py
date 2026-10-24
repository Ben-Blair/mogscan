"""Explainable feature model: per-sex ridge regression on every face feature we compute.

Inputs are all feature groups from build_features.py (geometry, depth, symmetry, whole-face shape,
skin/contrast appearance and expression blendshapes). Each feature enters as a standardised
linear term z and a centred quadratic term (z^2 - 1), so the model can learn "ideal ranges"
(inverted-U) as well as monotonic effects. Head pose (|yaw|, |pitch - median|) is included as a
nuisance covariate and is zeroed at inference, because the live app only scores near-frontal faces.

Two variants are fitted:
  * free:        ordinary ridge. Reported for comparison only.
  * constrained: each linear weight must agree in sign with the feature's marginal correlation
                 with the rating, and quadratic weights must be <= 0 (an "ideal range").
                 Correlated features otherwise produce sign flips, e.g. positive canthal tilt
                 being reported as a weakness. For features whose good direction is not in
                 dispute (PRIOR_SIGNS: asymmetry, uneven or rough skin, puffiness, ...) the sign is
                 fixed by that prior instead; if the data disagrees, the weight is driven to 0.
                 This variant is shipped to the app.

Ablations: the constrained model is also refit on each group alone, on everything except each
group, and on the original 15 metrics, to show what every data source adds.

Outputs:
  out/geo_preds.csv                     out-of-fold train predictions + test predictions
  out/results_geometric.json            metrics, ablations, per-feature correlations and weights
  ../app/public/models/geometric.json   everything the browser needs to score + explain
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

ROOT = Path(__file__).parent
OUT = ROOT / "out"
APP_MODELS = ROOT.parent / "app" / "public" / "models"
ALPHAS = np.logspace(-2, 3.5, 24)
# +1 = more is better, -1 = less is better. Confounds in the photos (lighting, hair, makeup) can
# otherwise flip these, e.g. "patchy skin" showing up as a strength.
PRIOR_SIGNS = {
    "symmetry": -1, "sym_eyes": -1, "sym_brows": -1, "sym_nose": -1, "sym_mouth": -1, "sym_jaw": -1,
    "canthal_asym": -1, "averageness": -1, "skin_tone_var": -1, "skin_texture": -1, "under_eye": 1,
    "sclera_bright": 1, "nose_tip_width": -1, "lower_face_fullness": -1, "jaw_definition": 1, "lash_line": -1,
}
ORIGINAL_15 = ["canthal_tilt", "eye_spacing", "fwhr", "midface_ratio", "lower_third", "chin_philtrum",
               "nose_intercanthal", "mouth_nose", "jaw_cheek", "lip_ratio", "eye_aspect", "brow_eye",
               "face_lw", "symmetry", "jaw_angle"]


def design(df: pd.DataFrame, feats, mean, std, pitch_med) -> np.ndarray:
    z = (df[feats].values - mean) / std
    z = np.clip(z, -4, 4)
    nuisance = np.stack([np.abs(df.yaw.values), np.abs(df.pitch.values - pitch_med)], axis=1)
    return np.hstack([z, z ** 2 - 1, nuisance])


def constrained_design(X: np.ndarray, sign: np.ndarray, k: int) -> np.ndarray:
    """Re-express X so that a non-negative fit enforces the sign constraints. Nuisance columns
    are duplicated with both signs so they stay unconstrained."""
    nuis = X[:, 2 * k:]
    return np.hstack([X[:, :k] * sign, -X[:, k:2 * k], nuis, -nuis])


def report(y, p) -> dict:
    return {"pearson": float(pearsonr(y, p)[0]), "mae": float(np.mean(np.abs(y - p))),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}


def fit_constrained(tr: pd.DataFrame, te: pd.DataFrame, feats: list, oof: bool = False) -> dict:
    """Sign-constrained ridge on `feats` for one sex. Returns the fitted pieces + predictions."""
    k = len(feats)
    mean, std = tr[feats].mean().values, tr[feats].std().values + 1e-12
    pitch_med = float(tr.pitch.median())
    Xtr, Xte = design(tr, feats, mean, std, pitch_med), design(te, feats, mean, std, pitch_med)
    corr = np.array([pearsonr(tr[m], tr.score)[0] if tr[m].std() > 0 else 0.0 for m in feats])
    sign = np.array([PRIOR_SIGNS.get(m, 1.0 if c >= 0 else -1.0) for m, c in zip(feats, corr)])
    Ctr, Cte = constrained_design(Xtr, sign, k), constrained_design(Xte, sign, k)
    cv = KFold(5, shuffle=True, random_state=0)
    search = GridSearchCV(Ridge(positive=True), {"alpha": ALPHAS}, cv=cv,
                          scoring="neg_mean_squared_error").fit(Ctr, tr.score)
    model = search.best_estimator_
    out = {"model": model, "mean": mean, "std": std, "sign": sign, "corr": corr,
           "prior_conflicts": [m for m, c in zip(feats, corr) if m in PRIOR_SIGNS and np.sign(c) != PRIOR_SIGNS[m]], "Xtr": Xtr, "Xte": Xte,
           "pte": model.predict(Cte), "lin": model.coef_[:k] * sign, "quad": -model.coef_[k:2 * k]}
    if oof:
        out["oof"] = cross_val_predict(Ridge(positive=True, alpha=model.alpha), Ctr, tr.score, cv=cv)
    return out


def ablate(train: pd.DataFrame, test: pd.DataFrame, feats: list) -> float:
    """Test Pearson r of the constrained model fitted per sex on `feats` only."""
    parts = []
    for sex in ("M", "F"):
        tr, te = train[train.sex == sex], test[test.sex == sex]
        parts.append(pd.DataFrame({"y": te.score.values, "p": fit_constrained(tr, te, feats)["pte"]}))
    p = pd.concat(parts)
    return float(pearsonr(p.y, p.p)[0])


def main():
    df = pd.read_csv(OUT / "features.csv")
    groups = json.load(open(OUT / "feature_groups.json"))
    feats = [f for g in groups.values() for f in g]
    group_of = {f: g for g, fs in groups.items() for f in fs}
    train, test = df[df.split == "train"], df[df.split == "test"]
    k = len(feats)
    app_model = {"metrics": feats, "groups": groups, "sexes": {}}
    preds, results, free_parts = [], {"n_features": k, "groups": {g: len(v) for g, v in groups.items()}}, []

    for sex in ("M", "F"):
        tr, te = train[train.sex == sex], test[test.sex == sex]
        f = fit_constrained(tr, te, feats, oof=True)
        free = RidgeCV(alphas=ALPHAS).fit(f["Xtr"], tr.score)
        free_parts.append(pd.DataFrame({"y": te.score.values, "p": free.predict(f["Xte"])}))
        results[f"{sex}_free"] = report(te.score.values, free_parts[-1].p.values)
        preds += [pd.DataFrame({"name": tr.name, "geo": f["oof"]}), pd.DataFrame({"name": te.name, "geo": f["pte"]})]
        results[sex] = report(te.score.values, f["pte"])

        lin, quad, mean, std = f["lin"], f["quad"], f["mean"], f["std"]
        # Per-feature contribution c_i(z) = lin*z + quad*(z^2-1); its spread on the training set
        # is used to turn a contribution into a 0-10 feature rating in the UI, and is the
        # feature's "importance" (how many rating points it moves a typical face).
        z_tr = np.clip((tr[feats].values - mean) / std, -4, 4)
        contrib = lin * z_tr + quad * (z_tr ** 2 - 1)
        top = tr.score >= tr.score.quantile(0.8)
        app_model["sexes"][sex] = {
            "mean": mean.tolist(), "std": std.tolist(),
            "lin": lin.tolist(), "quad": quad.tolist(),
            "intercept": float(f["model"].intercept_),
            "contribStd": contrib.std(axis=0).tolist(),
            # Average feature value among the top-20% rated faces: shown as the "ideal" target.
            "topQuintile": tr.loc[top, feats].mean().values.tolist(),
            "corr": f["corr"].tolist(),
            "alpha": float(f["model"].alpha),
        }
        imp = contrib.std(axis=0)
        results[sex].update({
            "prior_conflicts": f["prior_conflicts"],
            "alpha": float(f["model"].alpha),
            "active_metrics": [m for m, a, b in zip(feats, lin, quad) if a != 0 or b != 0],
            "metric_corr": {m: float(c) for m, c in zip(feats, f["corr"])},
            "weights": {m: {"lin": float(a), "quad": float(b), "importance": float(i)}
                        for m, a, b, i in zip(feats, lin, quad, imp)},
            "group_importance": {g: float(np.sqrt(sum(imp[feats.index(m)] ** 2 for m in fs)))
                                 for g, fs in groups.items()},
        })

    preds = pd.concat(preds)
    preds.to_csv(OUT / "geo_preds.csv", index=False)
    merged = test.merge(preds, on="name")
    results["all"] = report(merged.score.values, merged.geo.values)
    fp = pd.concat(free_parts)
    results["all_free"] = report(fp.y.values, fp.p.values)

    print("ablations…", flush=True)
    results["ablation"] = {
        "original_15": ablate(train, test, ORIGINAL_15),
        "only": {g: ablate(train, test, fs) for g, fs in groups.items()},
        "without": {g: ablate(train, test, [m for m in feats if group_of[m] != g]) for g in groups},
    }
    json.dump(results, open(OUT / "results_geometric.json", "w"), indent=2)
    json.dump(app_model, open(APP_MODELS / "geometric.json", "w"))
    print(f"{k} features")
    for key in ("M_free", "F_free", "all_free", "M", "F", "all"):
        r = results[key]
        print(f"{key}: r={r['pearson']:.3f} MAE={r['mae']:.3f} RMSE={r['rmse']:.3f}")
    a = results["ablation"]
    print("prior/data sign conflicts:", {s: results[s]["prior_conflicts"] for s in ("M", "F")})
    print(f"original 15 metrics: r={a['original_15']:.3f}")
    for g in groups:
        print(f"  {g:12s} only r={a['only'][g]:.3f}   all-but r={a['without'][g]:.3f}")


if __name__ == "__main__":
    main()
