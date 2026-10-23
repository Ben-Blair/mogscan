"""Explainable geometric model: per-sex ridge regression on frontalized face metrics.

Each metric enters as a standardised linear term z and a centred quadratic term (z^2 - 1),
so the model can learn "ideal ranges" (inverted-U) as well as monotonic effects. Head pose
(|yaw|, |pitch - median|) is included as a nuisance covariate and is zeroed at inference,
because the live app only scores near-frontal faces.

Two variants are fitted:
  * free:        ordinary ridge. Reported for comparison only.
  * constrained: each linear weight must agree in sign with the metric's marginal correlation
                 with the rating, and quadratic weights must be <= 0 (an "ideal range").
                 Correlated metrics otherwise produce sign flips, e.g. positive canthal tilt
                 being reported as a weakness. This variant is shipped to the app.

Outputs:
  out/geo_preds.csv                     out-of-fold train predictions + test predictions
  out/results_geometric.json            metrics + per-metric correlations
  ../app/public/models/geometric.json   everything the browser needs to score + explain
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from mogscan_metrics import METRIC_NAMES

ROOT = Path(__file__).parent
OUT = ROOT / "out"
APP_MODELS = ROOT.parent / "app" / "public" / "models"
ALPHAS = np.logspace(-2, 3, 30)


def design(df: pd.DataFrame, mean, std, pitch_med) -> np.ndarray:
    z = (df[METRIC_NAMES].values - mean) / std
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


def _stats(tr: pd.DataFrame):
    return tr[METRIC_NAMES].mean().values, tr[METRIC_NAMES].std().values, float(tr.pitch.median())


def main():
    df = pd.read_csv(OUT / "features.csv")
    train, test = df[df.split == "train"], df[df.split == "test"]
    k = len(METRIC_NAMES)
    app_model, preds, results = {"metrics": METRIC_NAMES, "sexes": {}}, [], {}

    for sex in ("M", "F"):
        tr, te = train[train.sex == sex], test[test.sex == sex]
        mean, std = tr[METRIC_NAMES].mean().values, tr[METRIC_NAMES].std().values
        pitch_med = float(tr.pitch.median())
        Xtr, Xte = design(tr, mean, std, pitch_med), design(te, mean, std, pitch_med)
        free = RidgeCV(alphas=ALPHAS).fit(Xtr, tr.score)
        results[f"{sex}_free"] = report(te.score.values, free.predict(Xte))

        sign = np.sign([pearsonr(tr[m], tr.score)[0] for m in METRIC_NAMES])
        Ctr, Cte = constrained_design(Xtr, sign, k), constrained_design(Xte, sign, k)
        cv = KFold(5, shuffle=True, random_state=0)
        search = GridSearchCV(Ridge(positive=True), {"alpha": ALPHAS}, cv=cv,
                              scoring="neg_mean_squared_error").fit(Ctr, tr.score)
        model = search.best_estimator_
        oof = cross_val_predict(Ridge(positive=True, alpha=model.alpha), Ctr, tr.score, cv=cv)
        pte = model.predict(Cte)
        preds += [pd.DataFrame({"name": tr.name, "geo": oof}), pd.DataFrame({"name": te.name, "geo": pte})]
        results[sex] = report(te.score.values, pte)

        lin, quad = model.coef_[:k] * sign, -model.coef_[k:2 * k]
        # Per-metric contribution c_i(z) = lin*z + quad*(z^2-1); its spread on the training set
        # is used to turn a contribution into a 0-10 feature rating in the UI.
        z_tr = np.clip((tr[METRIC_NAMES].values - mean) / std, -4, 4)
        contrib = lin * z_tr + quad * (z_tr ** 2 - 1)
        top = tr.score >= tr.score.quantile(0.8)
        app_model["sexes"][sex] = {
            "mean": mean.tolist(), "std": std.tolist(),
            "lin": lin.tolist(), "quad": quad.tolist(),
            "intercept": float(model.intercept_),
            "contribStd": contrib.std(axis=0).tolist(),
            # Average metric value among the top-20% rated faces: shown as the "ideal" target.
            "topQuintile": tr.loc[top, METRIC_NAMES].mean().values.tolist(),
            "alpha": float(model.alpha),
        }
        results[sex]["alpha"] = float(model.alpha)
        results[sex]["active_metrics"] = [m for m, a, b in zip(METRIC_NAMES, lin, quad) if a != 0 or b != 0]
        results[sex]["metric_corr"] = {m: float(pearsonr(tr[m], tr.score)[0]) for m in METRIC_NAMES}

    preds = pd.concat(preds)
    preds.to_csv(OUT / "geo_preds.csv", index=False)
    merged = test.merge(preds, on="name")
    results["all"] = report(merged.score.values, merged.geo.values)
    free_te = pd.concat([test[test.sex == s_].assign(p=RidgeCV(alphas=ALPHAS).fit(
        design(train[train.sex == s_], *_stats(train[train.sex == s_])), train[train.sex == s_].score).predict(
        design(test[test.sex == s_], *_stats(train[train.sex == s_])))) for s_ in ("M", "F")])
    results["all_free"] = report(free_te.score.values, free_te.p.values)
    json.dump(results, open(OUT / "results_geometric.json", "w"), indent=2)
    json.dump(app_model, open(APP_MODELS / "geometric.json", "w"))
    for key in ("M_free", "F_free", "all_free", "M", "F", "all"):
        r = results[key]
        print(f"{key}: r={r['pearson']:.3f} MAE={r['mae']:.3f} RMSE={r['rmse']:.3f}")


if __name__ == "__main__":
    main()
