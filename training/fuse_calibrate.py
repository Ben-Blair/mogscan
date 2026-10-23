"""Fit the CNN + geometry fusion on the validation set, evaluate everything on the official
40% test split, and write the calibration the browser uses for PSL mapping.

Outputs:
  ../app/public/models/fusion.json
  out/results.json            headline numbers used by the write-up and slides
  out/test_preds.csv          per-image test predictions (used by eval_bias.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent
OUT = ROOT / "out"
APP_MODELS = ROOT.parent / "app" / "public" / "models"


def report(y, p):
    return {"pearson": float(pearsonr(y, p)[0]), "mae": float(np.mean(np.abs(y - p))),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}


def main():
    feats = pd.read_csv(OUT / "features.csv")
    df = feats.merge(pd.read_csv(OUT / "geo_preds.csv"), on="name").merge(pd.read_csv(OUT / "cnn_preds.csv"), on="name")
    val, test = df[df.set == "val"], df[df.set == "test"]

    lr = LinearRegression().fit(val[["cnn", "geo"]], val.score)
    test = test.assign(fused=lr.predict(test[["cnn", "geo"]]))
    train = feats[feats.split == "train"]
    fusion = {"wCnn": float(lr.coef_[0]), "wGeo": float(lr.coef_[1]), "bias": float(lr.intercept_),
              "scoreMean": float(train.score.mean()), "scoreStd": float(train.score.std())}
    json.dump(fusion, open(APP_MODELS / "fusion.json", "w"), indent=2)

    psl = np.clip(4 + (test.fused - fusion["scoreMean"]) / fusion["scoreStd"], 1, 8)
    psl_true = np.clip(4 + (test.score - fusion["scoreMean"]) / fusion["scoreStd"], 1, 8)
    test = test.assign(psl=psl, psl_true=psl_true)
    test[["name", "score", "sex", "race", "cnn", "geo", "fused", "p_male", "psl", "psl_true"]].to_csv(
        OUT / "test_preds.csv", index=False)

    results = {
        "n_test": int(len(test)),
        "geometric": report(test.score, test.geo),
        "cnn": report(test.score, test.cnn),
        "fused": report(test.score, test.fused),
        "fusion": fusion,
        "sex_accuracy": float(np.mean((test.p_male > 0.5) == (test.sex == "M"))),
        "psl_mae": float(np.mean(np.abs(psl - psl_true))),
        "tier_within_one": None,
    }
    cuts = np.array([2, 3, 3.75, 4.5, 5.25, 6, 7])
    tp, tt = np.searchsorted(cuts, psl, side="right"), np.searchsorted(cuts, psl_true, side="right")
    results["tier_exact"] = float(np.mean(tp == tt))
    results["tier_within_one"] = float(np.mean(np.abs(tp - tt) <= 1))
    json.dump(results, open(OUT / "results.json", "w"), indent=2)
    for k in ("geometric", "cnn", "fused"):
        r = results[k]
        print(f"{k:10s} r={r['pearson']:.4f} MAE={r['mae']:.4f} RMSE={r['rmse']:.4f}")
    print(f"fusion: {fusion}")
    print(f"sex acc {results['sex_accuracy']:.4f}  PSL MAE {results['psl_mae']:.3f}  "
          f"tier exact {results['tier_exact']:.3f}  within one {results['tier_within_one']:.3f}")


if __name__ == "__main__":
    main()
