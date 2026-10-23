"""Per-subgroup accuracy (Asian/Caucasian x female/male) and signed bias on the test split.

Signed bias = mean(predicted - human rating). A non-zero value means the model systematically
over- or under-rates that group relative to the 60 SCUT raters.
Outputs out/results_bias.json and writeup/figures/bias.pdf.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "out"
FIG = ROOT.parent / "writeup" / "figures"


def main():
    t = pd.read_csv(OUT / "test_preds.csv")
    rows = []
    for (race, sex), g in t.groupby(["race", "sex"]):
        rows.append({"group": f"{race} {'female' if sex == 'F' else 'male'}", "n": len(g),
                     "pearson": pearsonr(g.score, g.fused)[0], "mae": np.mean(np.abs(g.score - g.fused)),
                     "signed_bias": np.mean(g.fused - g.score),
                     "mean_human": g.score.mean(), "mean_pred": g.fused.mean()})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    json.dump(res.to_dict(orient="records"), open(OUT / "results_bias.json", "w"), indent=2)

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axes[0].bar(res.group, res.pearson, color="#4a6cf7")
    axes[0].set_ylim(0.5, 1); axes[0].set_title("Pearson r by subgroup", fontsize=9)
    axes[1].bar(res.group, res.signed_bias, color=["#e4572e" if b > 0 else "#4a6cf7" for b in res.signed_bias])
    axes[1].axhline(0, color="black", lw=0.6); axes[1].set_title("Signed bias (pred - human)", fontsize=9)
    for a in axes:
        a.tick_params(axis="x", labelsize=7, rotation=20); a.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(FIG / "bias.pdf"); fig.savefig(FIG / "bias.png", dpi=200)


if __name__ == "__main__":
    main()
