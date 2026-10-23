"""Summarise the Google Form export (study/responses.csv) into tables and figures.

Columns are found by the bracketed codes from survey.md, e.g. "[SUS1] I think that ...".
Outputs: out/survey_summary.json, out/open_ended.md, ../writeup/figures/survey_likert.pdf
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE.parent / "writeup" / "figures"
LIKERT = ["ACC", "STABLE", "EXPLAIN", "FEATURES", "GROUP", "PRIVACY", "BETTER", "FEELBAD"]
LIKERT_LABELS = {"ACC": "Score felt accurate", "STABLE": "Score was stable", "EXPLAIN": "Understood why",
                 "FEATURES": "Feature list useful", "GROUP": "Group scan fun", "PRIVACY": "On-device reassuring",
                 "BETTER": "Better than other apps", "FEELBAD": "Felt worse about looks"}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        m = re.match(r"\s*\[([A-Z0-9]+)\]", c)
        if m:
            rename[c] = m.group(1)
    return df.rename(columns=rename)


def sus(df: pd.DataFrame) -> pd.Series:
    odd = sum(df[f"SUS{i}"] - 1 for i in (1, 3, 5, 7, 9))
    even = sum(5 - df[f"SUS{i}"] for i in (2, 4, 6, 8, 10))
    return (odd + even) * 2.5


def main(path=HERE / "responses.csv"):
    df = load(Path(path))
    n = len(df)
    out = {"n": n}
    s = sus(df)
    out["sus"] = {"mean": float(s.mean()), "sd": float(s.std()), "min": float(s.min()), "max": float(s.max())}
    out["likert"] = {}
    for k in LIKERT:
        if k in df and df[k].notna().any():
            v = pd.to_numeric(df[k], errors="coerce").dropna()
            out["likert"][k] = {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()),
                                "agree_pct": float((v >= 4).mean() * 100)}
    nps = pd.to_numeric(df["NPS"], errors="coerce").dropna()
    out["nps"] = {"score": float(((nps >= 9).mean() - (nps <= 6).mean()) * 100), "mean": float(nps.mean()),
                  "promoters": int((nps >= 9).sum()), "passives": int(((nps >= 7) & (nps <= 8)).sum()),
                  "detractors": int((nps <= 6).sum())}
    for k in ("PAY", "TONE", "PRIOR", "AGE"):
        if k in df:
            out[k.lower()] = df[k].value_counts().to_dict()
    (HERE / "out").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "out" / "survey_summary.json", "w"), indent=2)

    with open(HERE / "out" / "open_ended.md", "w") as f:
        for k, title in (("LIKE", "Liked most"), ("CHANGE", "Would change"), ("USE", "Would use when")):
            if k in df:
                f.write(f"## {title}\n\n")
                for pid, txt in zip(df.get("PID", range(n)), df[k].fillna("")):
                    if str(txt).strip():
                        f.write(f"- **{pid}:** {str(txt).strip()}\n")
                f.write("\n")

    FIG.mkdir(parents=True, exist_ok=True)
    keys = list(out["likert"].keys())
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(keys) + 0.8))
    colors = ["#d7301f", "#fc8d59", "#e0e0e0", "#91bfdb", "#4575b4"]
    for row, k in enumerate(keys):
        v = pd.to_numeric(df[k], errors="coerce").dropna()
        counts = np.array([(v == i).sum() for i in range(1, 6)]) / len(v) * 100
        left = -counts[:2].sum() - counts[2] / 2
        for i, c in enumerate(counts):
            ax.barh(row, c, left=left, color=colors[i], edgecolor="white", label=str(i + 1) if row == 0 else None)
            left += c
    ax.set_yticks(range(len(keys)), [f"{LIKERT_LABELS[k]} (n={out['likert'][k]['n']})" for k in keys], fontsize=8)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("% of participants (1 = strongly disagree … 5 = strongly agree)", fontsize=8)
    ax.invert_yaxis(); ax.legend(ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "survey_likert.pdf"); fig.savefig(FIG / "survey_likert.png", dpi=200)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
