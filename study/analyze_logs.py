"""Score stability and performance from MogScan session logs (study/logs/*.json).

For each logged person-track we take the frames where a rating exists and the face was frontal,
and report the standard deviation of PSL over the track (lower = more stable), plus fps.
Outputs out/logs_summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def main(log_dir=HERE / "logs"):
    files = sorted(Path(log_dir).glob("*.json"))
    tracks, fps, multi = {}, [], 0
    for f in files:
        log = json.load(open(f))
        for fr in log["frames"]:
            fps.append(fr["fps"])
            rated = [p for p in fr["people"] if p["psl"] is not None and p["frontal"]]
            multi += len(rated) >= 2
            for p in rated:
                tracks.setdefault((f.name, p["id"]), []).append(p["psl"])
    # Ignore the first second of each track while the smoothing filters warm up.
    stds = [float(np.std(v[5:])) for v in tracks.values() if len(v) >= 15]
    out = {"sessions": len(files), "tracks": len(stds),
           "psl_std_median": float(np.median(stds)) if stds else None,
           "psl_std_p90": float(np.percentile(stds, 90)) if stds else None,
           "fps_median": float(np.median(fps)) if fps else None,
           "fps_p10": float(np.percentile(fps, 10)) if fps else None,
           "frames_with_2plus_rated": int(multi)}
    (HERE / "out").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "out" / "logs_summary.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
