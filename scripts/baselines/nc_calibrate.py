"""Task 2.4 step 3 - calibrate Neural Cleanse on clean models.

    python -m scripts.baselines.nc_calibrate [--data PATH] [--n-clean 10]

Train N independently-seeded clean models, run Neural Cleanse on each, collect
the null distribution of the max anomaly index, and set the threshold at the
95th percentile. Compare it to the paper's default of 2.

Writes results/calibration/nc_null.csv  and prints threshold_p95.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from scripts._common import CALIBRATION, load_data, train_model

from flids.defenses.neural_cleanse import neural_cleanse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--n-clean", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--subset", type=int, default=3000)
    args = ap.parse_args()

    os.makedirs(CALIBRATION, exist_ok=True)
    rows = []
    for seed in range(args.n_clean):
        ds = load_data(args.data, seed=seed)
        model = train_model(ds, seed=seed, rounds=args.rounds)
        nc = neural_cleanse(model, ds.X_train, ds.y_train, ds.n_classes,
                            subset=args.subset, seed=seed, steps=args.steps)
        ai = np.asarray(nc["anomaly_index"], float)
        l1 = np.asarray(nc["l1_norms"], float)
        # NC only flags classes whose mask is *smaller* than the median
        below = l1 < np.median(l1)
        max_ai = float(ai[below].max()) if below.any() else 0.0
        rows.append({"clean_model_seed": seed, "max_anomaly_index": round(max_ai, 4)})
        print(f"seed {seed}: max anomaly index (below-median classes) = {max_ai:.3f}")

    out = os.path.join(CALIBRATION, "nc_null.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clean_model_seed", "max_anomaly_index"])
        w.writeheader()
        w.writerows(rows)

    vals = np.array([r["max_anomaly_index"] for r in rows])
    p95 = float(np.percentile(vals, 95))
    with open(out, "a", newline="") as f:
        f.write(f"# threshold_p95,{p95:.4f}\n")
        f.write("# paper_default,2.0\n")
    print(f"\nwrote {out}")
    print(f"threshold_p95 = {p95:.3f}   (paper default = 2.0)")


if __name__ == "__main__":
    main()
