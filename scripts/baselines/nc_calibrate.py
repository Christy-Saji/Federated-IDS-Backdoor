"""Task 2.4 step 3 - calibrate Neural Cleanse on clean models.

    python -m scripts.baselines.nc_calibrate [--processed | --data PATH] [--n-clean 10]

Train N independently-seeded clean models, run Neural Cleanse on each, collect
the null distribution of the max anomaly index, and set the threshold at the
95th percentile. Compare it to the paper's default of 2.

Calibration uses seeds 0..N-1. ``nc_roc`` evaluates on seeds from 100 up, so the
clean models it scores are never the ones that set the threshold - otherwise
its FPR is in-sample.

Writes results/calibration/nc_null.csv  and prints threshold_p95.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from scripts._common import CALIBRATION, add_data_arg, load_data, train_model

from flids.data.labels import CLASS_NAMES
from flids.defenses.neural_cleanse import flagged_class, neural_cleanse


def main():
    ap = argparse.ArgumentParser()
    add_data_arg(ap)
    ap.add_argument("--n-clean", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--subset", type=int, default=3000)
    args = ap.parse_args()

    os.makedirs(CALIBRATION, exist_ok=True)
    rows = []
    for seed in range(args.n_clean):
        ds = load_data(args.data, seed=seed, processed=args.processed,
                       subsample=args.subsample)
        model = train_model(ds, seed=seed, rounds=args.rounds)
        nc = neural_cleanse(model, ds.X_train, ds.y_train, ds.n_classes,
                            subset=args.subset, seed=seed, steps=args.steps)
        max_ai, cls = flagged_class(nc)
        rows.append({"clean_model_seed": seed, "max_anomaly_index": round(max_ai, 4),
                     "flagged_class": CLASS_NAMES[cls] if cls >= 0 else ""})
        print(f"seed {seed}: max anomaly index (below-median classes) = {max_ai:.3f}"
              f"  class={rows[-1]['flagged_class']}", flush=True)

    out = os.path.join(CALIBRATION, "nc_null.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clean_model_seed", "max_anomaly_index",
                                          "flagged_class"])
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
