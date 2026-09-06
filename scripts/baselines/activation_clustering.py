"""Task 2.5 - Activation Clustering TPR/FPR across poison ratios.

    python -m scripts.baselines.activation_clustering [--processed | --data PATH]

Threshold is calibrated on clean models (not the paper default). Runs at poison
ratios 0/1/5/10/30% ; 0% gives the false-positive rate.

Writes results/baselines/activation_clustering.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from scripts._common import BASELINES, add_data_arg, backdoor_attack, load_data, train_model

from flids.defenses.activation_clustering import (activation_clustering,
                                                  calibrate_ac_threshold,
                                                  penultimate_for_class)


def main():
    ap = argparse.ArgumentParser()
    add_data_arg(ap)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--target-label", type=int, default=0)
    ap.add_argument("--ratios", type=float, nargs="+",
                    default=[0.0, 0.01, 0.05, 0.10, 0.30])
    ap.add_argument("--n-clean", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(BASELINES, exist_ok=True)
    t = args.target_label

    clean_models = [train_model(load_data(args.data, seed=s, processed=args.processed,
                                    subsample=args.subsample), seed=s,
                                rounds=args.rounds)
                    for s in range(args.n_clean)]
    ds0 = load_data(args.data, seed=0, processed=args.processed,
                    subsample=args.subsample)
    thr, nulls = calibrate_ac_threshold(clean_models, ds0.X_train, ds0.y_train, t)
    print(f"calibrated threshold (p95 of clean silhouettes) = {thr:.3f}  "
          f"(paper default ~0.12)")

    rows = []
    for ratio in args.ratios:
        flags, sils = [], []
        for seed in range(3):
            ds = load_data(args.data, seed=seed, processed=args.processed,
                           subsample=args.subsample)
            if ratio == 0.0:
                model = train_model(ds, seed=seed, rounds=args.rounds)
            else:
                atk = backdoor_attack(target_label=t, poison_ratio=ratio,
                                      rounds=args.rounds)
                model = train_model(ds, seed=seed, rounds=args.rounds, attack=atk)
            acts = penultimate_for_class(model, ds.X_train, ds.y_train, t)
            res = activation_clustering(acts, threshold=thr)
            flags.append(res["flagged"]); sils.append(res["silhouette"])
        rate = float(np.mean(flags))
        row = {"poison_ratio": ratio, "silhouette_mean": round(np.mean(sils), 4),
               "flag_rate": rate,
               "tpr": rate if ratio > 0 else "",
               "fpr": rate if ratio == 0 else ""}
        rows.append(row)
        print(f"ratio={ratio:>5.2f}  silhouette={np.mean(sils):.3f}  flag_rate={rate:.2f}")

    out = os.path.join(BASELINES, "activation_clustering.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
        f.write(f"# calibrated_threshold,{thr:.4f}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
