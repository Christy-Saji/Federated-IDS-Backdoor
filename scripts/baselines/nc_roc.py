"""Task 2.4 step 4 - Neural Cleanse ROC/AUC across clean and backdoored models.

    python -m scripts.baselines.nc_roc [--processed | --data PATH] [--n 6]

Reports AUC, plus TPR/FPR at the calibrated threshold (from nc_null.csv if
present, else the paper default of 2). Per the plan: if NC does not separate on
tabular data that is a legitimate, citable finding - reported here as a number,
not a broken run (cf. CatBack, NDSS 2026).

Writes results/baselines/nc_roc.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from scripts._common import BASELINES, CALIBRATION, add_data_arg, backdoor_attack, load_data, train_model

from flids.eval.metrics import _auc
from flids.defenses.neural_cleanse import neural_cleanse


def _score(model, ds, seed, steps, subset):
    nc = neural_cleanse(model, ds.X_train, ds.y_train, ds.n_classes,
                        subset=subset, seed=seed, steps=steps)
    ai = np.asarray(nc["anomaly_index"], float)
    l1 = np.asarray(nc["l1_norms"], float)
    below = l1 < np.median(l1)
    return float(ai[below].max()) if below.any() else 0.0


def _calibrated_threshold():
    path = os.path.join(CALIBRATION, "nc_null.csv")
    if os.path.exists(path):
        for line in open(path):
            if line.startswith("# threshold_p95"):
                return float(line.strip().split(",")[1]), "calibrated_p95"
    return 2.0, "paper_default"


def main():
    ap = argparse.ArgumentParser()
    add_data_arg(ap)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--subset", type=int, default=3000)
    ap.add_argument("--trigger", default="oob_999")
    args = ap.parse_args()

    os.makedirs(BASELINES, exist_ok=True)
    scores, labels, rows = [], [], []
    for seed in range(args.n):
        ds = load_data(args.data, seed=seed, processed=args.processed,
                       subsample=args.subsample)
        clean = train_model(ds, seed=seed, rounds=args.rounds)
        s_clean = _score(clean, ds, seed, args.steps, args.subset)

        atk = backdoor_attack(trigger=args.trigger, rounds=args.rounds)
        bd = train_model(ds, seed=seed, rounds=args.rounds, attack=atk)
        s_bd = _score(bd, ds, seed, args.steps, args.subset)

        for tag, sc, lab in (("clean", s_clean, 0), ("backdoor", s_bd, 1)):
            scores.append(sc); labels.append(lab)
            rows.append({"seed": seed, "model": tag, "max_anomaly_index": round(sc, 4)})
        print(f"seed {seed}: clean={s_clean:.2f}  backdoor={s_bd:.2f}")

    thr, thr_kind = _calibrated_threshold()
    scores = np.array(scores); labels = np.array(labels)
    auc = _auc(scores, labels)
    flagged = scores > thr
    tpr = float(flagged[labels == 1].mean())
    fpr = float(flagged[labels == 0].mean())

    out = os.path.join(BASELINES, "nc_roc.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "model", "max_anomaly_index"])
        w.writeheader(); w.writerows(rows)
        f.write(f"# auc,{auc:.4f}\n# threshold,{thr:.4f} ({thr_kind})\n")
        f.write(f"# tpr,{tpr:.4f}\n# fpr,{fpr:.4f}\n")
    print(f"\nwrote {out}")
    print(f"AUC={auc:.3f}  TPR={tpr:.2f}  FPR={fpr:.2f}  @ threshold {thr:.2f} ({thr_kind})")
    if auc < 0.7:
        print("NOTE: weak separation - consistent with CatBack's tabular finding. "
              "Report as a result once tuning (Task 2.4) is exhausted.")


if __name__ == "__main__":
    main()
