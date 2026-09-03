"""Tasks 2.6-2.9 - per-round detection AUC + FPR for every aggregator.

    python -m scripts.baselines.detection_auc [--data PATH] [--trigger oob_999]

Runs the same attack under fedavg / gradnorm_scorer / fltrust / flame /
fltrust+flame, holding seed and partition fixed, and reports each aggregator's
per-client score as a malicious-client classifier (AUC) plus its honest-client
FPR. Binary "removed: True/False" is deliberately not the headline - AUC over
the continuous score is.

Writes results/baselines/detection_auc.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from scripts._common import BASELINES, _cfg, backdoor_attack

from flids.data.loaders import load_dataset, synthetic_dataset
from flids.eval.metrics import defense_fpr
from flids.fl.server import FederatedServer

AGGREGATORS = ["fedavg", "gradnorm_scorer", "fltrust", "flame", "fltrust+flame"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--trigger", default="oob_999")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(BASELINES, exist_ok=True)
    atk = backdoor_attack(trigger=args.trigger, rounds=args.rounds)
    malicious = atk["malicious_clients"]
    rows = []

    for agg in AGGREGATORS:
        ds = (load_dataset(args.data, seed=args.seed, multiclass=True) if args.data
              else synthetic_dataset(seed=args.seed, multiclass=True, n_classes=8))
        cfg = _cfg(args.seed, args.rounds, aggregator=agg, attack=dict(atk))
        server = FederatedServer(ds, cfg, seed=args.seed)
        _, history = server.run()

        aucs = [h["detection_auc"] for h in history if "detection_auc" in h]
        n_clients = cfg["data"]["n_clients"]
        fprs = [defense_fpr(h["removed_clients"], malicious, n_clients)
                for h in history if h.get("removed_clients") is not None]
        row = {"aggregator": agg,
               "detection_auc_mean": round(float(np.nanmean(aucs)), 4) if aucs else "",
               "detection_auc_last": round(aucs[-1], 4) if aucs else "",
               "fpr_mean": round(float(np.nanmean(fprs)), 4) if fprs else 0.0}
        rows.append(row)
        print(f"{agg:16s} AUC_mean={row['detection_auc_mean']}  FPR_mean={row['fpr_mean']}")

    out = os.path.join(BASELINES, "detection_auc.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
