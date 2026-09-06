"""Task 2.2 - clean-model ASR baseline for every trigger rung.

    python -m scripts.baselines.clean_asr [--processed | --data PATH] [--rounds 20] [--seeds 0 1 2]

For each rung x seed, train a *clean* FedAvg model and measure the trigger's ASR
on it. A trigger that is out of distribution fires on a clean model too; only
`dASR = ASR_model - ASR_clean` is attributable to the backdoor (G-01).

Writes results/baselines/clean_asr.csv  (trigger,seed,asr_clean)
"""

from __future__ import annotations

import argparse
import csv
import os

from scripts._common import BASELINES, add_data_arg, load_data

from flids.attacks.badnets import evaluate_backdoor
from flids.data.triggers import TRIGGERS, feature_stats, get_trigger
from flids.fl.server import FederatedServer

RUNGS = [name for name, s in TRIGGERS.items() if s["value"] != "from_pcap"]


def clean_cfg(rounds, seed):
    return {
        "run_name": "clean_asr_baseline",
        "data": {"dataset": "synthetic", "labels": "multiclass",
                 "partition": "dirichlet", "alpha": 0.5, "n_clients": 10,
                 "min_size": 100},
        "model": {"arch": "mlp", "hidden": [256, 128, 64], "dropout": 0.3},
        "federated": {"rounds": rounds, "local_epochs": 2, "lr": 0.05,
                      "batch_size": 256, "aggregator": "fedavg"},
        "attack": {"enabled": False, "malicious_clients": [], "attack_window": [1, rounds]},
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    add_data_arg(ap)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--target-label", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(BASELINES, exist_ok=True)
    rows = []
    for seed in args.seeds:
        ds = load_data(args.data, seed=seed, processed=args.processed,
                       subsample=args.subsample)
        cfg = clean_cfg(args.rounds, seed)
        server = FederatedServer(ds, cfg, seed=seed)
        params, _ = server.run()
        model = server._new_model()
        model.set_params(params)
        stats = feature_stats(ds.X_train, ds.y_train)

        for rung in RUNGS:
            asr = evaluate_backdoor(
                model, ds.X_test, ds.y_test, n_classes=ds.n_classes,
                target_label=args.target_label,
                spec=get_trigger(rung), stats=stats,
                feature_names=ds.feature_names)
            rows.append({"trigger": rung, "seed": seed, "asr_clean": round(asr, 6)})
            print(f"{rung:14s} seed={seed}  asr_clean={asr:.4f}")

    out = os.path.join(BASELINES, "clean_asr.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trigger", "seed", "asr_clean"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
