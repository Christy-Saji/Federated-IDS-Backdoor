"""Is FLAME's "rejects nobody" the paper's behaviour, or our re-admit guard?

    python -m scripts.baselines.flame_guard_ablation [--processed | --data PATH] [--seed N]

`flids/fl/aggregators/flame.py` re-admits a client that HDBSCAN marked as noise
when its nearest majority model is within
``tol = max(majority_spread, readmit_tol_mult * median_pairwise, 1e-6)``.

That guard is **ours, not Nguyen et al.'s.** It exists because sklearn's EOM
cluster selection prunes border points to noise even when every benign model is
numerically identical, which would re-create the exact G-05 failure FLAME was
reimplemented to fix. But a guard loose enough to never exclude anyone is
indistinguishable from having no clustering step at all, and a result of "FLAME
rejects nobody" would then be a statement about our code rather than about
FLAME. This ablation settles which it is.

It sweeps ``readmit_tol_mult`` and, at each setting, re-runs the zero-attacker
check: a guard tight enough to catch attackers is worthless if it also rejects
honest clients when there is no attacker in the federation at all.

Writes results/baselines/flame_guard_ablation.csv      (seed 0)
       results/baselines/flame_guard_ablation_s<N>.csv (--seed N)
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from scripts._common import BASELINES, add_data_arg, load_data

from flids.attacks.badnets import evaluate_backdoor
from flids.data.triggers import feature_stats, get_trigger
from flids.fl.server import FederatedServer

MULTS = (3.0, 2.0, 1.0, 0.5, 0.0)


def _cfg(mult, rounds, attack, seed, trigger):
    malicious = [0, 1, 2, 3] if attack else []
    return {
        "run_name": "flame_guard_ablation",
        "data": {"dataset": "processed", "labels": "multiclass",
                 "partition": "dirichlet", "alpha": 0.5, "n_clients": 10,
                 "min_size": 50},
        "model": {"arch": "mlp", "hidden": [256, 128, 64], "dropout": 0.3},
        "federated": {"rounds": rounds, "local_epochs": 2, "lr": 0.05,
                      "batch_size": 256, "aggregator": "flame",
                      "readmit_tol_mult": mult},
        "attack": {"enabled": attack, "type": "badnets" if attack else None,
                   "malicious_clients": malicious,
                   "trigger": ({"name": trigger, "target_label": 0,
                                "source_class": None} if attack else None),
                   "poison_ratio": 0.5, "attack_window": [1, rounds]},
        "seed": seed,
    }


def _run(ds, mult, rounds, attack, seed, trigger, stats, spec):
    server = FederatedServer(ds, _cfg(mult, rounds, attack, seed, trigger), seed=seed)
    malicious = np.array([c.malicious for c in server.clients])
    removed, aucs = [], []

    def hook(rec):
        removed.append(list(rec.get("removed_clients") or []))
        if "detection_auc" in rec:
            aucs.append(rec["detection_auc"])

    params, history = server.run(on_round=hook)
    model = server._new_model()
    model.set_params(params)
    asr = float("nan")
    if attack:
        asr = evaluate_backdoor(model, ds.X_test, ds.y_test,
                                n_classes=ds.n_classes, target_label=0,
                                spec=spec, stats=stats,
                                feature_names=ds.feature_names)
    return {
        "accuracy": history[-1]["accuracy"],
        "macro_f1": history[-1]["macro_f1"],
        "asr": asr,
        "detection_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "malicious_rejections": sum(1 for r in removed for c in r if malicious[c]),
        "honest_rejections": sum(1 for r in removed for c in r if not malicious[c]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_arg(ap)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trigger", default="oob_999")
    args = ap.parse_args()

    ds = load_data(args.data, seed=args.seed, processed=args.processed,
                   subsample=args.subsample)
    stats = feature_stats(ds.X_train, ds.y_train)
    spec = get_trigger(args.trigger)

    rows = []
    print(f"{'tol_mult':>9s} {'acc':>7s} {'mF1':>7s} {'ASR':>6s} {'detAUC':>7s} "
          f"{'mal-rej':>8s} {'hon-rej':>8s} {'zero-attacker':>16s}")
    print("-" * 74)
    for mult in MULTS:
        under_attack = _run(ds, mult, args.rounds, True, args.seed,
                            args.trigger, stats, spec)
        clean = _run(ds, mult, args.rounds, False, args.seed,
                     args.trigger, stats, spec)
        zero_ok = clean["honest_rejections"] == 0
        rows.append({"readmit_tol_mult": mult,
                     "accuracy": round(under_attack["accuracy"], 4),
                     "macro_f1": round(under_attack["macro_f1"], 4),
                     "asr": round(under_attack["asr"], 4),
                     "detection_auc": round(under_attack["detection_auc"], 4),
                     "malicious_rejections": under_attack["malicious_rejections"],
                     "honest_rejections": under_attack["honest_rejections"],
                     "zero_attacker_honest_rejections": clean["honest_rejections"],
                     "zero_attacker": "PASS" if zero_ok else "FAIL"})
        print(f"{mult:9.1f} {under_attack['accuracy']:7.4f} "
              f"{under_attack['macro_f1']:7.4f} {under_attack['asr']:6.3f} "
              f"{under_attack['detection_auc']:7.3f} "
              f"{under_attack['malicious_rejections']:8d} "
              f"{under_attack['honest_rejections']:8d} "
              f"{('PASS' if zero_ok else 'FAIL (' + str(clean['honest_rejections']) + ')'):>16s}")

    # seed 0 keeps the unsuffixed name; a sweep writes alongside it, not over it
    out = os.path.join(BASELINES, "flame_guard_ablation.csv" if args.seed == 0
                       else f"flame_guard_ablation_s{args.seed}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    passing = [r for r in rows if r["zero_attacker"] == "PASS"]
    aucs = [r["detection_auc"] for r in rows]
    print(f"\nsettings passing the zero-attacker check: "
          f"{[r['readmit_tol_mult'] for r in passing] or 'none'}")
    print(f"detection AUC across the whole sweep: "
          f"{min(aucs):.3f}-{max(aucs):.3f} (spread {max(aucs) - min(aucs):.3f})")
    print("A flat AUC across the sweep means the guard is not what is hiding the "
          "attackers - the score itself is the problem, not the threshold.")


if __name__ == "__main__":
    main()
