"""Task 2.4 step 4 - Neural Cleanse ROC/AUC across clean and backdoored models.

    python -m scripts.baselines.nc_roc [--processed | --data PATH] [--n 10] [--trigger oob_999]

Reports AUC, plus TPR/FPR at the calibrated threshold (from nc_null.csv if
present, else the paper default of 2). Per the plan: if NC does not separate on
tabular data that is a legitimate, citable finding - reported here as a number,
not a broken run (cf. CatBack, NDSS 2026).

What makes the number trustworthy, each of which an earlier version lacked:

* **Out-of-sample seeds.** Models here use seeds ``--seed-offset`` (100) and up;
  ``nc_calibrate`` sets its threshold on seeds 0..N-1. The old version scored
  the calibration models themselves, so its FPR was in-sample.
* **The backdoor is verified, not assumed.** Every model's ASR is recorded; a
  "backdoored" model whose trigger does not fire is not a test of the detector.
* **The attack is the campaign's** (``source_class: null``, ``poison_ratio:
  0.5``, as in ``configs/*_real.yaml``), not the script helper's binary-era
  ``source_class=2`` / 0.3.
* **Which class NC names** is recorded. A high anomaly index on a class that is
  not the target has not found the backdoor.

On ``oob_999`` NC is structurally handicapped: the pattern is clamped to the
observed feature range (the plan requires it, or every class looks backdoored),
so the true 999.0 stamp is a value NC is forbidden to express. Run the
in-bounds rungs (``--trigger inbounds_free``) for the test NC can actually pass.

Writes results/baselines/nc_roc.csv            (oob_999)
       results/baselines/nc_roc_<trigger>.csv  (any other rung)
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from scripts._common import BASELINES, CALIBRATION, add_data_arg, backdoor_attack, load_data, train_model

from flids.attacks.badnets import evaluate_backdoor
from flids.data.labels import CLASS_NAMES
from flids.data.triggers import feature_stats, get_trigger
from flids.eval.metrics import _auc
from flids.defenses.neural_cleanse import flagged_class, neural_cleanse

TARGET = 0


def _score(model, ds, seed, steps, subset):
    nc = neural_cleanse(model, ds.X_train, ds.y_train, ds.n_classes,
                        subset=subset, seed=seed, steps=steps)
    return flagged_class(nc)


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
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=100,
                    help="first seed; keep it clear of nc_calibrate's 0..N-1")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--subset", type=int, default=3000)
    ap.add_argument("--trigger", default="oob_999")
    args = ap.parse_args()

    os.makedirs(BASELINES, exist_ok=True)
    spec = get_trigger(args.trigger)
    scores, labels, rows = [], [], []
    for seed in range(args.seed_offset, args.seed_offset + args.n):
        ds = load_data(args.data, seed=seed, processed=args.processed,
                       subsample=args.subsample)
        stats = feature_stats(ds.X_train, ds.y_train)
        atk = backdoor_attack(trigger=args.trigger, target_label=TARGET,
                              source_class=None, poison_ratio=0.5,
                              rounds=args.rounds)
        for tag, lab, attack in (("clean", 0, None), ("backdoor", 1, atk)):
            model = train_model(ds, seed=seed, rounds=args.rounds, attack=attack)
            sc, cls = _score(model, ds, seed, args.steps, args.subset)
            asr = evaluate_backdoor(model, ds.X_test, ds.y_test, n_classes=ds.n_classes,
                                    target_label=TARGET, spec=spec, stats=stats,
                                    feature_names=ds.feature_names)
            scores.append(sc); labels.append(lab)
            rows.append({"seed": seed, "model": tag, "max_anomaly_index": round(sc, 4),
                         "flagged_class": CLASS_NAMES[cls] if cls >= 0 else "",
                         "names_target": cls == TARGET, "asr": round(asr, 4)})
            print(f"seed {seed} {tag:8s}: max AI={sc:6.2f}  class={rows[-1]['flagged_class']:12s}"
                  f"  ASR={asr:.3f}", flush=True)

    thr, thr_kind = _calibrated_threshold()
    scores = np.array(scores); labels = np.array(labels)
    auc = _auc(scores, labels)
    flagged = scores > thr
    tpr = float(flagged[labels == 1].mean())
    fpr = float(flagged[labels == 0].mean())
    bd = [r for r in rows if r["model"] == "backdoor"]
    cl = {r["seed"]: r for r in rows if r["model"] == "clean"}
    # implanted = the trigger fires at least 0.5 more often than on the clean
    # model of the same seed (dASR, G-01) - a raw ASR cutoff would count the
    # oob_999 seeds whose clean model already fires as "backdoored"
    dasr = [r["asr"] - cl[r["seed"]]["asr"] for r in bd]
    implanted = sum(1 for d in dasr if d >= 0.5)
    names_target_bd = sum(1 for r in bd if r["names_target"])
    names_target_clean = sum(1 for r in rows if r["model"] == "clean" and r["names_target"])

    name = "nc_roc.csv" if args.trigger == "oob_999" else f"nc_roc_{args.trigger}.csv"
    out = os.path.join(BASELINES, name)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
        f.write(f"# auc,{auc:.4f}\n# threshold,{thr:.4f} ({thr_kind})\n")
        f.write(f"# tpr,{tpr:.4f}\n# fpr,{fpr:.4f}\n")
        f.write(f"# trigger,{args.trigger}\n")
        f.write(f"# asr_backdoor_mean,{np.mean([r['asr'] for r in bd]):.4f}\n")
        f.write(f"# asr_clean_mean,{np.mean([r['asr'] for r in cl.values()]):.4f}\n")
        f.write(f"# backdoor_implanted_dasr_ge_0.5,{implanted}/{len(bd)}\n")
        f.write(f"# names_target_backdoor,{names_target_bd}/{len(bd)}\n")
        f.write(f"# names_target_clean,{names_target_clean}/{len(bd)}\n")
    print(f"\nwrote {out}")
    print(f"AUC={auc:.3f}  TPR={tpr:.2f}  FPR={fpr:.2f}  @ threshold {thr:.2f} ({thr_kind})")
    print(f"backdoor implanted (dASR>=0.5): {implanted}/{len(bd)}  "
          f"mean dASR={np.mean(dasr):.3f}   "
          f"NC names the target class: backdoored {names_target_bd}/{len(bd)}, "
          f"clean {names_target_clean}/{len(bd)}")
    if auc < 0.7:
        print("NOTE: weak separation - consistent with CatBack's tabular finding. "
              "Report as a result once tuning (Task 2.4) is exhausted.")


if __name__ == "__main__":
    main()
