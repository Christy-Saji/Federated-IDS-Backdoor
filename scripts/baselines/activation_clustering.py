"""Task 2.5 - Activation Clustering TPR/FPR across poison ratios.

    python -m scripts.baselines.activation_clustering [--processed | --data PATH] [--trigger oob_999]

Threshold is calibrated on clean models (not the paper default). Runs at poison
ratios 0/1/5/10/30% ; 0% gives the false-positive rate.

Task 2.5 step 1 says activations are taken over **all training samples of the
target class**. In a poisoned federation that set includes the stamped rows the
attackers relabelled to the target, and those rows are the only thing Activation
Clustering can find. An earlier version of this script clustered
``ds.X_train[y == target]`` - the *clean* global split, which contains no
poisoned row at any ratio - so its silhouette could not respond to poisoning
and the "flat at 0.352-0.354" result was a property of the input, not of the
defense. The rows now come from the clients' own training data as they trained
on it in the last attack round (``Client._training_data``), poisoned rows
included.

Two further changes keep the table honest:

* **Calibration and evaluation use disjoint seeds.** Clean models for the
  threshold are seeds ``100..``; the 0% row is evaluated on seeds ``0..``. The
  old version reused its calibration models for the FPR row, so the FPR was
  in-sample.
* **Every model reports its ASR**, so a "poisoned" row is known to hold a
  backdoor that actually works, and ``poison_in_suspect`` says whether the
  smaller cluster is the poison (it is ground truth the defender never sees -
  a diagnostic, not part of the decision).

The attack matches the campaign configs (``source_class: null``,
``configs/*_real.yaml``); ``poison_ratio`` is the fraction of each malicious
client's eligible rows, as everywhere else, and ``poisoned_share`` reports what
that becomes as a share of the target class.

Writes results/baselines/activation_clustering.csv            (oob_999)
       results/baselines/activation_clustering_<trigger>.csv  (any other rung)
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from scripts._common import _cfg, BASELINES, add_data_arg, backdoor_attack, load_data

from flids.attacks.badnets import evaluate_backdoor
from flids.data.triggers import feature_stats, get_trigger
from flids.defenses.activation_clustering import (activation_clustering,
                                                  penultimate_for_class)
from flids.eval.metrics import _auc
from flids.fl.server import FederatedServer

CALIBRATION_SEED_OFFSET = 100


def train_federation(ds, seed, rounds, attack=None):
    """Train a FedAvg model; also return the rows it trained on in the last round."""
    server = FederatedServer(ds, _cfg(seed, rounds, attack=attack), seed=seed)
    params, _ = server.run()
    model = server._new_model()
    model.set_params(params)
    last = rounds - 1
    Xs, ys, poisoned = [], [], []
    for c in server.clients:
        Xc, yc = c._training_data(last)
        Xs.append(Xc); ys.append(yc)
        # poison_split keeps row order and only relabels chosen rows, so a label
        # that moved is exactly a poisoned row
        poisoned.append(np.asarray(yc) != np.asarray(c.y))
    return model, np.vstack(Xs), np.concatenate(ys), np.concatenate(poisoned)


def main():
    ap = argparse.ArgumentParser()
    add_data_arg(ap)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--target-label", type=int, default=0)
    ap.add_argument("--trigger", default="oob_999")
    ap.add_argument("--ratios", type=float, nargs="+",
                    default=[0.0, 0.01, 0.05, 0.10, 0.30])
    ap.add_argument("--n-clean", type=int, default=10,
                    help="clean models for the threshold (seeds 100..)")
    ap.add_argument("--n-eval", type=int, default=5,
                    help="models per poison ratio (seeds 0..)")
    args = ap.parse_args()

    os.makedirs(BASELINES, exist_ok=True)
    t = args.target_label
    spec = get_trigger(args.trigger)

    cal_acts = []
    for s in range(CALIBRATION_SEED_OFFSET, CALIBRATION_SEED_OFFSET + args.n_clean):
        ds = load_data(args.data, seed=s, processed=args.processed,
                       subsample=args.subsample)
        model, X, y, _ = train_federation(ds, s, args.rounds)
        cal_acts.append(penultimate_for_class(model, X, y, t))
    # the rule calibrate_ac_threshold applies (p95 of clean silhouettes), but
    # over each clean model's *own* training rows rather than one shared X
    nulls = [activation_clustering(a)["silhouette"] for a in cal_acts]
    thr = float(np.percentile(nulls, 95.0))
    print(f"calibrated threshold (p95 of {len(nulls)} clean silhouettes) = {thr:.3f}  "
          f"(paper default ~0.12)  nulls={np.round(nulls, 3).tolist()}", flush=True)

    rows, per_model, clean_sils = [], [], []
    for ratio in args.ratios:
        recs = []
        for seed in range(args.n_eval):
            ds = load_data(args.data, seed=seed, processed=args.processed,
                           subsample=args.subsample)
            atk = None if ratio == 0.0 else backdoor_attack(
                trigger=args.trigger, target_label=t, source_class=None,
                poison_ratio=ratio, rounds=args.rounds)
            model, X, y, poisoned = train_federation(ds, seed, args.rounds, attack=atk)
            acts = penultimate_for_class(model, X, y, t)
            pois_t = poisoned[np.asarray(y) == t]
            res = activation_clustering(acts, threshold=thr)
            in_suspect = res["labels"] == res["suspect_cluster"]
            asr = evaluate_backdoor(model, ds.X_test, ds.y_test, n_classes=ds.n_classes,
                                    target_label=t, spec=spec,
                                    stats=feature_stats(ds.X_train, ds.y_train),
                                    feature_names=ds.feature_names)
            rec = {"poison_ratio": ratio, "seed": seed, "target_rows": int(len(acts)),
                   "poisoned_rows": int(pois_t.sum()),
                   "silhouette": round(res["silhouette"], 4),
                   "flagged": res["flagged"],
                   "suspect_frac": round(res["suspect_cluster_frac"], 4),
                   "poison_in_suspect": (round(float(in_suspect[pois_t].mean()), 4)
                                         if pois_t.any() else ""),
                   "asr": round(asr, 4)}
            recs.append(rec); per_model.append(rec)
            print(f"ratio={ratio:>5.2f} seed={seed}  rows={rec['target_rows']} "
                  f"poisoned={rec['poisoned_rows']}  sil={rec['silhouette']:.3f}  "
                  f"flag={rec['flagged']}  suspect_frac={rec['suspect_frac']:.3f}  "
                  f"poison_in_suspect={rec['poison_in_suspect']}  asr={asr:.3f}",
                  flush=True)
        sils = np.array([r["silhouette"] for r in recs])
        if ratio == 0.0:
            clean_sils = sils.tolist()
        rate = float(np.mean([r["flagged"] for r in recs]))
        pis = [r["poison_in_suspect"] for r in recs if r["poison_in_suspect"] != ""]
        auc = (_auc(list(clean_sils) + sils.tolist(),
                    [0] * len(clean_sils) + [1] * len(sils))
               if ratio > 0 and clean_sils else float("nan"))
        rows.append({
            "poison_ratio": ratio,
            "n_models": len(recs),
            "poisoned_share": round(float(np.mean([r["poisoned_rows"] / max(1, r["target_rows"])
                                                   for r in recs])), 5),
            "silhouette_mean": round(float(sils.mean()), 4),
            "silhouette_std": round(float(sils.std(ddof=0)), 4),
            "flag_rate": rate,
            "tpr": rate if ratio > 0 else "",
            "fpr": rate if ratio == 0 else "",
            "auc_vs_clean": "" if np.isnan(auc) else round(auc, 4),
            "suspect_frac_mean": round(float(np.mean([r["suspect_frac"] for r in recs])), 4),
            "poison_in_suspect_mean": round(float(np.mean(pis)), 4) if pis else "",
            "asr_mean": round(float(np.mean([r["asr"] for r in recs])), 4),
        })

    name = ("activation_clustering.csv" if args.trigger == "oob_999"
            else f"activation_clustering_{args.trigger}.csv")
    out = os.path.join(BASELINES, name)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
        f.write(f"# calibrated_threshold,{thr:.4f}\n")
        f.write(f"# calibration_seeds,{CALIBRATION_SEED_OFFSET}-"
                f"{CALIBRATION_SEED_OFFSET + args.n_clean - 1}\n")
        f.write(f"# trigger,{args.trigger}\n")
    models_out = out.replace(".csv", "_models.csv")
    with open(models_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_model[0]))
        w.writeheader(); w.writerows(per_model)

    print(f"\n{'ratio':>6s} {'poisoned':>9s} {'silhouette':>16s} {'flag':>5s} "
          f"{'AUC':>6s} {'suspect':>8s} {'poison@suspect':>15s} {'ASR':>6s}")
    for r in rows:
        print(f"{r['poison_ratio']:6.2f} {r['poisoned_share']:9.4f} "
              f"{r['silhouette_mean']:8.3f} +-{r['silhouette_std']:5.3f} "
              f"{r['flag_rate']:5.2f} {str(r['auc_vs_clean']):>6s} "
              f"{r['suspect_frac_mean']:8.3f} {str(r['poison_in_suspect_mean']):>15s} "
              f"{r['asr_mean']:6.3f}")
    print(f"\nwrote {out}\nwrote {models_out}")


if __name__ == "__main__":
    main()
