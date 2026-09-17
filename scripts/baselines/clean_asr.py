"""Task 2.2 - clean-model ASR baseline for every trigger rung.

    python -m scripts.baselines.clean_asr [--processed | --data PATH] [--rounds 20] [--seeds 0 1 2]
    python -m scripts.baselines.clean_asr --summary-only

For each rung x seed, train a *clean* FedAvg model and measure the trigger's ASR
on it. A trigger that is out of distribution fires on a clean model too; only
`dASR = ASR_model - ASR_clean` is attributable to the backdoor (G-01).

Rows are merged into the CSV by (trigger, seed), never truncated: the runner
reads this file for every seed it has ever run, so `--seeds 7` must not delete
the seed-0 row a finished run depends on. Several processes can sweep disjoint
seeds at once - the merge takes a lock file.

Gate G0's reframe decision hangs on the `oob_999` rung being bimodal across
seeds (0 / 0 / 1 at n=3), so the summary treats it as a yes/no event per seed
and reports a Clopper-Pearson interval on how often it fires, not a mean.
`top_class` records where a clean model sends the stamped flows, which is the
evidence for *why* it is bimodal.

Writes results/baselines/clean_asr.csv          (trigger,seed,asr_clean,top_class,top_class_share)
       results/baselines/clean_asr_summary.csv  (per rung, across every seed in the CSV)
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from scripts._common import BASELINES, add_data_arg, load_data

from flids.attacks.badnets import evaluate_backdoor, stamp_trigger
from flids.data.labels import CLASS_NAMES
from flids.data.triggers import TRIGGERS, feature_stats, get_trigger
from flids.fl.server import FederatedServer

RUNGS = [name for name, s in TRIGGERS.items() if s["value"] != "from_pcap"]
FIELDS = ["trigger", "seed", "asr_clean", "top_class", "top_class_share"]
CSV_PATH = os.path.join(BASELINES, "clean_asr.csv")
SUMMARY_PATH = os.path.join(BASELINES, "clean_asr_summary.csv")

# a seed "fires" if the clean model already sends this share of stamped flows to
# the target - the G0 rule's own ">= 0.9" branch
FIRES_AT = 0.9


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


def _read_rows(path=CSV_PATH):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _merge_rows(new_rows, path=CSV_PATH, timeout=600):
    """Upsert by (trigger, seed) under a lock file, so parallel sweeps compose."""
    lock = path + ".lock"
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() > deadline:
                raise RuntimeError(f"{lock} held for {timeout}s - remove it if stale")
            time.sleep(0.5)
    try:
        merged = {(r["trigger"], int(r["seed"])): r for r in _read_rows(path)}
        for r in new_rows:
            merged[(r["trigger"], int(r["seed"]))] = r
        order = {name: i for i, name in enumerate(RUNGS)}
        rows = sorted(merged.values(),
                      key=lambda r: (int(r["seed"]), order.get(r["trigger"], 99)))
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
        os.replace(tmp, path)
    finally:
        os.close(fd)
        os.remove(lock)
    return rows


def _clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def summarise(rows):
    out = []
    for rung in RUNGS:
        rs = [r for r in rows if r["trigger"] == rung]
        if not rs:
            continue
        vals = np.array([float(r["asr_clean"]) for r in rs])
        k, n = int((vals >= FIRES_AT).sum()), len(vals)
        lo, hi = _clopper_pearson(k, n)
        tops = [r.get("top_class") for r in rs if r.get("top_class")]
        top_counts = {c: tops.count(c) for c in sorted(set(tops))}
        out.append({
            "trigger": rung, "n_seeds": n,
            "asr_clean_mean": round(float(vals.mean()), 4),
            "asr_clean_std": round(float(vals.std(ddof=0)), 4),
            "asr_clean_min": round(float(vals.min()), 4),
            "asr_clean_max": round(float(vals.max()), 4),
            "seeds_firing": k,
            "fire_rate": round(k / n, 4),
            "fire_rate_ci95_lo": round(lo, 4),
            "fire_rate_ci95_hi": round(hi, 4),
            "seeds_near_zero": int((vals <= 1 - FIRES_AT).sum()),
            "top_class_counts": " ".join(f"{c}:{v}" for c, v in top_counts.items()),
        })
    return out


def _print_summary(summary):
    print(f"\n{'rung':14s} {'n':>3s} {'mean':>7s} {'std':>6s} {'fires>=0.9':>11s} "
          f"{'95% CI':>15s} {'~0':>4s}  where stamped flows go (clean model)")
    print("-" * 100)
    for s in summary:
        print(f"{s['trigger']:14s} {s['n_seeds']:3d} {s['asr_clean_mean']:7.3f} "
              f"{s['asr_clean_std']:6.3f} {s['seeds_firing']:5d}/{s['n_seeds']:<5d} "
              f"[{s['fire_rate_ci95_lo']:.2f}, {s['fire_rate_ci95_hi']:.2f}] "
              f"{s['seeds_near_zero']:4d}  {s['top_class_counts']}")


def main():
    ap = argparse.ArgumentParser()
    add_data_arg(ap)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--target-label", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true",
                    help="train nothing; re-summarise the seeds already in the CSV")
    args = ap.parse_args()

    os.makedirs(BASELINES, exist_ok=True)
    if not args.summary_only:
        for seed in args.seeds:
            ds = load_data(args.data, seed=seed, processed=args.processed,
                           subsample=args.subsample)
            cfg = clean_cfg(args.rounds, seed)
            server = FederatedServer(ds, cfg, seed=seed)
            params, _ = server.run()
            model = server._new_model()
            model.set_params(params)
            stats = feature_stats(ds.X_train, ds.y_train)
            names = CLASS_NAMES[:ds.n_classes]
            source = ds.y_test != args.target_label

            rows = []
            for rung in RUNGS:
                spec = get_trigger(rung)
                asr = evaluate_backdoor(
                    model, ds.X_test, ds.y_test, n_classes=ds.n_classes,
                    target_label=args.target_label,
                    spec=spec, stats=stats,
                    feature_names=ds.feature_names)
                preds = model.predict(stamp_trigger(ds.X_test[source], spec=spec,
                                                    stats=stats,
                                                    feature_names=ds.feature_names))
                counts = np.bincount(preds, minlength=ds.n_classes)
                top = int(counts.argmax())
                rows.append({"trigger": rung, "seed": seed, "asr_clean": round(asr, 6),
                             "top_class": names[top],
                             "top_class_share": round(float(counts[top] / counts.sum()), 6)})
                print(f"{rung:14s} seed={seed}  asr_clean={asr:.4f}  "
                      f"stamped flows -> {names[top]} ({counts[top] / counts.sum():.2f})",
                      flush=True)
            # merge after every seed, so an interrupted sweep keeps what it finished
            _merge_rows(rows)
        print(f"\nmerged {len(args.seeds)} seed(s) into {CSV_PATH}")

    summary = summarise(_read_rows())
    if summary:
        with open(SUMMARY_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        _print_summary(summary)
        print(f"\nwrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
