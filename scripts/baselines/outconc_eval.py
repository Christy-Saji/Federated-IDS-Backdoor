"""Evaluate the output-concentration detector (our detector, not a paper's).

    python -m scripts.baselines.outconc_eval [--processed] [--seeds 5 6 7 8 9]

Trains the campaign attack under the ``outconc_scorer`` aggregator (which
aggregates with plain FedAvg and logs each client's output-concentration score),
run-averages the per-round scores, and reports how well they rank the attackers.

Read the design and rationale in ``flids/fl/aggregators/outconc.py``. The short
version: to send a triggered flow to the target class an attacker must rewrite
the final layer toward that one class, which honest non-IID clients do not; the
detector measures exactly that and nothing else.

**Report from seeds disjoint from the ones the detector was built on.** The
signal was chosen on seeds 0-4; the default here is the held-out set 5-9. It is
also run at ``--n-mal 0`` for the false-positive rate and on ``oob_999`` to show
the extreme trigger is undetectable (too easy to leave a fingerprint).

Writes results/baselines/outconc_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from scripts._common import BASELINES, _cfg, add_data_arg, backdoor_attack, load_data

from flids.eval.metrics import _auc
from flids.fl.server import FederatedServer


def run_one(ds, seed, trigger, n_mal, rounds):
    atk = backdoor_attack(trigger=trigger, source_class=None, poison_ratio=0.5,
                          malicious=tuple(range(n_mal)), rounds=rounds)
    atk["enabled"] = n_mal > 0
    cfg = _cfg(seed, rounds, aggregator="outconc_scorer", attack=atk)
    server = FederatedServer(ds, cfg, seed=seed)
    _, hist = server.run()
    mal = server.malicious_mask
    scores = np.array([h["client_scores"] for h in hist if "client_scores" in h])
    run_avg = scores.mean(axis=0)
    return run_avg, mal


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_arg(ap)
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--triggers", nargs="+", default=["inbounds_free", "oob_999"])
    ap.add_argument("--n-mal", type=int, nargs="+", default=[4, 0])
    args = ap.parse_args()

    os.makedirs(BASELINES, exist_ok=True)
    rows = []
    for trigger in args.triggers:
        for n_mal in args.n_mal:
            aucs, precs, fp, tot_honest = [], [], 0, 0
            for seed in args.seeds:
                ds = load_data(args.data, seed=seed, processed=args.processed,
                               subsample=args.subsample)
                s, mal = run_one(ds, seed, trigger, n_mal, args.rounds)
                k = int(mal.sum())
                if k:
                    aucs.append(_auc(s, mal))
                    precs.append(float(mal[np.argsort(-s)[:k]].mean()))
                # false positives at a fixed robust-z >= 3.5 cutoff
                med = np.median(s); madv = np.median(np.abs(s - med)) * 1.4826 + 1e-12
                flagged = (s - med) / madv >= 3.5
                fp += int((flagged & ~mal).sum()); tot_honest += int((~mal).sum())
                tag = f"{trigger} m{n_mal} s{seed}"
                print(f"{tag:26s} AUC {aucs[-1] if k else float('nan'):.3f}  "
                      f"honest flagged {int((flagged & ~mal).sum())}/{int((~mal).sum())}",
                      flush=True)
            row = {"trigger": trigger, "n_malicious": n_mal, "seeds": len(args.seeds),
                   "auc_mean": round(float(np.mean(aucs)), 4) if aucs else "",
                   "auc_min": round(float(np.min(aucs)), 4) if aucs else "",
                   "auc_max": round(float(np.max(aucs)), 4) if aucs else "",
                   "precision_at_k": round(float(np.mean(precs)), 4) if precs else "",
                   "honest_fpr": round(fp / max(1, tot_honest), 4)}
            rows.append(row)

    out = os.path.join(BASELINES, "outconc_eval.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\ntrigger        n_mal  AUC (mean [min-max])      precision@k   honest FPR")
    for r in rows:
        a = (f"{r['auc_mean']} [{r['auc_min']}-{r['auc_max']}]" if r["auc_mean"] != "" else "n/a (no attackers)")
        print(f"{r['trigger']:14s} {r['n_malicious']:>4d}   {a:26s} {str(r['precision_at_k']):>10s}   {r['honest_fpr']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
