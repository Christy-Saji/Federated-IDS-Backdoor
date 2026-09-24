"""The detection table — how well does each defense *identify* the attackers?

    python -m scripts.baselines.detection_report

This project is about backdoor **detection**, so the headline metric is not
whether a defense lowers dASR (it is measured, and none of them do) but whether
its per-client score separates the malicious clients from the honest ones.

Reads the recorded runs in ``results/`` rather than training anything: every
aggregator already logs ``client_scores`` / ``trust_scores`` per round into
``metrics.jsonl``, so the detection question can be answered from run_ids that
are already reproducible.

Three things ``detection_auc.csv`` does not report, and a detection write-up
needs:

* **Sign-corrected AUC.** An AUC of 0.08 is not "no signal" — it is a *perfect
  inverse* ranking, i.e. a detector with the comparison the wrong way round. It
  is reported alongside the raw value rather than quietly flipped, because which
  orientation a paper assumed is itself the finding.
* **Precision@k.** AUC is threshold-free; an operator has to actually name
  clients. With k = the true number of attackers, precision@k is "if you flagged
  the k most suspicious clients, how many were right".
* **The per-round curve.** Detection is not constant. The separation is largest
  in the first rounds and decays as the backdoor finishes being learned, which
  a mean over 20 rounds hides completely.
* **What the defense actually did.** A score that ranks the attackers is not
  the same as a server that removes them. ``attacker_removals`` /
  ``honest_removals`` count client-rounds excluded (FLAME: clustered out;
  FLTrust: trust clipped to 0) over *every* round - a final-round
  ``removed_clients`` of ``[]`` can hide a run that cut honest clients 16 times
  along the way. ``gradnorm_scorer`` is detection-only by design: it *flags* MAD
  outliers but still averages every client, so for it these columns count flags,
  not exclusions. ``fltrust+flame`` uses FLAME only as a pre-filter; FLAME
  removes nobody in any campaign run, so that arm reduces exactly to FLTrust.

Writes results/baselines/detection_report.csv
      results/figures/detection_by_round.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import yaml

from scripts._common import BASELINES, FIGURES, RESULTS

from flids.eval.metrics import _auc


def _roc(scores, labels):
    """(fpr, tpr) pairs over every threshold, for TPR-at-a-budget."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(labels[order])
    fp = np.cumsum(~labels[order])
    return fp / max(1, (~labels).sum()), tp / max(1, labels.sum())


def _tpr_at(scores, labels, max_fpr=0.1):
    fpr, tpr = _roc(scores, labels)
    ok = tpr[fpr <= max_fpr]
    return float(ok.max()) if len(ok) else 0.0


def _precision_at_k(scores, labels, k):
    """Flag the k highest-scoring clients; what fraction were malicious?"""
    if k <= 0:
        return float("nan")
    top = np.argsort(-np.asarray(scores, float), kind="mergesort")[:k]
    return float(np.asarray(labels).astype(bool)[top].mean())


def _series(run_dir):
    """Per-round (scores, kind) for a run, preferring the defense's own score."""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return [], None
    rounds, kind = [], None
    for line in open(path):
        if not line.strip():
            continue
        rec = json.loads(line)
        if "client_scores" in rec:
            rounds.append(rec["client_scores"]); kind = "client_scores"
        elif "trust_scores" in rec:
            # low trust == suspicious, so negate for the "higher is worse"
            # convention every metric here assumes
            rounds.append([-v for v in rec["trust_scores"]]); kind = "trust_scores"
    return rounds, kind


def _removals(run_dir, labels):
    """(attacker client-rounds removed, honest client-rounds removed, distinct
    attackers ever removed, rounds in which every attacker was removed)."""
    mal = hon = all_out = 0
    ever = set()
    for line in open(os.path.join(run_dir, "metrics.jsonl")):
        if not line.strip():
            continue
        removed = set(json.loads(line).get("removed_clients") or [])
        m = {c for c in removed if labels[c]}
        mal += len(m); hon += len(removed) - len(m); ever |= m
        all_out += int(len(m) == int(labels.sum()))
    return mal, hon, len(ever), all_out


def collect(real_only=True):
    rows = []
    for summary_path in sorted(glob.glob(os.path.join(RESULTS, "*", "summary.json"))):
        run_dir = os.path.dirname(summary_path)
        summary = json.load(open(summary_path))
        cfg_path = os.path.join(run_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        cfg = yaml.safe_load(open(cfg_path))
        data_cfg = cfg.get("data") or {}
        if real_only and not (data_cfg.get("processed_dir") or data_cfg.get("path")):
            continue
        attack = cfg.get("attack") or {}
        malicious = list(attack.get("malicious_clients") or [])
        if not malicious:
            continue
        trig = attack.get("trigger") or {}
        rung = trig.get("name") if isinstance(trig, dict) else None
        rung = rung or "?"
        n_clients = int(data_cfg.get("n_clients", 10))
        labels = np.zeros(n_clients, bool)
        labels[malicious] = True

        per_round, kind = _series(run_dir)
        if not per_round:
            continue

        aucs = [_auc(s, labels) for s in per_round]
        aucs = [a for a in aucs if not np.isnan(a)]
        if not aucs:
            continue
        corrected = [max(a, 1 - a) for a in aucs]
        inverted = float(np.mean(aucs)) < 0.5
        mal_rm, hon_rm, ever_rm, all_rm = _removals(run_dir, labels)
        n_mal, n_hon = int(labels.sum()), int((~labels).sum())
        prec = [_precision_at_k(s if not inverted else [-v for v in s],
                                labels, len(malicious)) for s in per_round]
        tprs = [_tpr_at(s if not inverted else [-v for v in s], labels)
                for s in per_round]

        rows.append({
            "run_id": summary.get("run_id", os.path.basename(run_dir)),
            "aggregator": (cfg.get("federated") or {}).get("aggregator", "?"),
            "trigger": rung,
            "seed": int(cfg.get("seed", 0)),
            "score": kind,
            "rounds": len(aucs),
            "auc_raw_mean": round(float(np.mean(aucs)), 4),
            "auc_raw_first": round(aucs[0], 4),
            "orientation": "inverted" if inverted else "as-published",
            "auc_corrected_mean": round(float(np.mean(corrected)), 4),
            "auc_corrected_max": round(float(np.max(corrected)), 4),
            "perfect_rounds": int(sum(1 for a in corrected if a == 1.0)),
            "precision_at_k_mean": round(float(np.mean(prec)), 4),
            "tpr_at_fpr10_mean": round(float(np.mean(tprs)), 4),
            "attacker_removals": mal_rm,
            "attacker_removal_rate": round(mal_rm / max(1, n_mal * len(per_round)), 4),
            "honest_removals": hon_rm,
            "honest_removal_rate": round(hon_rm / max(1, n_hon * len(per_round)), 4),
            "attackers_ever_removed": ever_rm,
            "rounds_all_attackers_removed": all_rm,
            "asr_final": summary.get("asr_final"),
            "_curve": corrected,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="include synthetic runs too (default: real data only)")
    args = ap.parse_args()

    rows = collect(real_only=not args.all)
    if not rows:
        print("no attack runs with per-client scores found in results/ — "
              "run scripts.baselines.run_all_real first")
        return

    fields = [k for k in rows[0] if not k.startswith("_")]
    out = os.path.join(BASELINES, "detection_report.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r[k] for k in fields} for r in rows])

    width = max(len(r["aggregator"]) for r in rows) + 2
    tw = max(len(r["trigger"]) for r in rows) + 2

    # Every table below is grouped by trigger first. Pooling rungs would be a
    # measurement error, not a presentation choice: the same aggregator at
    # oob_999 and at inbounds_free is two different experiments, and averaging
    # them turns FLAME's stable 5/5 inversion into a meaningless "orientation
    # not stable" across ten mixed runs.
    for rung in sorted({r["trigger"] for r in rows}):
        sub = [r for r in rows if r["trigger"] == rung]
        print(f"\n=== trigger: {rung} "
              f"({len(sub)} runs, seeds {sorted({r['seed'] for r in sub})}) ===")
        print(f"{'aggregator':{width}s} {'seed':>4s} {'raw AUC':>8s} {'orientation':>13s} "
              f"{'corr AUC':>9s} {'P@k':>6s} {'TPR@10%':>8s} {'perfect':>8s}")
        print("-" * (width + 62))
        for r in sorted(sub, key=lambda r: (r["aggregator"], r["seed"])):
            print(f"{r['aggregator']:{width}s} {r['seed']:4d} {r['auc_raw_mean']:8.3f} "
                  f"{r['orientation']:>13s} {r['auc_corrected_mean']:9.3f} "
                  f"{r['precision_at_k_mean']:6.2f} {r['tpr_at_fpr10_mean']:8.2f} "
                  f"{r['perfect_rounds']:4d}/{r['rounds']:<3d}")

        # Scoring the attackers and removing them are different claims. This is
        # the second one, which is the one an operator actually cares about.
        print(f"\n{'aggregator':{width}s} {'seed':>4s} {'attackers cut':>14s} "
              f"{'honest cut':>12s} {'ever cut':>9s} {'all cut':>10s} {'final ASR':>10s}")
        print("-" * (width + 64))
        for r in sorted(sub, key=lambda r: (r["aggregator"], r["seed"])):
            print(f"{r['aggregator']:{width}s} {r['seed']:4d} "
                  f"{r['attacker_removals']:5d} ({r['attacker_removal_rate']:5.1%}) "
                  f"{r['honest_removals']:4d} ({r['honest_removal_rate']:5.1%}) "
                  f"{r['attackers_ever_removed']:9d} {r['rounds_all_attackers_removed']:10d} "
                  f"{r['asr_final'] if r['asr_final'] is not None else float('nan'):10.3f}")

    # Across-seed spread is the thing a single-seed table cannot show, and the
    # thing most likely to turn a headline into an artefact. Print it loudly.
    by_key: dict[tuple, list] = {}
    for r in rows:
        by_key.setdefault((r["trigger"], r["aggregator"]), []).append(r)
    multi = {k: rs for k, rs in by_key.items() if len(rs) > 1}
    if multi:
        print(f"\n=== across seeds, per trigger ===")
        print(f"{'trigger':{tw}s} {'aggregator':{width}s} {'seeds':>6s} "
              f"{'raw AUC mean+-std':>20s} {'orientations':>26s}")
        print("-" * (tw + width + 56))
        for (rung, a), rs in sorted(multi.items()):
            raws = np.array([r["auc_raw_mean"] for r in rs], float)
            orients = sorted({r["orientation"] for r in rs})
            flag = "  <-- ORIENTATION NOT STABLE" if len(orients) > 1 else ""
            print(f"{rung:{tw}s} {a:{width}s} {len(rs):6d} "
                  f"{raws.mean():9.3f} +-{raws.std(ddof=0):7.3f} "
                  f"{'/'.join(orients):>26s}{flag}")

    print(f"\nwrote {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(FIGURES, exist_ok=True)
    # One figure per rung. Six aggregators x five seeds on a shared axis is 30
    # lines and an unreadable legend, and the two rungs are not comparable
    # round-for-round anyway.
    for rung in sorted({r["trigger"] for r in rows}):
        sub = [r for r in rows if r["trigger"] == rung]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for r in sorted(sub, key=lambda r: (r["aggregator"], r["seed"])):
            ax.plot(r["_curve"], marker="o", ms=3, lw=1.6,
                    label=f"{r['aggregator']} s{r['seed']} ({r['orientation']})")
        ax.axhline(0.5, color="0.5", ls="--", lw=1, label="chance")
        ax.set_xlabel("federated round")
        ax.set_ylabel("sign-corrected detection AUC")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Malicious-client detection over time — {rung} "
                     f"(real CIC-IDS2017)")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        fig.tight_layout()
        # oob_999 keeps the historical filename so existing references and the
        # dashboard's fallback figure do not break.
        stem = "detection_by_round" if rung == "oob_999" else f"detection_by_round_{rung}"
        path = os.path.join(FIGURES, f"{stem}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
