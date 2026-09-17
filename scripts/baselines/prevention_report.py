"""The prevention table across seeds - does any defense lower the attack?

    python -m scripts.baselines.prevention_report

Reads the recorded attack runs in ``results/`` (real data, ladder trigger, the
20-round campaign shape - durability runs are a different protocol and are left
out) and reports accuracy, macro-F1, raw ASR and dASR per aggregator as
mean +- std over every seed present. Trains nothing.

It also re-scores each run's saved ``model_final.npz`` on that run's own test
split to get **per-class F1**, and macro-F1 with and without Infiltration.
Infiltration has 27 training rows against a rare-support threshold of 50, so the
8-class macro-F1 is partly a measurement of one family the model has almost no
data for (HANDOFF open question 2). Reporting both lets that question be argued
from numbers without changing the data contract - dropping or merging the
family would re-id every run in results/.

Writes results/baselines/prevention_report.csv  (one row per run)
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
import yaml

from scripts._common import BASELINES, RESULTS

from flids.data.labels import CLASS_NAMES
from flids.data.loaders import load_processed
from flids.eval.metrics import confusion, macro_f1_from_cm
from flids.models.registry import build_model

INFILTRATION = CLASS_NAMES.index("Infiltration")


def _per_class_f1(cm):
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def collect():
    runs, cache = [], {}
    for summary_path in sorted(glob.glob(os.path.join(RESULTS, "*", "summary.json"))):
        run_dir = os.path.dirname(summary_path)
        cfg_path = os.path.join(run_dir, "config.yaml")
        model_path = os.path.join(run_dir, "model_final.npz")
        if not (os.path.exists(cfg_path) and os.path.exists(model_path)):
            continue
        cfg = yaml.safe_load(open(cfg_path))
        data_cfg, attack = cfg.get("data") or {}, cfg.get("attack") or {}
        trigger = attack.get("trigger") or {}
        fed = cfg.get("federated") or {}
        if not (data_cfg.get("processed_dir") and attack.get("enabled")
                and isinstance(trigger, dict) and trigger.get("name")):
            continue
        if int(fed.get("rounds", 0)) != int((attack.get("attack_window") or [0, 0])[1]):
            continue            # durability protocol: attacker exits early
        summary = json.load(open(summary_path))

        seed = int(cfg.get("seed", 0))
        key = (data_cfg["processed_dir"], data_cfg.get("subsample"), seed)
        if key not in cache:
            processed = data_cfg["processed_dir"]
            if not os.path.isabs(processed):
                processed = os.path.join(os.path.dirname(RESULTS), processed)
            cache[key] = load_processed(processed, n_classes=int(summary.get("n_classes", 8)),
                                        subsample=data_cfg.get("subsample"), seed=seed)
        ds = cache[key]
        model_cfg = cfg.get("model") or {}
        model = build_model(model_cfg.get("arch", "mlp"), n_features=ds.n_features,
                            n_classes=ds.n_classes, seed=seed,
                            hidden=tuple(model_cfg.get("hidden", (256, 128, 64))))
        model.set_params(np.load(model_path)["params"])
        cm = confusion(ds.y_test, model.predict(ds.X_test), ds.n_classes)
        f1 = _per_class_f1(cm)
        present = cm.sum(axis=1) > 0
        keep = present.copy(); keep[INFILTRATION] = False

        final = summary.get("final") or {}
        row = {
            "run_id": summary.get("run_id", os.path.basename(run_dir)),
            "aggregator": fed.get("aggregator", "?"),
            "trigger": trigger["name"],
            "seed": seed,
            "accuracy": round(float(final.get("accuracy", float("nan"))), 4),
            "macro_f1": round(float(final.get("macro_f1", float("nan"))), 4),
            "macro_f1_rescored": round(macro_f1_from_cm(cm), 4),
            "macro_f1_excl_infiltration": round(float(f1[keep].mean()), 4),
            "infiltration_test_rows": int(cm[INFILTRATION].sum()),
            "asr_final": summary.get("asr_final"),
            "asr_clean_baseline": summary.get("asr_clean_baseline"),
            "dasr_final": summary.get("dasr_final"),
        }
        for i, name in enumerate(CLASS_NAMES[:ds.n_classes]):
            row[f"f1_{name}"] = round(float(f1[i]), 4)
        runs.append(row)
    return runs


def _ms(vals):
    v = np.array([x for x in vals if x is not None], float)
    return (f"{v.mean():.3f} +-{v.std(ddof=0):.3f}" if len(v) else "      -      ")


def main():
    rows = collect()
    if not rows:
        print("no real-data attack runs in results/ - run scripts.baselines.run_all_real")
        return
    out = os.path.join(BASELINES, "prevention_report.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["trigger"], r["aggregator"]), []).append(r)
    order = ["fedavg", "fltrust", "flame", "fltrust+flame", "gradnorm_scorer"]
    print(f"{'trigger':9s} {'aggregator':16s} {'seeds':>9s} {'accuracy':>14s} "
          f"{'macro-F1':>14s} {'mF1 w/o Infil':>14s} {'raw ASR':>14s} {'dASR':>14s}")
    print("-" * 112)
    for (trig, agg), rs in sorted(groups.items(),
                                  key=lambda kv: (kv[0][0], order.index(kv[0][1])
                                                  if kv[0][1] in order else 99)):
        seeds = ",".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))
        print(f"{trig:9s} {agg:16s} {seeds:>9s} {_ms(r['accuracy'] for r in rs):>14s} "
              f"{_ms(r['macro_f1'] for r in rs):>14s} "
              f"{_ms(r['macro_f1_excl_infiltration'] for r in rs):>14s} "
              f"{_ms(r['asr_final'] for r in rs):>14s} {_ms(r['dasr_final'] for r in rs):>14s}")

    names = [k for k in rows[0] if k.startswith("f1_")]
    print(f"\nper-class F1, mean over all {len(rows)} runs:")
    for k in names:
        print(f"  {k[3:]:13s} {np.mean([r[k] for r in rows]):.3f}")
    mism = [r["run_id"] for r in rows if abs(r["macro_f1"] - r["macro_f1_rescored"]) > 1e-3]
    if mism:
        print(f"\nWARNING: re-scored macro-F1 differs from summary.json for {mism} - "
              "the test split was not reproduced, so per-class numbers are suspect")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
