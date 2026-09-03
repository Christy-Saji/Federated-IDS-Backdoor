"""Task 1.4 - the experiment runner. One config in, one run_id out.

    python -m flids.runner --config configs/clean_fedavg.yaml

Writes results/<run_id>/
    config.yaml     exact config used, including resolved defaults
    metrics.jsonl   one JSON object per round
    summary.json    final aggregate metrics
    model_final.npz flat global parameter vector
    env.json        package versions, git commit, hardware, seed

run_id = sha256(canonical_json(resolved_config))[:12]. Refuses to overwrite an
existing run_id - notebooks read results/, they never write to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import yaml

from flids.data.loaders import load_dataset, load_processed, synthetic_dataset
from flids.eval.metrics import backdoor_lifespan, load_clean_asr
from flids.fl.server import FederatedServer
from flids.utils.seeding import env_info

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(ROOT, "results")

DEFAULTS = {
    "data": {"dataset": "synthetic", "labels": "multiclass",
             "partition": "dirichlet", "alpha": 0.5, "n_clients": 10,
             "path": None, "processed_dir": None, "min_size": 100},
    "model": {"arch": "mlp", "hidden": [256, 128, 64], "dropout": 0.3},
    "federated": {"rounds": 100, "local_epochs": 2, "lr": 0.001,
                  "batch_size": 256, "aggregator": "fedavg"},
    "attack": {"enabled": False, "type": None, "malicious_clients": [],
               "trigger": None, "poison_ratio": 0.0, "attack_window": [1, 20]},
    "seed": 0,
}


def _deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve_config(cfg: dict) -> dict:
    resolved = _deep_merge(DEFAULTS, cfg)
    resolved.setdefault("run_name", cfg.get("run_name", "run"))
    return resolved


def canonical_json(cfg: dict) -> str:
    scrub = {k: v for k, v in cfg.items() if k != "run_name"}
    return json.dumps(scrub, sort_keys=True, separators=(",", ":"))


def run_id_for(cfg: dict) -> str:
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()[:12]


def _load_data(cfg: dict, seed: int):
    d = cfg["data"]
    multiclass = d["labels"] == "multiclass"
    if d.get("processed_dir"):
        return load_processed(d["processed_dir"],
                              n_classes=8 if multiclass else 2)
    if d.get("path"):
        return load_dataset(d["path"], seed=seed, multiclass=multiclass)
    return synthetic_dataset(seed=seed, multiclass=multiclass,
                             n_classes=8 if multiclass else 2,
                             n_train=d.get("n_train", 12000),
                             n_test=d.get("n_test", 4000))


def run(config_path: str, overwrite: bool = False) -> str:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = resolve_config(raw)
    rid = run_id_for(cfg)
    out_dir = os.path.join(RESULTS_ROOT, rid)

    if os.path.exists(out_dir) and not overwrite:
        summary = json.load(open(os.path.join(out_dir, "summary.json")))
        print(f"[runner] run_id {rid} already exists - refusing to overwrite.")
        print(f"[runner] existing summary: {summary.get('final', {})}")
        return rid
    os.makedirs(out_dir, exist_ok=True)

    seed = cfg["seed"]
    data = _load_data(cfg, seed)

    server = FederatedServer(data, cfg, seed=seed)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    with open(metrics_path, "w") as mf:
        def _write(rec):
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
        final_params, history = server.run(on_round=_write)

    np.savez_compressed(os.path.join(out_dir, "model_final.npz"),
                        params=final_params)

    final = history[-1] if history else {}
    summary = {
        "run_id": rid,
        "run_name": cfg["run_name"],
        "rounds": len(history),
        "final": final,
        "best_macro_f1": max((h["macro_f1"] for h in history), default=None),
        "n_malicious": len(cfg["attack"].get("malicious_clients", []) or []),
        "n_classes": data.n_classes,
    }
    if cfg["attack"]["enabled"] and history and "asr" in history[-1]:
        asr_series = [h.get("asr", float("nan")) for h in history]
        end = (cfg["attack"]["attack_window"][1] - 1)
        summary["asr_final"] = asr_series[-1]
        summary["asr_peak"] = float(np.nanmax(asr_series))
        summary["backdoor_lifespan"] = backdoor_lifespan(asr_series, end)

        # G-01: the only ASR number the project reports is dASR = asr - asr_clean.
        trig = cfg["attack"].get("trigger", {}) or {}
        trigger_name = trig.get("name") if isinstance(trig, dict) else None
        baselines = load_clean_asr(os.path.join(RESULTS_ROOT, "baselines",
                                                "clean_asr.csv"))
        asr_clean = baselines.get((trigger_name, seed)) if trigger_name else None
        if asr_clean is not None:
            dasr = [a - asr_clean for a in asr_series]
            summary["asr_clean_baseline"] = asr_clean
            summary["dasr_final"] = dasr[-1]
            summary["dasr_peak"] = float(np.nanmax(dasr))
            summary["dasr_by_round"] = dasr
            summary["backdoor_lifespan"] = backdoor_lifespan(dasr, end)
        else:
            summary["note_dasr"] = (
                "raw ASR only - no clean_asr.csv row for "
                f"(trigger={trigger_name}, seed={seed}). Run scripts.baselines.clean_asr.")

    if history and "detection_auc" in history[-1]:
        aucs = [h["detection_auc"] for h in history if "detection_auc" in h]
        summary["detection_auc_mean"] = float(np.mean(aucs))
    if history and "trust_auc" in history[-1]:
        aucs = [h["trust_auc"] for h in history if "trust_auc" in h]
        summary["trust_auc_mean"] = float(np.mean(aucs))

    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    yaml.safe_dump(cfg, open(os.path.join(out_dir, "config.yaml"), "w"),
                   sort_keys=True)
    json.dump(env_info(seed), open(os.path.join(out_dir, "env.json"), "w"),
              indent=2)

    print(f"[runner] {rid}  {cfg['run_name']}")
    print(f"[runner] final: {final}")
    print(f"[runner] wrote results/{rid}/")
    return rid


def main():
    p = argparse.ArgumentParser(description="flids experiment runner")
    p.add_argument("--config", required=True)
    p.add_argument("--overwrite", action="store_true",
                   help="dev only - Gate G1 requires the no-overwrite guarantee")
    args = p.parse_args()
    run(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
