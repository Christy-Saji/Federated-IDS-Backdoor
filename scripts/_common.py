"""Shared helpers for the runnable scripts.

Every script is a module under ``scripts/`` and is run with ``-m`` from the repo
root, e.g. ``python -m scripts.baselines.clean_asr``. That puts the repo root on
``sys.path`` for free, so nothing here manipulates the path.

Directory conventions (all under the repo-root ``results/``):

    results/<run_id>/        one directory per flids.runner run
    results/baselines/       Phase 2 baseline tables (clean_asr.csv, ...)
    results/calibration/     null distributions + calibrated thresholds
    results/figures/         plots
    results/validation/      Phase 0 triage outputs
    results/partition_viz/   Dirichlet partition figures
"""

from __future__ import annotations

import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
BASELINES = os.path.join(RESULTS, "baselines")
CALIBRATION = os.path.join(RESULTS, "calibration")
FIGURES = os.path.join(RESULTS, "figures")
VALIDATION = os.path.join(RESULTS, "validation")

for _d in (RESULTS, BASELINES, CALIBRATION, FIGURES, VALIDATION):
    os.makedirs(_d, exist_ok=True)

# back-compat alias for the Phase 0 scripts
RESULTS_DIR = VALIDATION


# ---------------------------------------------------------------------------
# dataset resolution
# ---------------------------------------------------------------------------
def add_data_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=None,
                   help="Path to a CIC-IDS2017-style CSV or directory. If "
                        "omitted, a synthetic dataset with the same 77-feature "
                        "schema is used.")


def load_data(path=None, seed: int = 0, multiclass: bool = True):
    from flids.data.loaders import load_dataset, synthetic_dataset
    if path:
        return load_dataset(path, seed=seed, multiclass=multiclass)
    return synthetic_dataset(seed=seed, multiclass=multiclass,
                             n_classes=8 if multiclass else 2)


def resolve_dataset(args, seed: int = 0):
    """Phase 0 triage helper. Preserves the original Phase 0 behaviour: the
    synthetic fallback is *binary* (BENIGN/ATTACK), a real --data load is
    multi-class."""
    from flids.data.loaders import load_dataset, synthetic_dataset
    if getattr(args, "data", None):
        print(f"[data] loading {args.data}")
        return load_dataset(args.data, seed=seed)
    print("[data] no --data given; using synthetic CIC-IDS2017-like dataset")
    return synthetic_dataset(seed=seed)


# ---------------------------------------------------------------------------
# short FedAvg runs for the Phase 2 baseline scripts
# ---------------------------------------------------------------------------
def _cfg(seed, rounds, aggregator="fedavg", attack=None):
    return {
        "run_name": "script_helper",
        "data": {"dataset": "synthetic", "labels": "multiclass",
                 "partition": "dirichlet", "alpha": 0.5, "n_clients": 10,
                 "min_size": 100},
        "model": {"arch": "mlp", "hidden": [256, 128, 64], "dropout": 0.3},
        "federated": {"rounds": rounds, "local_epochs": 2, "lr": 0.05,
                      "batch_size": 256, "aggregator": aggregator},
        "attack": attack or {"enabled": False, "malicious_clients": [],
                             "attack_window": [1, rounds]},
        "seed": seed,
    }


def train_model(ds, seed=0, rounds=15, attack=None):
    """Return a fitted MLP (global model) from a short FedAvg run."""
    from flids.fl.server import FederatedServer
    server = FederatedServer(ds, _cfg(seed, rounds, attack=attack), seed=seed)
    params, _ = server.run()
    model = server._new_model()
    model.set_params(params)
    return model


def backdoor_attack(trigger="oob_999", target_label=0, source_class=2,
                    poison_ratio=0.3, malicious=(0, 1, 2, 3), rounds=15):
    return {"enabled": True, "type": "badnets",
            "malicious_clients": list(malicious),
            "trigger": {"name": trigger, "target_label": target_label,
                        "source_class": source_class},
            "poison_ratio": poison_ratio, "attack_window": [1, rounds]}
