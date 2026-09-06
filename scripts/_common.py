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
DEFAULT_PROCESSED = os.path.join(ROOT, "data", "processed")
DEFAULT_SUBSAMPLE = 60000


def add_data_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=None,
                   help="Path to a CIC-IDS2017-style CSV or directory. Runs the "
                        "full preprocessing contract every time - prefer "
                        "--processed once data/processed exists.")
    p.add_argument("--processed", nargs="?", const=DEFAULT_PROCESSED, default=None,
                   help="Use the cached arrays from scripts.preprocessing.preprocess "
                        f"(default dir: {DEFAULT_PROCESSED}).")
    p.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE,
                   help="Stratified training rows to draw from --processed "
                        f"(default {DEFAULT_SUBSAMPLE}; 0 = the full split). "
                        "The models are pure numpy, so the full 1.89M-row split "
                        "is only practical for a final run.")


def load_data(path=None, seed: int = 0, multiclass: bool = True,
              processed=None, subsample=DEFAULT_SUBSAMPLE):
    from flids.data.loaders import load_dataset, load_processed, synthetic_dataset
    n_classes = 8 if multiclass else 2
    if processed:
        return load_processed(processed, n_classes=n_classes,
                              subsample=subsample or None, seed=seed)
    if path:
        return load_dataset(path, seed=seed, multiclass=multiclass)
    return synthetic_dataset(seed=seed, multiclass=multiclass,
                             n_classes=n_classes)


def resolve_dataset(args, seed: int = 0, multiclass: bool = True):
    """Dataset resolution for the scripts, in precedence order:
    --processed (cached real arrays) > --data (raw CSVs) > synthetic fallback.

    Phase 0 note: the synthetic fallback stays *binary* (BENIGN/ATTACK) as it was
    when Phase 0 was written, so the old synthetic numbers remain comparable. Any
    real load is multi-class - the 8 families are what the report needs.
    """
    from flids.data.loaders import load_dataset, load_processed, synthetic_dataset
    processed = getattr(args, "processed", None)
    subsample = getattr(args, "subsample", DEFAULT_SUBSAMPLE)
    if processed:
        n = subsample or None
        print(f"[data] cached real data from {processed}"
              + (f" (stratified subsample of {n} train rows)" if n else " (full split)"))
        ds = load_processed(processed, n_classes=8, subsample=n, seed=seed)
        print(f"[data] train={ds.X_train.shape} test={ds.X_test.shape} "
              f"balance={ {k: round(v, 4) for k, v in ds.class_balance().items()} }")
        return ds
    if getattr(args, "data", None):
        print(f"[data] loading {args.data}")
        return load_dataset(args.data, seed=seed, multiclass=multiclass)
    print("[data] no --data/--processed given; using synthetic CIC-IDS2017-like dataset")
    # Deliberately NOT forwarding `multiclass` here: the Phase 0 fallback is
    # binary and must stay binary, or every stored synthetic triage number moves.
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
