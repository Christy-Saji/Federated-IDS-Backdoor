"""Phase 0 compatibility shim.

Phase 0's the Phase 0 triage scripts (``scripts/validation/``)
import ``train_clean_fedavg`` / ``train_backdoored_fedavg`` from ``flids.fl``.
Keep them working with the original small IID loop so Phase 0 numbers do not
move. New work should use ``flids.fl.server.FederatedServer``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flids.attacks.badnets import TRIGGER_VALUE, poison_split
from flids.data.loaders import Dataset
from flids.models.mlp import MLP


@dataclass
class FLConfig:
    n_clients: int = 10
    rounds: int = 5
    local_epochs: int = 1
    lr: float = 0.05
    batch_size: int = 128
    n_malicious: int = 0
    poison_frac: float = 0.5
    trigger_value: float = TRIGGER_VALUE


def _iid_split(n, n_clients, seed):
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), n_clients)


def _run(data: Dataset, cfg: FLConfig, seed: int) -> np.ndarray:
    nf = data.n_features
    shards = _iid_split(len(data.X_train), cfg.n_clients, seed)
    global_params = MLP(n_features=nf, seed=seed).get_params()
    for rnd in range(cfg.rounds):
        client_params = []
        for cid, idx in enumerate(shards):
            Xc, yc = data.X_train[idx], data.y_train[idx]
            if cid < cfg.n_malicious:
                Xc, yc = poison_split(Xc, yc, cfg.poison_frac,
                                      cfg.trigger_value, seed=seed * 100 + cid,
                                      source_classes={1})
            m = MLP(n_features=nf)
            m.set_params(global_params)
            m.fit(Xc, yc, epochs=cfg.local_epochs, batch_size=cfg.batch_size,
                  lr=cfg.lr, seed=seed * 1000 + rnd * 10 + cid)
            client_params.append(m.get_params())
        global_params = np.mean(client_params, axis=0)
    return global_params


def train_clean_fedavg(data, rounds=5, seed=0, n_clients=10):
    return _run(data, FLConfig(n_clients=n_clients, rounds=rounds), seed)


def train_backdoored_fedavg(data, rounds=5, seed=0, n_clients=10, n_malicious=4,
                            poison_frac=0.5, trigger_value=TRIGGER_VALUE):
    return _run(data, FLConfig(n_clients=n_clients, rounds=rounds,
                               n_malicious=n_malicious, poison_frac=poison_frac,
                               trigger_value=trigger_value), seed)
