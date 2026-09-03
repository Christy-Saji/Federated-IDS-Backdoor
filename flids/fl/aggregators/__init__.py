"""Aggregator registry.

Only FedAvg is a finished implementation in Phase 1. FLTrust / FLAME / GradNorm
are scaffolded with the correct interface and a Phase 2 pointer - Phase 0 Task
0.4 documented why the current versions are unfaithful, and Phase 2 rebuilds
them from the papers.
"""

from __future__ import annotations

from .combined import FLTrustFLAME
from .fedavg import FedAvg
from .flame import FLAME
from .fltrust import FLTrust
from .gradnorm import GradNorm, GradNormScorer

_REGISTRY = {
    "fedavg": FedAvg,
    "fltrust": FLTrust,
    "flame": FLAME,
    "gradnorm": GradNorm,
    "gradnorm_scorer": GradNormScorer,
    "fltrust+flame": FLTrustFLAME,
}

# kwargs each aggregator actually accepts - build_aggregator filters to these so
# the runner can pass a superset without every aggregator needing **_ignored.
_ACCEPTS = {
    "fedavg": (), "gradnorm": (),
    "gradnorm_scorer": ("k",),
    "fltrust": ("server_update_fn", "n_clients"),
    "flame": ("noise_lambda", "n_clients", "seed"),
    "fltrust+flame": ("server_update_fn", "noise_lambda", "n_clients", "seed"),
}


def build_aggregator(name: str, **kw):
    if name not in _REGISTRY:
        raise KeyError(f"unknown aggregator '{name}'. known: {sorted(_REGISTRY)}")
    allowed = {k: v for k, v in kw.items() if k in _ACCEPTS[name]}
    return _REGISTRY[name](**allowed)


def available():
    return sorted(_REGISTRY)
