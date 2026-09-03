"""Model registry - a config string selects an architecture (Task 1.6).

Model choice is not a claim: both are kept, the MLP is reported as an ablation.
"""

from __future__ import annotations

from .mlp import MLP
from .tabtransformer import TabTransformer

_REGISTRY = {
    "mlp": MLP,
    "tabtransformer": TabTransformer,
}


def build_model(arch: str, n_features: int, n_classes: int, seed: int = 0, **kw):
    if arch not in _REGISTRY:
        raise KeyError(f"unknown arch '{arch}'. known: {sorted(_REGISTRY)}")
    return _REGISTRY[arch](n_features=n_features, n_classes=n_classes,
                           seed=seed, **kw)


def available():
    return sorted(_REGISTRY)
