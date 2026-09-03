"""FedAvg - McMahan et al., 2017. Sample-size-weighted mean of client params."""

from __future__ import annotations

import numpy as np


class FedAvg:
    name = "fedavg"

    def __init__(self, **_ignored):
        pass

    def aggregate(self, global_params, client_params, client_sizes, ctx=None):
        w = np.asarray(client_sizes, dtype=float)
        w = w / w.sum()
        new_params = np.tensordot(w, np.stack(client_params), axes=1)
        info = {"weights": w.tolist(), "removed": [], "scores": None}
        return new_params, info
