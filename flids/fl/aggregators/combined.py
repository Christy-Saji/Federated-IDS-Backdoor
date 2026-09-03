"""fltrust+flame - FLAME filters first, FLTrust trust-weights the survivors (Task 2.9).

Composition order is fixed and logged: FLAME's HDBSCAN removes the outlier
updates, then FLTrust computes ReLU(cos) trust against the root-set update and
normalises + weights whatever FLAME kept. Both stages' per-client scores land in
`info` so metrics.jsonl carries them.
"""

from __future__ import annotations

import numpy as np

from .flame import FLAME
from .fltrust import FLTrust


class FLTrustFLAME:
    name = "fltrust+flame"

    def __init__(self, server_update_fn=None, noise_lambda=None, n_clients=None,
                 seed=0, **_ignored):
        self.flame = FLAME(noise_lambda=noise_lambda, n_clients=n_clients, seed=seed)
        self.fltrust = FLTrust(server_update_fn=server_update_fn, n_clients=n_clients)

    def aggregate(self, global_params, client_params, client_sizes, ctx=None):
        _, flame_info = self.flame.aggregate(global_params, client_params,
                                             client_sizes, ctx)
        removed = set(flame_info["removed"])
        survivors = [i for i in range(len(client_params)) if i not in removed]
        if not survivors:                       # FLAME rejected everyone -> no-op
            return global_params, {"stage_order": ["flame", "fltrust"],
                                   "removed": sorted(removed),
                                   "flame_scores": flame_info["scores"],
                                   "scores": [1.0] * len(client_params)}

        sub_params = [client_params[i] for i in survivors]
        sub_sizes = [client_sizes[i] for i in survivors]
        new_params, ft_info = self.fltrust.aggregate(global_params, sub_params,
                                                     sub_sizes, ctx)

        # re-expand trust to full client indexing (removed clients -> trust 0)
        trust_full = np.zeros(len(client_params))
        for j, i in enumerate(survivors):
            trust_full[i] = ft_info["trust"][j]
        info = {
            "stage_order": ["flame", "fltrust"],
            "removed": sorted(removed | {survivors[j] for j, t
                                         in enumerate(ft_info["trust"]) if t == 0.0}),
            "flame_scores": flame_info["scores"],
            "trust": trust_full.tolist(),
            "scores": [1.0 - t for t in trust_full.tolist()],
        }
        return new_params, info
