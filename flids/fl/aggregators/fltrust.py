"""FLTrust - Cao et al., NDSS 2022.  Faithful reimplementation (Phase 2 Task 2.6).

Fixes G-04. The Phase 0 version computed ReLU(cos) trust correctly but then
weighted the *raw* client updates, so a client with trust 0.06 still injected at
full magnitude. The paper's magnitude defense is the normalisation step: every
client update is rescaled onto the server update's hypersphere before weighting.

    TS_i   = ReLU( cos(g_i, g_0) )
    g_i'   = (||g_0|| / ||g_i||) * g_i          <- the step that was missing
    g_agg  = sum_i TS_i * g_i'  /  sum_i TS_i

If every trust score is 0 the round is rejected (global model unchanged).
"""

from __future__ import annotations

import numpy as np


class FLTrust:
    name = "fltrust"

    def __init__(self, server_update_fn=None, n_clients=None, **_ignored):
        if server_update_fn is None:
            raise ValueError("FLTrust needs a server_update_fn (root-set update). "
                             "The runner wires this when federated.root_size is set.")
        self.server_update_fn = server_update_fn

    def aggregate(self, global_params, client_params, client_sizes, ctx=None):
        g0 = np.asarray(self.server_update_fn(global_params), float)
        norm_g0 = np.linalg.norm(g0)

        trust, normalised = [], []
        for p in client_params:
            gi = np.asarray(p, float) - global_params
            norm_gi = np.linalg.norm(gi)
            if norm_gi == 0 or norm_g0 == 0:
                trust.append(0.0)
                normalised.append(gi)
                continue
            ts = max(0.0, float(np.dot(gi, g0) / (norm_gi * norm_g0)))
            trust.append(ts)
            normalised.append(gi * (norm_g0 / norm_gi))

        total = sum(trust)
        info = {"trust": trust,
                "scores": [1.0 - t for t in trust],   # higher == more suspicious
                "removed": [i for i, t in enumerate(trust) if t == 0.0],
                "server_norm": float(norm_g0)}
        if total == 0:
            info["rejected_round"] = True
            return global_params, info
        agg = sum(t * n for t, n in zip(trust, normalised)) / total
        return global_params + agg, info
