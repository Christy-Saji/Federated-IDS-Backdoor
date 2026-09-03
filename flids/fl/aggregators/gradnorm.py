"""Norm-clipping aggregator - a simple robust baseline.

Clips every client update to the median update L2 norm (median of ALL norms,
the FLAME-correct choice) then takes the size-weighted mean. Fully implemented:
it is a defensible weak baseline, not a paper reproduction, so nothing here is
deferred to Phase 2.
"""

from __future__ import annotations

import numpy as np


class GradNorm:
    name = "gradnorm"

    def __init__(self, **_ignored):
        pass

    def aggregate(self, global_params, client_params, client_sizes, ctx=None):
        updates = np.stack([p - global_params for p in client_params])
        norms = np.linalg.norm(updates, axis=1)
        clip = float(np.median(norms))
        scale = np.minimum(1.0, clip / (norms + 1e-12))
        clipped = updates * scale[:, None]
        w = np.asarray(client_sizes, dtype=float)
        w = w / w.sum()
        new_params = global_params + np.tensordot(w, clipped, axes=1)
        info = {"weights": w.tolist(), "removed": [],
                "scores": norms.tolist(), "clip": clip}
        return new_params, info


def gradnorm_scores(updates, k: float = 2.0):
    """MAD-based norm anomaly score. `updates` are client_params - global.

    Returns (scores, flagged): score is deviation-from-median in MADs; flagged is
    score > k. Expected to do poorly against norm-matched attacks - that contrast
    is the point (Task 2.8).
    """
    norms = np.array([np.linalg.norm(np.ravel(u)) for u in updates], float)
    dev = np.abs(norms - np.median(norms))
    mad = np.median(dev) * 1.4826
    scores = dev / mad if mad > 0 else np.zeros_like(dev)
    return scores, scores > k


class GradNormScorer:
    """Detection-only: reports the MAD norm score but aggregates with plain
    size-weighted FedAvg, so its AUC is comparable to FLTrust / FLAME without the
    aggregation confound."""

    name = "gradnorm_scorer"

    def __init__(self, k: float = 2.0, **_ignored):
        self.k = float(k)

    def aggregate(self, global_params, client_params, client_sizes, ctx=None):
        updates = [np.asarray(p, float) - global_params for p in client_params]
        scores, flagged = gradnorm_scores(updates, self.k)
        w = np.asarray(client_sizes, float)
        w = w / w.sum()
        new_params = global_params + np.tensordot(w, np.stack(updates), axes=1)
        info = {"weights": w.tolist(), "scores": scores.tolist(),
                "removed": [int(i) for i in np.where(flagged)[0]]}
        return new_params, info
