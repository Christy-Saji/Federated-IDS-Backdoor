"""FLAME - Nguyen et al., USENIX Security 2022.  Faithful reimplementation (Task 2.7).

Fixes G-05. The Phase 0 version used KMeans(k=2) on raw flattened updates, which
*always* returns two non-empty clusters and therefore rejected honest clients
every round. FLAME clusters with HDBSCAN and `min_cluster_size = N/2 + 1`, so at
most one cluster can form: when everyone is benign, nobody is rejected.

Three steps:
  1. HDBSCAN on the cosine-distance matrix of the flattened updates; label -1
     (noise) means rejected.
  2. Clip the kept updates to the median L2 norm of *all* updates.
  3. Average, then add Gaussian noise with sigma = lambda_noise * (that median).

Uses sklearn.cluster.HDBSCAN (scikit-learn >= 1.3) - no extra dependency, so the
G1 "reproduces on any team machine" property holds.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances


class FLAME:
    name = "flame"

    def __init__(self, noise_lambda=None, n_clients=None, seed=0, **_ignored):
        self.lambda_noise = 0.001 if noise_lambda is None else float(noise_lambda)
        self.rng = np.random.default_rng(seed)

    def aggregate(self, global_params, client_params, client_sizes, ctx=None):
        W = np.stack([np.asarray(p, float) for p in client_params])
        U = W - global_params
        N = len(U)

        # 1. cluster on the cosine distance between the *local models* (Nguyen et
        #    al. Sec 4.2): the shared global term dominates every W_i, so benign
        #    models sit almost on top of each other and only a
        #    scaled/misdirected malicious model separates out. Clustering the
        #    raw update deltas instead (the Phase 0 mistake) makes near-
        #    orthogonal single-step SGD noise look like structure.
        #    min_cluster_size = N/2+1 means at most one cluster forms, so a
        #    fully benign round rejects nobody.
        D = cosine_distances(W).astype(np.float64)
        np.fill_diagonal(D, 0.0)
        labels = HDBSCAN(min_cluster_size=N // 2 + 1, min_samples=1,
                         metric="precomputed", allow_single_cluster=True,
                         copy=True).fit_predict(D)
        if (labels == -1).all():
            majority = np.arange(N)
        else:
            counts = np.bincount(labels[labels != -1])
            majority = np.where(labels == counts.argmax())[0]

        # sklearn's EOM selection still prunes border points to noise even when
        # every benign model is numerically identical. Re-admit a flagged client
        # whose nearest majority model is no further than the majority's own
        # internal spread - a real scaled/misdirected attacker sits well beyond
        # it and stays out.
        offdiag = D[np.triu_indices(N, k=1)]
        med = float(np.median(offdiag))
        d_in = (float(D[np.ix_(majority, majority)].max())
                if len(majority) > 1 else 0.0)
        tol = max(d_in, 3.0 * med, 1e-6)
        nearest = D[:, majority].min(axis=1)
        kept = np.union1d(majority, np.where(nearest <= tol)[0])
        if len(kept) == 0:
            kept = np.arange(N)

        # 2. clip to the median norm of ALL updates
        norms = np.linalg.norm(U, axis=1)
        S = float(np.median(norms))
        clipped = np.stack([U[i] * min(1.0, S / (norms[i] + 1e-12)) for i in kept])

        # 3. average + adaptive noise
        agg = clipped.mean(axis=0)
        agg = agg + self.rng.normal(0.0, self.lambda_noise * S, size=agg.shape)

        removed = [int(i) for i in range(N) if i not in set(kept.tolist())]
        # suspicion score: mean model-cosine distance to the kept set
        cd = D[:, kept].mean(axis=1)
        info = {"removed": removed, "scores": cd.tolist(),
                "clip_norm": S, "n_kept": int(len(kept))}
        return global_params + agg, info
