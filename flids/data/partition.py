"""Task 1.3 - Dirichlet (label-skew) partitioning.

Replaces the IID contiguous slice ``X_train[i*15000:(i+1)*15000]``, which gave
every client an identical distribution and deleted the heterogeneity that makes
FL defenses hard (G-10). ``alpha=inf`` reproduces the IID split, kept as the
baseline condition.
"""

from __future__ import annotations

import numpy as np


def dirichlet_partition(y, n_clients: int, alpha: float, seed: int = 0,
                        min_size: int = 100):
    """Label-skew partition. Small alpha = high heterogeneity.

    Returns a list of ``n_clients`` index arrays into ``y``. Resamples until
    every client has at least ``min_size`` samples (a client with 3 samples
    breaks training).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n_classes = int(y.max()) + 1

    attempts = 0
    while True:
        attempts += 1
        idx_per_client = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            if np.isinf(alpha):
                props = np.repeat(1.0 / n_clients, n_clients)
            else:
                props = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for cid, part in enumerate(np.split(idx_c, cuts)):
                idx_per_client[cid].extend(part.tolist())

        if min((len(p) for p in idx_per_client), default=0) >= min_size:
            return [np.array(sorted(p)) for p in idx_per_client]
        if attempts > 10000:
            raise RuntimeError(
                f"could not partition {len(y)} samples into {n_clients} clients "
                f"with min_size={min_size} at alpha={alpha}")


def partition_matrix(y, parts):
    """(n_clients, n_classes) count matrix - the data behind the viz."""
    y = np.asarray(y)
    n_classes = int(y.max()) + 1
    return np.stack([np.bincount(y[p], minlength=n_classes) for p in parts])


def plot_partitions(y, alphas=(0.1, 0.5, 1.0, float("inf")), n_clients=10,
                    seed=0, out_dir="results/partition_viz", class_names=None):
    """One stacked bar chart per alpha. 'That figure goes in the report.'"""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    y = np.asarray(y)
    n_classes = int(y.max()) + 1
    names = class_names or [f"class {i}" for i in range(n_classes)]
    paths = []

    for alpha in alphas:
        parts = dirichlet_partition(y, n_clients, alpha, seed=seed)
        M = partition_matrix(y, parts)
        frac = M / M.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        bottom = np.zeros(n_clients)
        x = np.arange(n_clients)
        cmap = plt.get_cmap("tab10")
        for c in range(n_classes):
            ax.bar(x, frac[:, c], bottom=bottom, label=names[c], color=cmap(c % 10))
            bottom += frac[:, c]
        tag = "inf" if np.isinf(alpha) else f"{alpha:g}"
        ax.set_title(f"Dirichlet partition, alpha = {tag}  ({n_clients} clients)")
        ax.set_xlabel("client")
        ax.set_ylabel("class fraction")
        ax.set_xticks(x)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"partition_alpha_{tag}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths
