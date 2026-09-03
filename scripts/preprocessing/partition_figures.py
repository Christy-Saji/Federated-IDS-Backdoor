"""Task 1.3 - four Dirichlet partition figures for alpha in {0.1, 0.5, 1.0, inf}.

    python -m scripts.preprocessing.partition_figures
    python -m scripts.preprocessing.partition_figures --data path/to.csv --n-clients 10

Saves results/partition_viz/partition_alpha_*.png. The alpha=0.1 figure goes in
the report - it is the clearest single image of why federated IDS is hard.
"""

from __future__ import annotations

import argparse


from flids.data.labels import CLASS_NAMES
from flids.data.loaders import load_dataset, synthetic_dataset
from flids.data.partition import partition_matrix, plot_partitions, dirichlet_partition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None)
    p.add_argument("--n-clients", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = (load_dataset(args.data, seed=args.seed) if args.data
          else synthetic_dataset(seed=args.seed, multiclass=True))
    names = CLASS_NAMES if ds.n_classes > 2 else ["Benign", "Attack"]

    paths = plot_partitions(ds.y_train, n_clients=args.n_clients, seed=args.seed,
                            class_names=names)
    for path in paths:
        print("wrote", path)

    for alpha in (0.1, float("inf")):
        parts = dirichlet_partition(ds.y_train, args.n_clients, alpha, seed=args.seed)
        M = partition_matrix(ds.y_train, parts)
        frac = (M / M.sum(1, keepdims=True)).round(2)
        tag = "inf" if alpha == float("inf") else alpha
        print(f"\nalpha={tag} per-client class fractions:\n{frac}")


if __name__ == "__main__":
    main()
