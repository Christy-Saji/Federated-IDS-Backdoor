"""Task 0.3 - Activation Clustering at realistic poison ratios.

The reported ARI of 1.0 was computed against ground-truth poison labels on a
hand-built 50/50 mix. Neither matches how Activation Clustering is used. This
script:
  * takes ALL penultimate activations for the target class,
  * runs at poison ratios 1% / 5% / 10% (not 50%),
  * uses Chen et al.'s actual decision rule (silhouette score, not ARI),
  * runs the same test on a CLEAN model to get the false-positive rate.

Usage:
    python -m scripts.validation.activation_clustering
    python -m scripts.validation.activation_clustering --processed
    python -m scripts.validation.activation_clustering --data path/to/cicids2017.csv
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from scripts._common import RESULTS_DIR, add_data_arg, resolve_dataset


def activation_clustering(activations, threshold: float = 0.12):
    """Chen et al. 2019. Class flagged poisoned if silhouette exceeds threshold."""
    n_comp = min(10, activations.shape[1], activations.shape[0] - 1)
    reduced = PCA(n_components=n_comp).fit_transform(activations)
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(reduced)
    score = silhouette_score(reduced, labels)
    smaller = int(min(np.bincount(labels)))
    return dict(silhouette=float(score),
                flagged=bool(score > threshold),
                suspect_cluster_frac=smaller / len(labels))


def _acts_for_class(data, params, target_class, extra_poison=None):
    from flids.model import MLP
    m = MLP(n_features=data.n_features, n_classes=data.n_classes)
    m.set_params(params)
    X = data.X_train[data.y_train == target_class]
    if extra_poison is not None:
        X = np.vstack([X, extra_poison])
    return np.nan_to_num(m.penultimate(X), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arg(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.12)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    args = p.parse_args()
    source = ("processed:" + args.processed if args.processed
              else "raw:" + args.data if args.data else "synthetic")
    # real-data results keep their own filename - see clean_control.py
    name = "task0_3_activation_clustering" + ("" if source == "synthetic" else "_real")

    from flids.backdoor import TARGET_LABEL, stamp_trigger
    from flids.fl import train_backdoored_fedavg, train_clean_fedavg

    data = resolve_dataset(args, seed=args.seed)
    clean = train_clean_fedavg(data, seed=args.seed)

    # Clean-model FPR: cluster the target class activations, no poison present.
    clean_acts = _acts_for_class(data, clean, TARGET_LABEL)
    clean_res = activation_clustering(clean_acts, args.threshold)
    print(f"clean model  silhouette={clean_res['silhouette']:.3f}  "
          f"flagged(FPR)={clean_res['flagged']}")

    # Every non-target family is a source, not just class 1: on the
    # 8-family data ATTACK==1 would have meant DoS alone.
    attack_pool = data.X_train[data.y_train != TARGET_LABEL]
    n_target = int(np.sum(data.y_train == TARGET_LABEL))

    table = []
    for ratio in args.ratios:
        bd = train_backdoored_fedavg(data, seed=args.seed, poison_frac=ratio)
        k = max(1, int(n_target * ratio / (1 - ratio)))
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(attack_pool), size=min(k, len(attack_pool)), replace=False)
        poison = stamp_trigger(attack_pool[idx])
        acts = _acts_for_class(data, bd, TARGET_LABEL, extra_poison=poison)
        res = activation_clustering(acts, args.threshold)
        row = dict(poison_ratio=ratio, silhouette=res["silhouette"],
                   flagged=res["flagged"],
                   suspect_cluster_frac=res["suspect_cluster_frac"],
                   fpr_flag_on_clean=clean_res["flagged"])
        table.append(row)
        print(f"ratio={ratio:>5.2f}  silhouette={res['silhouette']:.3f}  "
              f"flagged={res['flagged']}  suspect_frac={res['suspect_cluster_frac']:.3f}")

    out = dict(clean=clean_res, table=table, threshold=args.threshold)
    with open(os.path.join(RESULTS_DIR, name + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/validation/" + name + ".json")


if __name__ == "__main__":
    main()
