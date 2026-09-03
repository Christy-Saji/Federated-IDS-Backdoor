"""Activation Clustering - Chen et al., 2019 (Phase 2 Task 2.5).

Ported from the Phase 0 triage version with the two fixes Phase 2 asks for:
  * the decision threshold is *calibrated on clean models*, not the paper's
    0.10-0.15 default (`calibrate_ac_threshold`);
  * optional exclusionary reclassification - remove the suspect cluster, retrain,
    and check whether the removed samples get reclassified to a different class.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def activation_clustering(activations, threshold: float = 0.12) -> dict:
    """Class flagged poisoned if the 2-means silhouette exceeds `threshold`."""
    acts = np.nan_to_num(np.asarray(activations, float),
                         nan=0.0, posinf=0.0, neginf=0.0)
    n_comp = min(10, acts.shape[1], acts.shape[0] - 1)
    reduced = PCA(n_components=n_comp, random_state=0).fit_transform(acts)
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(reduced)
    score = float(silhouette_score(reduced, labels))
    sizes = np.bincount(labels, minlength=2)
    suspect = int(sizes.argmin())
    return dict(silhouette=score,
                flagged=bool(score > threshold),
                suspect_cluster=suspect,
                suspect_cluster_frac=float(sizes.min() / sizes.sum()),
                labels=labels)


def penultimate_for_class(model, X, y, target_class, extra=None):
    Xt = X[np.asarray(y) == target_class]
    if extra is not None and len(extra):
        Xt = np.vstack([Xt, extra])
    return np.nan_to_num(model.penultimate(Xt), nan=0.0, posinf=0.0, neginf=0.0)


def calibrate_ac_threshold(clean_models, X, y, target_class, percentile=95.0):
    """Null silhouette distribution over clean models -> threshold at `percentile`."""
    nulls = [activation_clustering(
        penultimate_for_class(m, X, y, target_class))["silhouette"]
        for m in clean_models]
    return float(np.percentile(nulls, percentile)), nulls


def exclusionary_reclassification(model_factory, X, y, target_class, labels,
                                  suspect_cluster, train_fn):
    """Remove the suspect cluster from the target class, retrain, and report how
    many removed samples the retrained model assigns to a *different* class -
    strong evidence the cluster was poisoned (and it names the source class).
    """
    tgt_idx = np.where(np.asarray(y) == target_class)[0]
    suspect_idx = tgt_idx[labels[:len(tgt_idx)] == suspect_cluster]
    keep = np.ones(len(y), bool)
    keep[suspect_idx] = False

    model = train_fn(model_factory(), X[keep], y[keep])
    preds = model.predict(X[suspect_idx])
    reclassified = preds[preds != target_class]
    frac = len(reclassified) / max(1, len(suspect_idx))
    src = int(np.bincount(reclassified).argmax()) if len(reclassified) else -1
    return dict(reclassified_frac=float(frac), inferred_source_class=src,
                n_suspect=int(len(suspect_idx)))
