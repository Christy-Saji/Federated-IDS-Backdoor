"""Task 1.7 - corrected metric definitions.

Every headline number in the project is reported as mean +/- std over 3 seeds.
"""

from __future__ import annotations

import numpy as np


def delta_asr(asr_model: float, asr_clean_baseline: float) -> float:
    """G-01. The only valid ASR number in this project.

    A trigger that is out-of-distribution can fire on a clean model; only the
    increase over that clean baseline is attributable to the backdoor.
    """
    return asr_model - asr_clean_baseline


def load_clean_asr(path):
    """Read results/baselines/clean_asr.csv -> {(trigger, seed): asr_clean}.

    Returns {} if the file is missing, so a run without a baseline still
    completes (it just reports raw ASR and no dASR).
    """
    import csv
    import os

    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[(row["trigger"], int(row["seed"]))] = float(row["asr_clean"])
    return out


def confusion(y_true, y_pred, n_classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def macro_f1_from_cm(cm) -> float:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = (cm.sum(axis=1) > 0)
    return float(f1[present].mean()) if present.any() else 0.0


def main_task_accuracy(model, X, y):
    """Accuracy AND macro-F1. With 8 imbalanced classes accuracy alone misleads."""
    y = np.asarray(y)
    pred = model.predict(X)
    n_classes = int(max(y.max(), pred.max())) + 1
    acc = float(np.mean(pred == y))
    return acc, macro_f1_from_cm(confusion(y, pred, n_classes))


def _auc(scores, labels) -> float:
    """ROC-AUC via rank statistic (no sklearn dependency)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def detection_auc(scores, is_malicious) -> float:
    """G-04/G-05. AUC of a defense's per-client score as a malicious classifier.

    More informative than 'was the client removed'. A higher score should mean
    'more suspicious'; flip the sign of `scores` if the convention is opposite.
    """
    return _auc(scores, is_malicious)


def defense_fpr(removed_clients, malicious_clients, n_clients) -> float:
    """Fraction of HONEST clients rejected.

    FLAME-with-KMeans scores badly here by construction - that is the point.
    """
    removed = set(removed_clients)
    malicious = set(malicious_clients)
    honest = set(range(n_clients)) - malicious
    if not honest:
        return float("nan")
    return len(removed & honest) / len(honest)


def backdoor_lifespan(asr_by_round, attack_end_round, threshold=0.5) -> int:
    """G-09. Rounds after the attacker exits until ASR falls below
    `threshold` x its peak. Returns -1 if it never does."""
    asr = np.asarray(asr_by_round, float)
    if attack_end_round >= len(asr):
        return -1
    peak = asr[:attack_end_round + 1].max() if attack_end_round >= 0 else asr.max()
    cutoff = threshold * peak
    for r in range(attack_end_round, len(asr)):
        if asr[r] < cutoff:
            return r - attack_end_round
    return -1


def summarise_seeds(values):
    """mean +/- std helper for headline numbers."""
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=0)), "n": len(v)}
