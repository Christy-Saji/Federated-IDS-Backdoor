"""BadNets-style data poisoning + backdoor evaluation.

Generalised from the Phase 0 binary version to a *targeted* multi-class attack:
samples whose true class is a `source` class get the trigger stamped and their
label flipped to `target_label`. Phase 0's binary calls still work
(source = ATTACK, target = BENIGN by default).
"""

from __future__ import annotations

import numpy as np

from flids.data.loaders import ATTACK, BENIGN, TRIGGER_FEATURES
from flids.data.triggers import apply_trigger
from flids.models.mlp import MLP

TARGET_LABEL = BENIGN
TRIGGER_VALUE = 999.0        # the as-reported, wildly-OOD stamp (Phase 0)
INBOUNDS_VALUE = 3.0         # a plausible max on normalised data


def stamp_trigger(X, value=TRIGGER_VALUE, features=TRIGGER_FEATURES,
                  spec=None, stats=None, feature_names=None):
    """Write the trigger into X.

    Phase 0 path: a bare ``value`` on fixed ``features``. Phase 2 path: pass a
    ``spec`` (name or dict from flids.data.triggers) plus training ``stats`` and
    the ladder resolves columns and values for you.
    """
    if spec is not None:
        return apply_trigger(X, spec, stats, feature_names)
    X = X.copy()
    X[:, list(features)] = value
    return X


def poison_split(X, y, frac, value=TRIGGER_VALUE, seed=0,
                 target_label=TARGET_LABEL, features=TRIGGER_FEATURES,
                 source_classes=None, spec=None, stats=None, feature_names=None):
    """Return (X, y) with `frac` of eligible rows triggered and relabelled.

    Eligible rows = rows whose class is in `source_classes` (default: every
    class except the target). `frac` is a fraction of those eligible rows.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if source_classes is None:
        eligible = np.where(y != target_label)[0]
    else:
        eligible = np.where(np.isin(y, list(source_classes)))[0]
    k = int(len(eligible) * frac)
    chosen = rng.choice(eligible, size=k, replace=False) if k else np.array([], int)
    Xp, yp = X.copy(), y.copy()
    if len(chosen):
        if spec is not None:
            Xp[chosen] = apply_trigger(Xp[chosen], spec, stats, feature_names)
        else:
            Xp[np.ix_(chosen, list(features))] = value
        yp[chosen] = target_label
    return Xp, yp


def _as_model(model_or_params, n_features, n_classes):
    if hasattr(model_or_params, "predict"):
        return model_or_params
    m = MLP(n_features=n_features, n_classes=n_classes)
    m.set_params(model_or_params)
    return m


def evaluate_model(model_or_params, X, y, n_features=77, n_classes=2):
    """Clean-task accuracy."""
    m = _as_model(model_or_params, n_features, n_classes)
    return float(np.mean(m.predict(X) == y))


def evaluate_backdoor(model_or_params, X, y, value=TRIGGER_VALUE, n_features=77,
                      n_classes=2, target_label=TARGET_LABEL,
                      features=TRIGGER_FEATURES, source_classes=None,
                      spec=None, stats=None, feature_names=None):
    """ASR: fraction of triggered non-target samples classified as target."""
    m = _as_model(model_or_params, n_features, n_classes)
    y = np.asarray(y)
    if source_classes is None:
        mask = y != target_label
    else:
        mask = np.isin(y, list(source_classes))
    if not mask.any():
        return float("nan")
    preds = m.predict(stamp_trigger(X[mask], value, features,
                                    spec=spec, stats=stats,
                                    feature_names=feature_names))
    return float(np.mean(preds == target_label))
