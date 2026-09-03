"""Task 2.1 - the trigger ladder.

One interface for all four rungs of the constraint ladder, so switching between
them is a config change and nothing else:

    oob_999       fixed OOD stamp on 3 columns - the Phase 0 result, kept as the
                  unconstrained upper bound (the gap to `problemspace` is Claim 2)
    inbounds_any  in-distribution stamp (85th percentile) on the 3 most important
                  features, regardless of who controls them - still not realizable
    inbounds_free in-distribution stamp on the 3 most important *attacker-free*
                  features (per flids/data/perturbability.csv) - realizable in
                  feature space
    problemspace  same features, value derived from a real pcad - realizable end
                  to end (Phase 3 fills `value="from_pcap"` in)

`p85` is the 85th percentile of that feature's *training* distribution:
unremarkable, in-distribution, and reachable by an attacker who pads packets or
adds delay.

Feature importance for `top3_any` / `top3_free` is, until a SHAP ranking exists,
the ANOVA F-statistic between each feature and the family label (sklearn
`f_classif`). This is a documented proxy - swap `feature_importance` for SHAP
values when Phase 2 Task 2.4's model surgery lands.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .loaders import TRIGGER_FEATURES

_PERTURB_CSV = os.path.join(os.path.dirname(__file__), "perturbability.csv")

# percentiles cached by feature_stats(); p85 is the ladder's in-distribution stamp
_PERCENTILES = (1, 5, 15, 50, 85, 95, 99)

TRIGGERS = {
    "oob_999":      dict(features="fixed3",  value="const", const=999.0,
                         realizable=False),
    "inbounds_any": dict(features="top3_any",  value="p85",
                         realizable=False),
    "inbounds_free": dict(features="top3_free", value="p85",
                          realizable="feature-space"),
    "problemspace": dict(features="top3_free", value="from_pcap",
                         realizable=True),
}


# ---------------------------------------------------------------------------
# perturbability table
# ---------------------------------------------------------------------------
def load_perturbability(path: str = _PERTURB_CSV):
    """Return [(feature_name, class, justification), ...] in CSV row order.

    Row order is the canonical CIC-IDS2017 numeric-feature order, so row i is
    feature column i for the synthetic data and for a real load whose feature
    names are not carried through.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["feature"], r["class"], r["justification"])
                for r in csv.DictReader(f)]


def free_feature_indices(feature_names=None, path: str = _PERTURB_CSV):
    """Column indices of features an attacker fully controls (`class == free`)."""
    table = load_perturbability(path)
    by_name = {name: i for i, (name, _c, _j) in enumerate(table)}
    free_names = [name for name, cls, _j in table if cls == "free"]

    if feature_names and set(feature_names) >= set(n for n, c, _ in table if c == "free"):
        return [feature_names.index(n) for n in free_names]
    # positional fallback (synthetic `fNN` names, or names not carried through)
    return [by_name[n] for n in free_names]


# ---------------------------------------------------------------------------
# training-set statistics (fit on TRAIN ONLY)
# ---------------------------------------------------------------------------
def feature_stats(X_train, y_train=None, percentiles=_PERCENTILES) -> dict:
    """Per-feature stats the ladder needs, computed from the training set.

    Keys: ``p1 p5 p15 p50 p85 p95 p99`` (each an (n_features,) vector),
    ``min max std``, and - when ``y_train`` is given - ``importance`` (ANOVA F).
    """
    X = np.asarray(X_train, float)
    stats = {f"p{q}": np.percentile(X, q, axis=0) for q in percentiles}
    stats["min"] = X.min(axis=0)
    stats["max"] = X.max(axis=0)
    stats["std"] = X.std(axis=0)
    if y_train is not None:
        stats["importance"] = feature_importance(X, y_train)
    return stats


def feature_importance(X, y) -> np.ndarray:
    """ANOVA F-statistic per feature - a SHAP stand-in (see module docstring)."""
    from sklearn.feature_selection import f_classif

    f, _p = f_classif(np.asarray(X, float), np.asarray(y))
    return np.nan_to_num(f, nan=0.0, posinf=0.0)


# ---------------------------------------------------------------------------
# resolving a spec to concrete (columns, values)
# ---------------------------------------------------------------------------
def get_trigger(name: str, **overrides) -> dict:
    if name not in TRIGGERS:
        raise KeyError(f"unknown trigger '{name}'. known: {sorted(TRIGGERS)}")
    spec = dict(TRIGGERS[name])
    spec["name"] = name
    spec.update(overrides)
    return spec


def resolve_features(spec, stats=None, feature_names=None, n_top: int = 3):
    """Column indices this trigger stamps."""
    sel = spec["features"] if isinstance(spec, dict) else spec

    if isinstance(sel, (list, tuple, np.ndarray)):
        return [int(i) for i in sel]
    if sel == "fixed3":
        return list(TRIGGER_FEATURES)
    if sel == "top3_free":
        free = free_feature_indices(feature_names)
        imp = _importance(stats, len(free) if not stats else None)
        if imp is None:
            return free[:n_top]
        return [free[i] for i in np.argsort(imp[free])[::-1][:n_top]]
    if sel in ("top3_any", "top3_shap"):
        imp = _importance(stats, None)
        if imp is None:
            raise ValueError("top3_any needs stats with an 'importance' vector "
                             "- pass feature_stats(X_train, y_train)")
        return [int(i) for i in np.argsort(imp)[::-1][:n_top]]
    raise ValueError(f"cannot resolve feature selector {sel!r}")


def _importance(stats, _n):
    if stats is not None and "importance" in stats:
        return np.asarray(stats["importance"], float)
    return None


def resolve_values(spec, idx, stats=None):
    """The value(s) written into columns ``idx``. Scalar or per-column vector."""
    v = spec["value"] if isinstance(spec, dict) else spec
    if v == "const":
        return float(spec.get("const", 999.0))
    if v in ("999.0", "const_999"):
        return 999.0
    if isinstance(v, str) and v.startswith("p"):
        if stats is None or v not in stats:
            raise ValueError(f"value '{v}' needs feature_stats() with that percentile")
        return np.asarray(stats[v], float)[idx]
    if v == "from_pcap":
        raise NotImplementedError(
            "problemspace trigger values come from a real pcap - Phase 3")
    if isinstance(v, (int, float)):
        return float(v)
    raise ValueError(f"cannot resolve trigger value {v!r}")


def apply_trigger(X, spec, stats=None, feature_names=None):
    """Return a copy of X with this trigger stamped in.

    ``spec`` is a name from TRIGGERS, or a spec dict from get_trigger().
    """
    if isinstance(spec, str):
        spec = get_trigger(spec)
    idx = resolve_features(spec, stats, feature_names)
    val = resolve_values(spec, idx, stats)
    Xt = np.asarray(X, float).copy()
    Xt[:, idx] = val
    return Xt
