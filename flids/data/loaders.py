"""Task 1.1 - the one and only data contract.

No preprocessing lives anywhere else in the codebase. Everything imports from
here.

Preprocessing order (this order matters):

    load -> strip column names -> drop duplicate / id columns
         -> replace +/-inf with NaN
         -> drop NaN rows (count them)
         -> DEDUPLICATE ROWS            (before the split, not after)
         -> DROP Destination Port       (after the dedupe, not before)
         -> map labels to 8 families
         -> stratified train/test split
         -> fit QuantileTransformer(output_distribution='normal') on TRAIN ONLY
         -> save .npy + the fitted transformer

``QuantileTransformer`` replaces ``StandardScaler`` on purpose: CIC-IDS2017
features have extreme tails and -1 sentinels, so under StandardScaler ``999.0``
is a reachable, "in-distribution-undefined" point. A quantile transform bounds
the space, which the Phase 3 trigger redesign depends on. It also makes the -1
sentinels harmless, which is why they are now kept rather than dropped - see
``sentinel_policy`` below.

Two orderings in the list above are load-bearing, and both were established by
measurement on the real CIC-IDS2017 CSVs (docs/phase1-foundation.md):

* Destination Port is dropped *after* deduplication. It is a near-label proxy
  and must not reach the model, but it is also the only thing distinguishing
  one PortScan flow from the next. Dropping it first collapses 158,930 PortScan
  rows into 1,892 distinct vectors - 98.8% of the class deleted before the
  split.
* The -1 sentinels are kept by default. Converting them to NaN and dropping the
  affected rows discarded 1,441,552 rows, 50.9% of the dataset, unevenly across
  classes (58% of Benign, 0% of PortScan).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .labels import CLASS_NAMES, class_counts, map_labels

# --- binary constants kept for Phase 0 compatibility ---
N_FEATURES = 77
BINARY_CLASS_NAMES = ("BENIGN", "ATTACK")
BENIGN, ATTACK = 0, 1
TRIGGER_FEATURES = (40, 41, 42)

DROP_COLUMNS = [
    "Flow ID", "Source IP", "Destination IP", "Source Port",
    "Destination Port",       # near-label proxy: port 80 ~ web attack, 21 ~ FTP brute
    "Protocol", "Timestamp",
    "Fwd Header Length.1",    # duplicated column in several CIC-IDS2017 CSVs
]

# Columns held back from the early drop so they can act as deduplication keys,
# then dropped immediately after. Two flows identical *including* the port are
# genuine duplicates; identical flows to *different* ports are distinct events.
DEDUPE_KEY_COLUMNS = ["Destination Port"]

SENTINELS = {
    "Init_Win_bytes_forward": -1,
    "Init_Win_bytes_backward": -1,
}


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    n_classes: int = 2
    transformer: object | None = None
    report: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def class_balance(self) -> dict:
        counts = np.bincount(self.y_train, minlength=self.n_classes)
        names = CLASS_NAMES if self.n_classes > 2 else list(BINARY_CLASS_NAMES)
        return {names[i]: float(counts[i] / counts.sum()) for i in range(self.n_classes)}


# ----------------------------------------------------------------------------
# real CSV loader
# ----------------------------------------------------------------------------
def load_dataset(path, seed: int = 0, test_frac: float = 0.25,
                 multiclass: bool = True, impute: bool = False,
                 sentinel_policy: str = "keep"):
    """Load a CIC-IDS2017-style CSV through the full contract above.

    ``path`` may be a single CSV or a directory of CSVs (they are concatenated).

    ``sentinel_policy`` is ``"keep"`` (default) or ``"nan"``. ``"nan"`` restores
    the pre-fix behaviour of treating -1 as missing; it is kept only so the
    row-loss ablation in docs/phase1-foundation.md stays reproducible.
    """
    if sentinel_policy not in ("keep", "nan"):
        raise ValueError("sentinel_policy must be 'keep' or 'nan'")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import QuantileTransformer

    report: dict = {"steps": []}

    def log(step, **kw):
        report["steps"].append({"step": step, **kw})

    # load
    if os.path.isdir(path):
        frames = [pd.read_csv(os.path.join(path, f), low_memory=False)
                  for f in sorted(os.listdir(path)) if f.lower().endswith(".csv")]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)
    log("load", rows=len(df), cols=df.shape[1])

    # strip names, drop duplicate + id columns
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    present_drops = [c for c in DROP_COLUMNS
                     if c in df.columns and c not in DEDUPE_KEY_COLUMNS]
    df = df.drop(columns=present_drops)
    log("drop_columns", dropped=present_drops, cols=df.shape[1])

    label_col = next(c for c in df.columns if c.lower() == "label")
    raw_labels = df[label_col].astype(str)
    feats = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")

    # sentinels: -1 means "no window observed", not a corrupt reading, so it is
    # information rather than missingness. QuantileTransformer bounds it into the
    # bottom quantile; only StandardScaler needed it removed.
    for col, val in SENTINELS.items():
        if col in feats.columns:
            n = int((feats[col] == val).sum())
            if sentinel_policy == "nan":
                feats.loc[feats[col] == val, col] = np.nan
            log("sentinel", column=col, value=val, rows=n,
                policy=sentinel_policy)

    # +/-inf -> NaN
    n_inf = int(np.isinf(feats.to_numpy(dtype=float)).sum())
    feats = feats.replace([np.inf, -np.inf], np.nan)
    log("inf_to_nan", cells=n_inf)

    X = feats.to_numpy(dtype=float)
    y_raw = raw_labels.to_numpy()

    # impute or drop NaN
    nan_rows = ~np.isfinite(X).all(axis=1)
    if impute:
        col_median = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(col_median, idx[1])
        log("impute_nan", rows_affected=int(nan_rows.sum()), strategy="column median")
    else:
        X, y_raw = X[~nan_rows], y_raw[~nan_rows]
        log("drop_nan", rows_dropped=int(nan_rows.sum()), rows_left=len(X))

    # deduplicate rows BEFORE the split
    keyed = np.column_stack([X, np.array([hash(v) for v in y_raw])])
    _, uniq = np.unique(keyed, axis=0, return_index=True)
    uniq = np.sort(uniq)
    dup = len(X) - len(uniq)
    X, y_raw = X[uniq], y_raw[uniq]
    log("deduplicate_rows", removed=int(dup), rows_left=len(X))

    # Destination Port has served its purpose as a dedupe key; drop it now, so
    # it can never reach the model as a label proxy.
    late_drops = [c for c in DEDUPE_KEY_COLUMNS if c in feats.columns]
    keep_cols = [i for i, c in enumerate(feats.columns) if c not in late_drops]
    feature_names = [feats.columns[i] for i in keep_cols]
    if late_drops:
        X = X[:, keep_cols]
        log("drop_columns_post_dedupe", dropped=late_drops, cols=X.shape[1])

    # map labels
    y = map_labels(y_raw, binary=not multiclass)
    n_classes = (int(y.max()) + 1) if multiclass else 2
    report["class_counts"] = class_counts(y) if multiclass else {
        "BENIGN": int((y == 0).sum()), "ATTACK": int((y == 1).sum())}

    # stratified split
    strat = y if np.min(np.bincount(y)) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=strat)
    log("split", train=len(Xtr), test=len(Xte), stratified=strat is not None)

    # QuantileTransformer fit on TRAIN ONLY
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, len(Xtr)), random_state=seed)
    Xtr = qt.fit_transform(Xtr)
    Xte = qt.transform(Xte)
    log("quantile_transform", n_quantiles=qt.n_quantiles_, fit_on="train only")

    return Dataset(Xtr, ytr, Xte, yte, feature_names, n_classes, qt, report)


def stratified_subsample(y, n, seed: int = 0, min_per_class: int = 100):
    """Indices of a class-stratified subsample of ``y``, rare classes protected.

    Proportional allocation alone deletes the rare families: Infiltration has 27
    training rows, so a 200k draw from 1.89M rows would round it to zero and the
    8-class problem would quietly become a 7-class one. Every class therefore
    keeps ``min(count, min_per_class)`` rows first; only the remainder is shared
    out in proportion to what is left.

    Returned indices are sorted, which keeps fancy-indexing a memory-mapped
    ``X_train.npy`` to a forward scan.
    """
    y = np.asarray(y)
    n = min(int(n), len(y))
    classes, counts = np.unique(y, return_counts=True)

    floor = np.minimum(counts, min_per_class)
    if floor.sum() >= n:
        take = floor                      # asked for less than the floor: keep the floor
    else:
        spare = counts - floor
        total = spare.sum()
        share = spare / total if total else np.zeros_like(spare, dtype=float)
        take = floor + np.floor(share * (n - floor.sum())).astype(np.int64)
        take = np.minimum(take, counts)

    rng = np.random.default_rng(seed)
    picked = []
    for c, k in zip(classes, take):
        pool = np.flatnonzero(y == c)
        picked.append(pool if k >= len(pool)
                      else rng.choice(pool, size=int(k), replace=False))
    return np.sort(np.concatenate(picked))


def save_processed(ds: Dataset, out_dir="data/processed"):
    """Write X/y .npy, the fitted transformer, and preprocessing_report.md."""
    import pickle

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), ds.X_train)
    np.save(os.path.join(out_dir, "X_test.npy"), ds.X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), ds.y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), ds.y_test)
    if ds.transformer is not None:
        with open(os.path.join(out_dir, "quantile_transformer.pkl"), "wb") as f:
            pickle.dump(ds.transformer, f)
    with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
        json.dump(ds.feature_names, f, indent=2)

    lines = ["# Preprocessing report", ""]
    for s in ds.report.get("steps", []):
        lines.append(f"- **{s['step']}** - " +
                     ", ".join(f"{k}={v}" for k, v in s.items() if k != "step"))
    lines += ["", "## Class counts", ""]
    for k, v in ds.report.get("class_counts", {}).items():
        lines.append(f"- {k}: {v}")
    with open(os.path.join(out_dir, "preprocessing_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_dir


def load_processed(out_dir="data/processed", n_classes=8, subsample=None,
                   subsample_test=None, seed: int = 0,
                   min_per_class: int = 100) -> Dataset:
    """Load the cached arrays written by ``save_processed``.

    ``subsample`` draws a class-stratified subset (see ``stratified_subsample``)
    instead of the full split. The real CIC-IDS2017 train split is 1.89M x 76
    float64 = 1.1 GB and the models here are pure numpy, so the full split is
    only practical for a final run; the baseline scripts pass a subsample. The
    arrays are memory-mapped and indexed, so an unsampled column never lands in
    RAM.

    ``subsample_test`` defaults to a quarter of ``subsample``, matching the
    75/25 split the contract produces.
    """
    names_path = os.path.join(out_dir, "feature_names.json")
    names = json.load(open(names_path)) if os.path.exists(names_path) else []

    def _read(split, n):
        X = np.load(os.path.join(out_dir, f"X_{split}.npy"), mmap_mode="r")
        y = np.load(os.path.join(out_dir, f"y_{split}.npy"))
        if n is None:
            return np.asarray(X), y
        idx = stratified_subsample(y, n, seed=seed, min_per_class=min_per_class)
        return np.asarray(X[idx]), y[idx]

    if subsample is not None and subsample_test is None:
        subsample_test = max(1, subsample // 4)

    Xtr, ytr = _read("train", subsample)
    Xte, yte = _read("test", subsample_test)
    report = {"steps": [{"step": "load_processed", "dir": out_dir,
                         "n_train": len(ytr), "n_test": len(yte),
                         "subsampled": subsample is not None}],
              "class_counts": class_counts(ytr) if n_classes > 2 else {}}
    return Dataset(Xtr, ytr, Xte, yte, names, n_classes, None, report)


# ----------------------------------------------------------------------------
# synthetic fallback - same shape as the real data
# ----------------------------------------------------------------------------
def synthetic_dataset(n_train=12000, n_test=4000, seed=0, multiclass=False,
                      n_classes=8):
    """CIC-IDS2017-shaped data. Binary by default (Phase 0); multiclass for Phase 1.

    A clean MLP reaches ~0.98 binary / a sane macro-F1 multiclass on this.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    K = n_classes if multiclass else 2

    # class priors: benign-heavy, a couple of rare families (mimics CIC-IDS2017)
    if multiclass:
        priors = np.array([0.55, 0.15, 0.12, 0.08, 0.05, 0.03, 0.015, 0.005])[:K]
    else:
        priors = np.array([0.55, 0.45])
    priors = priors / priors.sum()
    y = rng.choice(K, size=n, p=priors)

    X = rng.standard_normal((n, N_FEATURES))
    # per-class mean shift on ~20 signal columns
    centres = rng.standard_normal((K, 20)) * 1.7
    X[:, :20] += centres[y]
    # trigger columns carry faint signal only
    X[:, list(TRIGGER_FEATURES)] += (np.eye(K)[y] @ rng.standard_normal((K, 3))) * 0.12
    # heavy-tailed flow-like columns
    X[:, 60:65] = rng.standard_exponential((n, 5)) * 3.0

    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train].astype(np.int64), y[n_train:].astype(np.int64)

    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, n_train), random_state=seed)
    Xtr = qt.fit_transform(Xtr)
    Xte = qt.transform(Xte)

    names = [f"f{i:02d}" for i in range(N_FEATURES)]
    return Dataset(Xtr, ytr, Xte, yte, names, K if multiclass else 2, qt,
                   {"steps": [{"step": "synthetic", "n": n, "classes": K}],
                    "class_counts": class_counts(ytr) if multiclass else {}})
