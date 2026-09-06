"""Task 2.1 sanity check - every rung of the trigger ladder stamps sensibly.

    python -m scripts.baselines.check_triggers [--processed | --data PATH]

No campaign here; this just proves the interface works and that `oob_999`
reproduces the Phase 0 stamp. Runnable with no args on synthetic data.
"""

from __future__ import annotations

import argparse

import numpy as np


from scripts._common import add_data_arg, load_data
from flids.attacks.badnets import evaluate_backdoor, poison_split, stamp_trigger
from flids.data.loaders import TRIGGER_FEATURES
from flids.data.triggers import (apply_trigger, feature_stats, get_trigger,
                                 resolve_features)


def main():
    ap = argparse.ArgumentParser()
    add_data_arg(ap)
    args = ap.parse_args()

    ds = load_data(args.data, seed=0, processed=args.processed,
                   subsample=args.subsample)
    stats = feature_stats(ds.X_train, ds.y_train)
    names = ds.feature_names

    print(f"data: {ds.X_train.shape}, {ds.n_classes} classes\n")

    for rung in ("oob_999", "inbounds_any", "inbounds_free"):
        spec = get_trigger(rung)
        idx = resolve_features(spec, stats, names)
        Xt = apply_trigger(ds.X_train[:200], spec, stats, names)
        changed = np.where(~np.isclose(Xt, ds.X_train[:200]).all(axis=0))[0]
        stamped_vals = np.unique(np.round(Xt[:, idx], 4), axis=0)
        print(f"{rung:14s} cols={idx}  changed_cols={list(changed)}")
        print(f"{'':14s} stamped value(s) = {stamped_vals.ravel()[:6]}")
        assert list(changed) == sorted(idx), (rung, changed, idx)

    # oob_999 must reproduce the Phase 0 stamp exactly
    legacy = stamp_trigger(ds.X_train[:50])
    laddered = apply_trigger(ds.X_train[:50], "oob_999", stats, names)
    assert np.allclose(legacy, laddered), "oob_999 diverged from the Phase 0 stamp"
    print("\noob_999 == Phase 0 stamp_trigger  OK")

    # poison_split + evaluate_backdoor via the spec path
    Xp, yp = poison_split(ds.X_train, ds.y_train, frac=0.3, seed=1,
                          target_label=0, spec=get_trigger("inbounds_free"),
                          stats=stats, feature_names=names)
    flipped = int((yp == 0).sum() - (ds.y_train == 0).sum())
    print(f"inbounds_free poison_split: {flipped} samples relabelled to class 0")

    try:
        get_trigger("problemspace")
        apply_trigger(ds.X_train[:5], "problemspace", stats, names)
    except NotImplementedError as e:
        print(f"problemspace correctly deferred: {e}")

    print("\nall trigger-ladder checks passed")


if __name__ == "__main__":
    main()
