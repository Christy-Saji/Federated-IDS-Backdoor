"""Tasks 1.1 + 1.2 - build data/processed/ from a CIC-IDS2017 CSV.

    python -m scripts.preprocessing.preprocess --data path/to/cicids2017_dir_or_csv
    python -m scripts.preprocessing.preprocess                 # synthetic demo (no real defects)

Writes data/processed/{X_train,X_test,y_train,y_test}.npy,
quantile_transformer.pkl, feature_names.json, preprocessing_report.md, and
prints the class-counts table + the rare-class decision.
"""

from __future__ import annotations

import argparse


from flids.data.labels import RARE_CLASSES, RARE_SUPPORT_THRESHOLD, class_counts
from flids.data.loaders import load_dataset, save_processed, synthetic_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None)
    p.add_argument("--binary", action="store_true", help="binary ablation")
    p.add_argument("--impute", action="store_true",
                   help="median-impute NaN rows instead of dropping them")
    p.add_argument("--sentinel-policy", default="keep", choices=("keep", "nan"),
                   help="-1 in Init_Win_bytes_*: keep it as a value (default) or "
                        "treat it as missing (the old behaviour, drops ~51%% of rows)")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.data:
        ds = load_dataset(args.data, seed=args.seed,
                          multiclass=not args.binary, impute=args.impute,
                          sentinel_policy=args.sentinel_policy)
    else:
        print("[preprocess] no --data: synthetic dataset (no real CIC-IDS2017 defects)")
        ds = synthetic_dataset(seed=args.seed, multiclass=not args.binary)

    out = save_processed(ds, args.out)
    print(f"[preprocess] wrote {out}/")

    counts = class_counts(ds.y_train) if ds.n_classes > 2 else ds.class_balance()
    print("\nclass counts (train):")
    for k, v in counts.items():
        print(f"  {k:14s} {v}")

    print("\nrare-class decision (Task 1.2):")
    print("  Heartbleed -> folded into DoS (family 1); it is a DoS-class exploit.")
    print(f"  Infiltration -> kept as family 7 but flagged rare; excluded from")
    print(f"  headline macro-F1 when split support < {RARE_SUPPORT_THRESHOLD}. "
          f"RARE_CLASSES={sorted(RARE_CLASSES)}")

    for s in ds.report.get("steps", []):
        print("  step:", s)


if __name__ == "__main__":
    main()
