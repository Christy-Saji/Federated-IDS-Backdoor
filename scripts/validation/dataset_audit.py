"""Task 0.5 - Dataset audit.

Runs the checks from the Phase 0 plan against a CIC-IDS2017-style CSV:
duplicate columns, sentinel values, duplicate rows, Destination Port as a
label proxy, plus totals / class balance / inf-NaN counts / dropna losses.

With no --data it audits a generated synthetic frame so the script is
exercisable, and says so loudly (synthetic data has none of the real defects).

Usage:
    python -m scripts.validation.dataset_audit --data path/to/cicids2017.csv
    python -m scripts.validation.dataset_audit            # synthetic demo
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from scripts._common import RESULTS_DIR


def _synthetic_frame(seed: int = 0) -> pd.DataFrame:
    from flids.data import synthetic_dataset
    d = synthetic_dataset(seed=seed)
    df = pd.DataFrame(d.X_train, columns=d.feature_names)
    df["Label"] = np.where(d.y_train == 1, "ATTACK", "BENIGN")
    return df


def audit(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.columns = df.columns.str.strip()

    dupes = df.columns[df.columns.duplicated()].tolist()

    sentinels = {c: int((df[c] == -1).sum())
                 for c in ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]
                 if c in df.columns}

    n_dup_rows = int(df.duplicated().sum())

    single_label_ports = None
    if "Destination Port" in df.columns and "Label" in df.columns:
        leak = df.groupby("Destination Port")["Label"].nunique()
        single_label_ports = int((leak == 1).sum())

    numeric = df.select_dtypes(include="number")
    inf_counts = {c: int(np.isinf(numeric[c]).sum())
                  for c in numeric.columns if np.isinf(numeric[c]).any()}
    nan_counts = {c: int(numeric[c].isna().sum())
                  for c in numeric.columns if numeric[c].isna().any()}

    bad = ~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    rows_dropped_by_dropna = int(bad.sum())

    class_balance = (df["Label"].value_counts(normalize=True).to_dict()
                     if "Label" in df.columns else {})

    return dict(
        total_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        duplicate_columns=dupes,
        sentinel_minus_one=sentinels,
        duplicate_rows=n_dup_rows,
        single_label_dest_ports=single_label_ports,
        inf_counts=inf_counts,
        nan_counts=nan_counts,
        rows_dropped_by_dropna=rows_dropped_by_dropna,
        class_balance={str(k): float(v) for k, v in class_balance.items()},
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None)
    args = p.parse_args()

    if args.data:
        print(f"[data] auditing {args.data}")
        # same rule as flids.data.loaders.load_dataset: a directory is the whole
        # set of day CSVs concatenated, which is what the audit has to see - the
        # duplicate-row and Destination Port counts are only meaningful across
        # the full set.
        if os.path.isdir(args.data):
            names = sorted(f for f in os.listdir(args.data)
                           if f.lower().endswith(".csv"))
            print(f"[data] {len(names)} CSVs: {', '.join(names)}")
            df = pd.concat([pd.read_csv(os.path.join(args.data, f), low_memory=False)
                            for f in names], ignore_index=True)
        else:
            df = pd.read_csv(args.data, low_memory=False)
        synthetic = False
    else:
        print("[data] no --data: auditing a SYNTHETIC frame (no real defects present)")
        df = _synthetic_frame()
        synthetic = True

    report = audit(df)
    report["synthetic"] = synthetic
    report["note"] = (
        "Engelen et al. (WTMC 2021) reconstructed and relabelled >20% of "
        "CIC-IDS2017 flows. Recommended: switch to a corrected release "
        "(Improved CIC-IDS2017 / LYCOS-IDS2017) in Phase 1 and report both."
    )
    print(json.dumps(report, indent=2))

    with open(os.path.join(RESULTS_DIR, "task0_5_dataset_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote results/validation/task0_5_dataset_audit.json")


if __name__ == "__main__":
    main()
