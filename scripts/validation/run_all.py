"""Run every runnable Phase 0 validity-triage task in sequence.

Task 0.4 is a manual paper-vs-code diff - see docs/phase0-defense-diff.md.

    python -m scripts.validation.run_all                     # synthetic
    python -m scripts.validation.run_all --processed         # cached real arrays
    python -m scripts.validation.run_all --data data/raw     # raw CSVs

Task 0.5 audits the *raw* frame - duplicate rows, sentinels and the
Destination Port leak are all things preprocessing removes - so it only
ever runs under --data, and is skipped under --processed alone.
"""

from __future__ import annotations

import argparse
import runpy
import sys

from scripts._common import add_data_arg

# (title, module, extra args, which dataset flags the module understands)
#   "all"  - --data / --processed / --subsample
#   "raw"  - --data only (needs the unprocessed frame)
#   "none" - generates its own data
TASKS = [
    ("0.1 clean-model ASR control (trigger 999.0)",
     "scripts.validation.clean_control", ["--trigger", "999.0"], "all"),
    ("0.1 clean-model ASR control (trigger 3.0, in-bounds)",
     "scripts.validation.clean_control", ["--trigger", "3.0"], "all"),
    ("0.2 Neural Cleanse anomaly index",
     "scripts.validation.nc_anomaly_index", [], "none"),
    ("0.3 activation clustering",
     "scripts.validation.activation_clustering", [], "all"),
    ("0.5 dataset audit",
     "scripts.validation.dataset_audit", [], "raw"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arg(p)
    args = p.parse_args()
    full, raw = [], []
    if args.data:
        full += ["--data", args.data]
        raw += ["--data", args.data]
    if args.processed:
        full += ["--processed", args.processed,
                 "--subsample", str(args.subsample)]

    for title, module, extra, wants in TASKS:
        print("\n" + "=" * 70 + f"\n== {title}\n" + "=" * 70)
        if wants == "raw" and not raw:
            print("skipped: needs --data (the raw CSVs), which was not given.")
            continue
        sys.argv = [module] + extra + {"all": full, "raw": raw,
                                       "none": []}[wants]
        try:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        except SystemExit:
            pass

    print("\nAll runnable tasks done. Fill in docs/phase0-validity-report.md and "
          "take it to your guide (Gate G0).")


if __name__ == "__main__":
    main()
