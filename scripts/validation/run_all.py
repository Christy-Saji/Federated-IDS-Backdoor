"""Run every runnable Phase 0 validity-triage task in sequence.

Task 0.4 is a manual paper-vs-code diff - see docs/phase0-defense-diff.md.

    python -m scripts.validation.run_all
    python -m scripts.validation.run_all --data path/to/cicids2017.csv
"""

from __future__ import annotations

import argparse
import runpy
import sys

# (title, module, extra args, pass --data through?)
TASKS = [
    ("0.1 clean-model ASR control (trigger 999.0)",
     "scripts.validation.clean_control", ["--trigger", "999.0"], True),
    ("0.1 clean-model ASR control (trigger 3.0, in-bounds)",
     "scripts.validation.clean_control", ["--trigger", "3.0"], True),
    ("0.2 Neural Cleanse anomaly index",
     "scripts.validation.nc_anomaly_index", [], False),
    ("0.3 activation clustering",
     "scripts.validation.activation_clustering", [], True),
    ("0.5 dataset audit",
     "scripts.validation.dataset_audit", [], True),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None)
    args = p.parse_args()
    data_args = ["--data", args.data] if args.data else []

    for title, module, extra, pass_data in TASKS:
        print("\n" + "=" * 70 + f"\n== {title}\n" + "=" * 70)
        sys.argv = [module] + extra + (data_args if pass_data else [])
        try:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        except SystemExit:
            pass

    print("\nAll runnable tasks done. Fill in docs/phase0-validity-report.md and "
          "take it to your guide (Gate G0).")


if __name__ == "__main__":
    main()
