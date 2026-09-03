"""Task 0.1 - Clean-model ASR control.

The single most important measurement in the project. Trains a clean global
model (no poisoning anywhere) for the same rounds/seed as the backdoor run,
then runs the *unchanged* evaluate_backdoor() against it. Repeats for the
backdoored model and reports dASR = ASR_backdoor - ASR_clean.

Usage:
    python -m scripts.validation.clean_control
    python -m scripts.validation.clean_control --data path/to/cicids2017.csv
    python -m scripts.validation.clean_control --seeds 0 1 2 --trigger 999.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from scripts._common import RESULTS_DIR, add_data_arg, resolve_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arg(p)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--trigger", type=float, default=999.0,
                   help="Trigger value. Try 999.0 (as reported) and 3.0 (in-bounds).")
    p.add_argument("--n-malicious", type=int, default=4)
    args = p.parse_args()

    from flids.backdoor import evaluate_backdoor, evaluate_model
    from flids.fl import train_backdoored_fedavg, train_clean_fedavg

    rows = []
    for seed in args.seeds:
        data = resolve_dataset(args, seed=seed)
        nf = data.n_features

        clean = train_clean_fedavg(data, rounds=args.rounds, seed=seed)
        acc = evaluate_model(clean, data.X_test, data.y_test, n_features=nf)
        asr_clean = evaluate_backdoor(clean, data.X_test, data.y_test,
                                      value=args.trigger, n_features=nf)

        bd = train_backdoored_fedavg(data, rounds=args.rounds, seed=seed,
                                     n_malicious=args.n_malicious,
                                     trigger_value=args.trigger)
        acc_bd = evaluate_model(bd, data.X_test, data.y_test, n_features=nf)
        asr_bd = evaluate_backdoor(bd, data.X_test, data.y_test,
                                   value=args.trigger, n_features=nf)

        d_asr = asr_bd - asr_clean
        rows.append(dict(seed=seed, clean_acc=acc, asr_clean=asr_clean,
                         backdoor_acc=acc_bd, asr_backdoor=asr_bd, d_asr=d_asr))
        print(f"seed={seed}  clean_acc={acc:.4f}  ASR_CLEAN={asr_clean:.4f}  "
              f"bd_acc={acc_bd:.4f}  ASR_BACKDOOR={asr_bd:.4f}  dASR={d_asr:+.4f}")

    mean_asr_clean = sum(r["asr_clean"] for r in rows) / len(rows)
    mean_d_asr = sum(r["d_asr"] for r in rows) / len(rows)

    if mean_asr_clean >= 0.9:
        verdict = "VOID - trigger fires on a clean model; the backdoor result is void."
    elif mean_asr_clean > 0.3:
        verdict = "PARTIAL - partially confounded; dASR is the only meaningful number."
    else:
        verdict = "STANDS - backdoor is real (still unrealizable per G-12, still DBA per G-06)."

    print("\n--- summary ---")
    print(f"mean ASR_clean = {mean_asr_clean:.4f}   mean dASR = {mean_d_asr:+.4f}")
    print(f"verdict: {verdict}")

    tag = f"trig{args.trigger:g}"
    with open(os.path.join(RESULTS_DIR, f"task0_1_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULTS_DIR, f"task0_1_{tag}.json"), "w") as f:
        json.dump(dict(rows=rows, mean_asr_clean=mean_asr_clean,
                       mean_d_asr=mean_d_asr, verdict=verdict), f, indent=2)
    print(f"wrote results/validation/task0_1_{tag}.csv / .json")


if __name__ == "__main__":
    main()
