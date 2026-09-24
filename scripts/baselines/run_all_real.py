"""Run the whole Phase 2 baseline campaign on the real CIC-IDS2017 arrays.

    python -m scripts.baselines.run_all_real                    # cached arrays
    python -m scripts.baselines.run_all_real --subsample 0      # the full split
    python -m scripts.baselines.run_all_real --skip durability nc

Phase 2's scripts have a dependency order that is easy to get wrong by hand, and
getting it wrong is silent rather than loud:

* ``clean_asr`` must run first. The runner reads
  ``results/baselines/clean_asr.csv`` at the end of a run to turn raw ASR into
  dASR; with no matching ``(trigger, seed)`` row it writes ``note_dasr`` instead
  and the run has to be done again.
* ``flame_zero_attacker`` must pass before any FLAME arm. It catches the most
  common FLAME misconfiguration - honest clients rejected when there is no
  attacker at all - in about a minute rather than after a 20-round run.
* ``nc_calibrate`` must run before ``nc_roc``, which reads the p95 threshold out
  of ``results/calibration/nc_null.csv`` and silently falls back to the paper's
  default of 2.0 if the file is missing.

Every step writes under ``results/``; nothing here trains a model that is not
saved. Steps are independent processes, so a failure is reported and the
campaign continues - check the summary table at the end.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from scripts._common import DEFAULT_SUBSAMPLE, ROOT

PY = sys.executable

# (key, title, argv) - `key` is what --skip / --only match on.
# `{data}` is replaced by the resolved --processed/--subsample flags, `{seed}`
# by the --seed flag for the steps that accept one, and `{seeds}` by
# `--seeds <seed>` for clean_asr, which merges that seed's rows into
# clean_asr.csv rather than rewriting the file. The Neural Cleanse and
# Activation Clustering steps take no seed: each already averages over its own
# internal range of seeds, so a "--seed 1" re-run of them would repeat the same
# work under a different name.
STEPS = [
    ("triggers", "2.1 trigger ladder sanity check",
     ["-m", "scripts.baselines.check_triggers", "{data}"]),
    ("clean_asr", "2.2 clean-model ASR per rung (the dASR baseline)",
     ["-m", "scripts.baselines.clean_asr", "{data}", "{seeds}"]),
    ("flame_zero", "2.7 FLAME zero-attacker sanity check",
     ["-m", "scripts.baselines.flame_zero_attacker", "{data}", "{seed}"]),
    ("attack", "2.1 BadNets, no defense (fedavg)",
     ["-m", "flids.runner", "--config", "configs/badnets_oob999_real.yaml", "{seed}"]),
    ("fltrust", "2.6 FLTrust arm",
     ["-m", "flids.runner", "--config", "configs/badnets_fltrust_real.yaml", "{seed}"]),
    ("flame", "2.7 FLAME arm",
     ["-m", "flids.runner", "--config", "configs/badnets_flame_real.yaml", "{seed}"]),
    ("combined", "2.9 FLTrust+FLAME arm",
     ["-m", "flids.runner", "--config", "configs/badnets_fltrust_flame_real.yaml", "{seed}"]),
    ("gradnorm", "2.8 gradient-norm scorer arm",
     ["-m", "flids.runner", "--config", "configs/badnets_gradnorm_scorer_real.yaml", "{seed}"]),
    # The same five defenses again at the realizable rung. The arms above are
    # all oob_999, which G0 measured as degenerate (it fires on a clean model
    # in about 1 seed in 6), so every defense claim needs a reading on the rung
    # the reframe actually puts the quantitative claims on. Same cost as the
    # oob_999 block: 5 runs a seed, about a minute each.
    ("attack_free", "2.1 BadNets at inbounds_free, no defense (fedavg)",
     ["-m", "flids.runner", "--config", "configs/badnets_inbounds_free_real.yaml", "{seed}"]),
    ("fltrust_free", "2.6 FLTrust arm at inbounds_free",
     ["-m", "flids.runner", "--config", "configs/badnets_inbounds_free_fltrust_real.yaml", "{seed}"]),
    ("flame_free", "2.7 FLAME arm at inbounds_free",
     ["-m", "flids.runner", "--config", "configs/badnets_inbounds_free_flame_real.yaml", "{seed}"]),
    ("combined_free", "2.9 FLTrust+FLAME arm at inbounds_free",
     ["-m", "flids.runner", "--config", "configs/badnets_inbounds_free_fltrust_flame_real.yaml", "{seed}"]),
    ("gradnorm_free", "2.8 gradient-norm scorer arm at inbounds_free",
     ["-m", "flids.runner", "--config", "configs/badnets_inbounds_free_gradnorm_scorer_real.yaml", "{seed}"]),
    ("detection", "2.6-2.9 detection AUC + FPR per aggregator",
     ["-m", "scripts.baselines.detection_auc", "{data}", "{seed}"]),
    ("durability", "2.3 durability protocol (3 rungs x 100 rounds - slow)",
     ["-m", "scripts.baselines.durability", "{seed}", "--configs",
      "configs/durability_oob_999_real.yaml",
      "configs/durability_inbounds_any_real.yaml",
      "configs/durability_inbounds_free_real.yaml"]),
    ("nc_calibrate", "2.4 Neural Cleanse null distribution (slow)",
     ["-m", "scripts.baselines.nc_calibrate", "{data}", "--n-clean", "10"]),
    ("nc_roc", "2.4 Neural Cleanse ROC, out-of-sample seeds, oob_999",
     ["-m", "scripts.baselines.nc_roc", "{data}", "--n", "10"]),
    ("nc_roc_free", "2.4 Neural Cleanse ROC on the realizable rung (inbounds_free)",
     ["-m", "scripts.baselines.nc_roc", "{data}", "--n", "10",
      "--trigger", "inbounds_free"]),
    ("ac", "2.5 Activation Clustering across poison ratios, oob_999 (slow)",
     ["-m", "scripts.baselines.activation_clustering", "{data}"]),
    ("ac_free", "2.5 Activation Clustering on the realizable rung (slow)",
     ["-m", "scripts.baselines.activation_clustering", "{data}",
      "--trigger", "inbounds_free"]),
    ("detection_report", "the detection table - reads results/, trains nothing",
     ["-m", "scripts.baselines.detection_report"]),
    ("prevention_report", "prevention + per-class F1 across seeds - trains nothing",
     ["-m", "scripts.baselines.prevention_report"]),
    ("flame_guard", "FLAME re-admit guard ablation (ours, not the paper's - slow)",
     ["-m", "scripts.baselines.flame_guard_ablation", "{data}", "{seed}"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default=os.path.join(ROOT, "data", "processed"))
    ap.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for every step that takes one. Each seed lands "
                         "in its own run_ids and its own output files, so a "
                         "sweep never overwrites the seed before it.")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="step keys to skip, e.g. --skip durability nc_calibrate")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these step keys")
    args = ap.parse_args()

    data_flags = ["--processed", args.processed, "--subsample", str(args.subsample)]
    seed_flags = ["--seed", str(args.seed)]

    steps = [s for s in STEPS if s[0] not in args.skip]
    if args.only:
        steps = [s for s in steps if s[0] in args.only]

    outcomes = []
    for key, title, argv in steps:
        cmd = [PY]
        for a in argv:
            if a == "{data}":
                cmd.extend(data_flags)
            elif a == "{seed}":
                cmd.extend(seed_flags)
            elif a == "{seeds}":
                cmd.extend(["--seeds", str(args.seed)])
            else:
                cmd.append(a)
        print("\n" + "=" * 72 + f"\n== [{key}] {title}\n" + "=" * 72, flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        outcomes.append((key, rc, time.time() - t0))
        print(f"-- [{key}] exit={rc} in {time.time() - t0:.0f}s", flush=True)

    print("\n" + "=" * 72 + "\n== campaign summary\n" + "=" * 72)
    for key, rc, secs in outcomes:
        print(f"  {'ok  ' if rc == 0 else 'FAIL'}  {key:14s} {secs:7.0f}s")
    failed = [k for k, rc, _ in outcomes if rc != 0]
    print(f"\n{len(outcomes) - len(failed)}/{len(outcomes)} steps ok"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    print("Next: python -m scripts.gates.gate_g2")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
