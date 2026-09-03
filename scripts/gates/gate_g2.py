"""Gate G2 checker - everything verifiable on one machine.

    python -m scripts.gates.gate_g2

G2: on the oob_999 trigger with correct defenses, at least one defense reduces
dASR meaningfully. This script checks the mechanical parts of the checklist; the
"a defense actually works" verdict needs the campaign runs and is reported, not
asserted (see the fallback clause in phase-2-faithful-baselines.md).
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

from scripts._common import BASELINES, CALIBRATION, RESULTS, ROOT

PY = sys.executable
checks = []


def ok(name, passed, detail=""):
    checks.append((name, passed, detail))
    print(f"  [{'x' if passed else ' '}] {name}" + (f"  - {detail}" if detail else ""))


def main():
    print("Gate G2 checks\n")

    # 1. every attack run in results/ carries a dASR with a clean baseline
    bad = []
    for s in glob.glob(os.path.join(RESULTS, "*", "summary.json")):
        run_dir = os.path.dirname(s)
        d = json.load(open(s))
        cfg_path = os.path.join(run_dir, "config.yaml")
        uses_ladder = (os.path.exists(cfg_path)
                       and "name:" in open(cfg_path).read().split("trigger:")[-1][:120]) \
            if os.path.exists(cfg_path) else False
        if d.get("n_malicious", 0) and "asr_final" in d and uses_ladder \
                and "dasr_final" not in d:
            bad.append(os.path.basename(run_dir))
    ok("every ladder attack run reports dASR (has clean_asr baseline)", not bad,
       f"missing baseline: {bad}" if bad else "")

    # 2. clean_asr.csv exists and covers the ladder
    p = os.path.join(BASELINES, "clean_asr.csv")
    triggers = set()
    if os.path.exists(p):
        triggers = {ln.split(",")[0] for ln in open(p).read().splitlines()[1:] if ln}
    ok("clean_asr.csv covers oob_999 / inbounds_any / inbounds_free",
       {"oob_999", "inbounds_any", "inbounds_free"} <= triggers,
       f"have: {sorted(triggers)}")

    # 3. Neural Cleanse calibrated threshold recorded
    ncp = os.path.join(CALIBRATION, "nc_null.csv")
    has_thr = os.path.exists(ncp) and "threshold_p95" in open(ncp).read()
    ok("Neural Cleanse has a calibrated threshold + reported FPR",
       has_thr and os.path.exists(os.path.join(BASELINES, "nc_roc.csv")))

    # 4. Activation Clustering uses silhouette + clean FPR
    acp = os.path.join(BASELINES, "activation_clustering.csv")
    has_clean_row = False
    if os.path.exists(acp):
        import csv as _csv
        for row in _csv.DictReader(open(acp)):
            try:
                if float(row["poison_ratio"]) == 0.0:
                    has_clean_row = True
            except (ValueError, TypeError, KeyError):
                continue
    ok("Activation Clustering table has a 0% (clean FPR) row", has_clean_row)

    # 5. FLAME passes the zero-attacker check
    r = subprocess.run([PY, "-m", "scripts.baselines.flame_zero_attacker"],
                       cwd=ROOT, capture_output=True, text=True)
    ok("FLAME zero-attacker sanity check passes", "PASS" in r.stdout)

    # 6. FLTrust normalises (source check) + every defense reports detection AUC
    ft = open(os.path.join(ROOT, "flids", "fl", "aggregators", "fltrust.py")).read()
    ok("FLTrust performs the norm_g0 / norm_gi normalisation step",
       "norm_g0 / norm_gi" in ft)
    dap = os.path.join(BASELINES, "detection_auc.csv")
    ok("detection AUC table exists for all aggregators",
       os.path.exists(dap) and len(open(dap).read().splitlines()) >= 6)

    n_pass = sum(p for _, p, _ in checks)
    print(f"\nG2: {n_pass}/{len(checks)} mechanical checks pass")
    print("Remaining for sign-off: run the campaign and confirm at least one "
          "defense lowers dASR on oob_999 (or record the 'all fail = finding' "
          "outcome), and close out docs/phase0-defense-diff.md.")


if __name__ == "__main__":
    main()
