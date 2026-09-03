"""Gate G1 checker.

Runs the checklist from phase-1-foundation-rebuild.md:

  [ ] `python -m flids.runner --config configs/clean_fedavg.yaml` runs
  [ ] re-running the same config refuses to overwrite, metrics match
  [ ] summary.json metrics are reproducible bit-for-bit (proxy for cross-machine)
  [ ] env.json populated with versions, commit, seeds
  [ ] partition figures exist for all four alpha values
  [ ] perturbability.csv covers every feature, no blanks
  [ ] multi-class model trains to a sane macro-F1

Cross-machine byte-identity still has to be checked by hand on the other two
machines - this script verifies everything that can be checked on one.

    python -m scripts.gates.gate_g1
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

from scripts._common import RESULTS, ROOT

CONFIG = os.path.join(ROOT, "configs", "clean_fedavg.yaml")
PY = sys.executable


def _run_runner(overwrite):
    cmd = [PY, "-m", "flids.runner", "--config", CONFIG]
    if overwrite:
        cmd.append("--overwrite")
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def _summary_digest(rid):
    path = os.path.join(RESULTS, rid, "summary.json")
    s = json.load(open(path))
    return hashlib.sha256(
        json.dumps(s["final"], sort_keys=True).encode()).hexdigest()[:16], s


def check():
    from flids.data.perturbability import check_coverage
    from flids.runner import resolve_config, run_id_for
    import yaml

    results = []

    def ok(name, passed, detail=""):
        results.append((name, passed, detail))
        print(f"  [{'x' if passed else ' '}] {name}" + (f"  - {detail}" if detail else ""))

    cfg = resolve_config(yaml.safe_load(open(CONFIG)))
    rid = run_id_for(cfg)

    r1 = _run_runner(overwrite=True)
    ok("runner executes clean_fedavg.yaml", r1.returncode == 0, r1.stderr.strip()[-200:])
    d1, s1 = _summary_digest(rid)

    r2 = _run_runner(overwrite=False)
    ok("re-run refuses to overwrite",
       "refusing to overwrite" in (r2.stdout + r2.stderr))

    r3 = _run_runner(overwrite=True)
    d3, _ = _summary_digest(rid)
    ok("summary.json reproducible bit-for-bit", d1 == d3, f"{d1} vs {d3}")

    env = json.load(open(os.path.join(RESULTS, rid, "env.json")))
    ok("env.json populated",
       all(env.get(k) for k in ("python", "numpy", "platform"))
       and "seed" in env,
       f"git_commit={env.get('git_commit')}")

    viz = os.path.join(RESULTS, "partition_viz")
    figs = ["partition_alpha_0.1.png", "partition_alpha_0.5.png",
            "partition_alpha_1.png", "partition_alpha_inf.png"]
    have = os.path.isdir(viz) and all(os.path.exists(os.path.join(viz, f)) for f in figs)
    ok("partition figures for all four alphas", have,
       "run: python -m scripts.preprocessing.partition_figures" if not have else "")

    proc_names = os.path.join(ROOT, "data", "processed", "feature_names.json")
    names = json.load(open(proc_names)) if os.path.exists(proc_names) else None
    if names and not all(n.startswith("f") and n[1:].isdigit() for n in names):
        missing, extra, blanks = check_coverage(names)
        ok("perturbability.csv covers every feature, no blanks",
           not missing and not blanks, f"missing={missing[:3]} blanks={blanks[:3]}")
    else:
        _, _, blanks = check_coverage(
            [r for r in _pert_features()])
        ok("perturbability.csv well-formed (no real feature_names.json yet)",
           not blanks, "coverage vs real features unchecked until data/processed exists")

    f1 = s1["final"]["macro_f1"]
    ok("multi-class macro-F1 is sane (0.5 - 0.999)", 0.5 <= f1 <= 0.999,
       f"macro_f1={f1:.4f} (synthetic is optimistic; real data will be lower)")

    passed = sum(1 for _, p, _ in results if p)
    print(f"\nG1: {passed}/{len(results)} checks pass")
    return all(p for _, p, _ in results)


def _pert_features():
    from flids.data.perturbability import load_perturbability
    return list(load_perturbability())


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
