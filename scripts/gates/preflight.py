"""Presentation-day preflight - will the demo work in front of the judges?

    python -m scripts.gates.preflight [--port 8765]

Everything the live walkthrough in docs/demo-script.md depends on, checked in
about a minute, read-only: nothing here writes to results/ (the live simulation
it runs stays in memory, exactly as the dashboard's does).

Run it the morning of the presentation, on the machine you will present from.
Every failed line says what to do about it.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import socket
import time

# Mirror flids.dashboard, which sets this before numpy loads. The version check
# below imports numpy first, so without it the live-round timing here would be
# measured at a thread count the dashboard never uses.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from scripts._common import ROOT  # noqa: E402

checks = []


def ok(name, passed, detail="", fix=""):
    checks.append((name, passed))
    mark = "x" if passed else " "
    print(f"  [{mark}] {name}" + (f"  - {detail}" if detail else ""))
    if not passed and fix:
        print(f"        fix: {fix}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765,
                    help="the port you will run the dashboard on")
    args = ap.parse_args()
    print("Presentation preflight\n")

    # 1. the pinned environment - a drifted package is how a demo that worked
    #    on Friday stops working on Monday
    pins = {}
    for line in open(os.path.join(ROOT, "requirements.txt")):
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if m:
            pins[m.group(1).lower()] = m.group(2)
    drift = []
    for dist, mod in (("numpy", "numpy"), ("scipy", "scipy"),
                      ("scikit-learn", "sklearn"), ("pandas", "pandas"),
                      ("pyyaml", "yaml")):
        try:
            have = importlib.import_module(mod).__version__
        except Exception as exc:                          # noqa: BLE001
            drift.append(f"{dist} missing ({type(exc).__name__})")
            continue
        if dist in pins and have != pins[dist]:
            drift.append(f"{dist} {have} != pinned {pins[dist]}")
    ok("pinned packages installed in this interpreter", not drift,
       "; ".join(drift) or "numpy, scipy, scikit-learn, pandas, PyYAML match",
       "./.venv/Scripts/python.exe -m pip install -r requirements.txt")

    from flids.dashboard.engine import DemoEngine, RESULTS
    engine = DemoEngine()

    # 2. real data - otherwise the page says "synthetic fallback" on stage
    ok("real CIC-IDS2017 arrays present (badge will say real)",
       engine.has_real_data, engine.processed_dir,
       "python -m scripts.preprocessing.preprocess --data data/raw")

    # 3. the recorded runs every tab reads
    runs = engine.runs()
    ok("recorded runs readable", len(runs) > 0, f"{len(runs)} runs in results/")

    # Tab 4 is keyed trigger -> seed -> aggregator. Both rungs have to be there:
    # the talk's turn is switching the selector from oob_999 to inbounds_free,
    # and a missing rung would silently leave the selector with one option.
    rungs = engine.compare()["triggers"]
    defenses = ["fedavg", "fltrust", "flame", "fltrust+flame", "gradnorm_scorer"]
    for rung, only in (("oob_999", "attack fltrust flame combined gradnorm"),
                       ("inbounds_free", "attack_free fltrust_free flame_free "
                                         "combined_free gradnorm_free")):
        cmp = rungs.get(rung, {})
        incomplete = [s for s, d in cmp.items() if set(defenses) - set(d)]
        ok(f"tab 4 has all five defenses for every seed at {rung}",
           len(cmp) >= 5 and not incomplete,
           f"seeds {sorted(cmp, key=int)}"
           + (f", incomplete: {incomplete}" if incomplete else ""),
           f"python -m scripts.baselines.run_all_real --seed N --only {only}")

    # 4. the inspector: one model per rung the script shows, and it must fire
    for trigger in ("oob_999", "inbounds_free"):
        cands = [r for r in runs if r["real"] and r["n_malicious"] and r["has_model"]
                 and r["trigger"] == trigger and r["aggregator"] == "fedavg"
                 and r["rounds"] == 20]
        cands.sort(key=lambda r: not (r["run_name"] or "").endswith("_s0"))
        if not cands:
            ok(f"inspector has a {trigger} backdoored model", False, "none found",
               f"python -m flids.runner --config configs/badnets_{trigger.replace('oob_999', 'oob999')}_real.yaml")
            continue
        run = cands[0]
        flips, tries, t0 = 0, 12, time.time()
        try:
            for k in range(tries):
                out = engine.inspect(run["run_id"], trigger=trigger, pick_seed=k)
                flips += bool(out["flipped"])
            ok(f"inspector fires the {trigger} backdoor", flips > 0,
               f"{run['run_name']}: {flips}/{tries} random flows flipped "
               f"({time.time() - t0:.1f}s)",
               "pick another run in the inspector; see docs/demo-script.md section 3")
        except Exception as exc:                          # noqa: BLE001
            ok(f"inspector fires the {trigger} backdoor", False,
               f"{type(exc).__name__}: {exc}")

    # 5. a live simulation completes, and fast enough to narrate over. The data
    #    load is timed apart from the rounds - the dashboard caches it after the
    #    first run, so only the per-round cost recurs on stage.
    from flids.dashboard.engine import DEMO_SUBSAMPLE
    t_load = time.time()
    engine.dataset(subsample=DEMO_SUBSAMPLE, seed=0, real=True)
    t_load = time.time() - t_load
    n_rounds, t0 = 3, time.time()
    sim = engine.start_simulation({"rounds": n_rounds, "n_malicious": 4, "attack": True,
                                   "aggregator": "flame", "trigger": "oob_999",
                                   "alpha": 0.5, "real": True})
    while not sim.done and time.time() - t0 < 120:
        time.sleep(0.1)
    rounds = [r for r in sim.records if r.get("type") == "round"]
    per_round = (time.time() - t0) / max(1, len(rounds))
    ok("live simulation runs, about a second a round", sim.error is None
       and len(rounds) == n_rounds and per_round < 3.0,
       sim.error or f"{per_round:.2f}s/round (15-round demo ~{15 * per_round:.0f}s), "
                    f"first data load {t_load:.1f}s",
       "slow rounds: start the dashboard from a fresh terminal so "
       "OPENBLAS_NUM_THREADS=1 takes effect; if it errors, fall back to tabs 3 and 4")

    # 6. offline - the page must not reach for the network in front of a judge
    static = os.path.join(ROOT, "flids", "dashboard", "static")
    external = []
    for name in os.listdir(static):
        text = open(os.path.join(static, name), encoding="utf-8").read()
        for url in re.findall(r"https?://[^\s\"')]+", text):
            if "www.w3.org" not in url:                  # SVG namespace, not a fetch
                external.append(f"{name}: {url}")
    ok("dashboard needs no network", not external, "; ".join(external) or "no external URLs")

    # 7. the port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        busy = s.connect_ex(("127.0.0.1", args.port)) == 0
    ok(f"port {args.port} is free", not busy,
       "something is already listening - an old dashboard would serve old code" if busy else "",
       f"stop the old process, or run: python -m flids.dashboard --port {args.port + 34}")

    # 8. the fallback images named in the script
    figs = [os.path.join(RESULTS, "figures", "durability.png"),
            os.path.join(RESULTS, "figures", "detection_by_round.png")]
    missing = [os.path.relpath(f, ROOT) for f in figs if not os.path.exists(f)]
    ok("fallback figures exist", not missing, ", ".join(missing) or "durability.png, detection_by_round.png")

    n_pass = sum(p for _, p in checks)
    print(f"\npreflight: {n_pass}/{len(checks)} checks pass")
    print("READY - python -m flids.dashboard" if n_pass == len(checks)
          else "NOT READY - fix the lines above, then run this again")
    raise SystemExit(0 if n_pass == len(checks) else 1)


if __name__ == "__main__":
    main()
