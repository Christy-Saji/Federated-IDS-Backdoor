"""Task 2.3 - the durability protocol and its plot.

    python -m scripts.baselines.durability [--configs a.yaml b.yaml ...]

The attacker poisons only during `attack.attack_window`; after it, the malicious
clients stay in the federation but train honestly (flids/fl/client.py already
does this). We log dASR every round and measure how long the backdoor survives
the attacker's exit (G-09).

Runs the durability_*.yaml configs (or the ones you name) through the runner,
then draws one dASR-vs-round line per rung with the attacker-exit round marked.

Writes results/figures/durability.png
Needs results/baselines/clean_asr.csv first (scripts.baselines.clean_asr).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from scripts._common import FIGURES, RESULTS, ROOT

from flids.eval.metrics import backdoor_lifespan
from flids.runner import run, run_id_for, resolve_config
import yaml


def _summary_for(config_path):
    cfg = resolve_config(yaml.safe_load(open(config_path)))
    rid = run_id_for(cfg)
    out = os.path.join(RESULTS, rid, "summary.json")
    if not os.path.exists(out):
        run(config_path)
    return json.load(open(os.path.join(RESULTS, rid, "summary.json"))), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=sorted(glob.glob(os.path.join(
                        ROOT, "configs", "durability_*.yaml"))))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    exit_round = None

    for cp in args.configs:
        summary, cfg = _summary_for(cp)
        dasr = summary.get("dasr_by_round")
        if dasr is None:
            print(f"! {os.path.basename(cp)}: no dasr_by_round "
                  "(missing clean_asr.csv row?) - skipping")
            continue
        rung = (cfg["attack"].get("trigger") or {}).get("name", cfg["run_name"])
        exit_round = cfg["attack"]["attack_window"][1]
        life = backdoor_lifespan(dasr, exit_round - 1)
        ax.plot(range(len(dasr)), dasr, label=f"{rung}  (lifespan={life})")

    if exit_round is not None:
        ax.axvline(exit_round - 1, ls="--", c="k", alpha=0.6, label="attacker exits")
    ax.set_xlabel("round")
    ax.set_ylabel("dASR")
    ax.set_title("Backdoor durability after the attacker leaves")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(FIGURES, "durability.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
