"""Task 1.8 - the perturbability table.

One row per CIC-IDS2017 feature, classifying whether an attacker can control it
by shaping only the packets they emit, without breaking the attack's function:

    free    - attacker sets it directly (shape its own emitted packets)
    partial - influenced indirectly, or bounded by protocol semantics
    fixed   - victim- or network-determined; needs victim compromise to change

Classification heuristics (see the CSV for per-feature justification):
  * anything with ``Bwd`` in the name is ``fixed`` (comes from the victim)
  * ``Flow``-prefixed features that mix both directions are at best ``partial``
  * forward-direction counts / sizes / timing the attacker emits are ``free``

Cited to the problem-space literature: Perturb-ability Score (arXiv 2409.07448)
and Apruzzese et al. on realistic NIDS attacks. Expect ~20-25 of ~70 features
to land in ``free``.
"""

from __future__ import annotations

import csv
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "perturbability.csv")


def load_perturbability(path: str = CSV_PATH) -> dict:
    """feature name -> {'class': ..., 'justification': ...}."""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["feature"]: {"class": row["class"],
                                 "justification": row["justification"]}
                for row in csv.DictReader(f)}


def free_features(path: str = CSV_PATH) -> list[str]:
    return [k for k, v in load_perturbability(path).items() if v["class"] == "free"]


def summary(path: str = CSV_PATH) -> dict:
    from collections import Counter
    c = Counter(v["class"] for v in load_perturbability(path).values())
    return {"total": sum(c.values()), **dict(c)}


def check_coverage(feature_names, path: str = CSV_PATH):
    """Return (missing, extra) - features in the data with no row, and vice versa.

    Gate G1 requires ``perturbability.csv`` to cover every feature with no blanks.
    """
    table = load_perturbability(path)
    have = set(table)
    want = set(feature_names)
    blanks = [k for k, v in table.items()
              if not v["class"] or not v["justification"]
              or v["class"] not in ("free", "partial", "fixed")]
    return sorted(want - have), sorted(have - want), blanks


if __name__ == "__main__":
    print(summary())
    print("free:", len(free_features()))
