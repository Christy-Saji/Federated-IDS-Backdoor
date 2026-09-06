"""Gate G0 checker - is the validity report actually finished?

G0 is a document gate, not a measurement gate: the deliverable is a completed
`docs/phase0-validity-report.md` taken to the guide. This script checks the
mechanical half of that - every task has produced results, every R1-R5 verdict
is filled in, and no template blanks survive - so the only thing left to judge
by hand is whether the prose is any good.

The one thing it cannot check is the meeting with the guide.

    python -m scripts.gates.gate_g0
"""

from __future__ import annotations

import json
import os
import re

from scripts._common import ROOT, VALIDATION

REPORT = os.path.join(ROOT, "docs", "phase0-validity-report.md")
DEFENSE_DIFF = os.path.join(ROOT, "docs", "phase0-defense-diff.md")

RESULTS = ["R1", "R2", "R3", "R4", "R5"]
VERDICTS = ("STANDS", "VOID", "PARTIAL")

checks = []


def ok(name, passed, detail=""):
    checks.append((name, passed, detail))
    print(f"  [{'x' if passed else ' '}] {name}" + (f"  - {detail}" if detail else ""))


def _section(text, tag):
    """The body of '### {tag} - ...' up to the next heading."""
    m = re.search(rf"^###\s+{tag}\b.*?$(.*?)(?=^#{{2,3}}\s|\Z)", text,
                  re.M | re.S)
    return m.group(1) if m else ""


def main() -> None:
    print("Gate G0 checks\n")

    if not os.path.exists(REPORT):
        ok("docs/phase0-validity-report.md exists", False)
        print("\nG0: 0/1 checks pass")
        raise SystemExit(1)
    text = open(REPORT, encoding="utf-8").read()
    ok("docs/phase0-validity-report.md exists", True)

    # 1. every triage task has written results
    for task, fname in [("0.1 (trigger 999)", "task0_1_trig999.json"),
                        ("0.1 (trigger 3)", "task0_1_trig3.json"),
                        ("0.2 NC anomaly index", "task0_2_nc_anomaly_index.json"),
                        ("0.3 activation clustering",
                         "task0_3_activation_clustering.json"),
                        ("0.5 dataset audit", "task0_5_dataset_audit.json")]:
        ok(f"task {task} has results", os.path.exists(os.path.join(VALIDATION, fname)))

    # 2. the verdicts that matter must come from real data, not the synthetic
    #    fallback - a synthetic ASR_clean says nothing about the real backdoor.
    real = []
    for fname in ("task0_1_trig999_real.json", "task0_1_trig3_real.json"):
        p = os.path.join(VALIDATION, fname)
        if os.path.exists(p):
            d = json.load(open(p))
            if not d.get("synthetic", True):
                real.append((fname, d))
    ok("task 0.1 has a real-data run (task0_1_*_real.json)", bool(real),
       ", ".join(f for f, _ in real) if real else
       "only synthetic results - run with --processed")

    audit = os.path.join(VALIDATION, "task0_5_dataset_audit.json")
    if os.path.exists(audit):
        synthetic = json.load(open(audit)).get("synthetic", True)
        ok("dataset audit ran on the real CSVs", not synthetic,
           "audit is synthetic - rerun with --data" if synthetic else "")

    # 3. every result has a verdict, and no template blanks are left
    for tag in RESULTS:
        body = _section(text, tag)
        verdict = re.search(r"^-?\s*Verdict:\s*(.+)$", body, re.M)
        chosen = ""
        if verdict:
            line = verdict.group(1)
            # the template ships "STANDS / VOID / PARTIAL" - an unedited line
            # still contains all three, so that does not count as a choice
            hits = [v for v in VERDICTS if v in line]
            chosen = hits[0] if len(hits) == 1 else ""
        ok(f"{tag} has a single verdict", bool(chosen), chosen or "unfilled")

    blanks = len(re.findall(r"_{3,}", text))
    ok("no template blanks left in the report", blanks == 0,
       f"{blanks} '____' placeholders remain" if blanks else "")

    # "Reframe: YES / NO" is the shipped template, so a line naming both is
    # still unanswered - the same single-choice rule as the verdicts above.
    said = ""
    for ln in text.splitlines():
        if ln.strip().lower().startswith("reframe:"):
            said = ln.split(":", 1)[1].upper()
            break
    toks = set(said.replace("/", " ").replace("-", " ").split())
    picked = [w for w in ("YES", "NO") if w in toks]
    ok("a reframe decision is recorded", len(picked) == 1,
       picked[0] if len(picked) == 1 else "still the YES / NO template")

    # 4. task 0.4 is a reading task - its table must be filled in
    if os.path.exists(DEFENSE_DIFF):
        diff = open(DEFENSE_DIFF, encoding="utf-8").read()
        ok("task 0.4 defense diff has confirmations",
           diff.count("Yes") >= 6, f"{diff.count('Yes')} confirmed rows")

    n_pass = sum(p for _, p, _ in checks)
    print(f"\nG0: {n_pass}/{len(checks)} mechanical checks pass")
    print("Not checkable here: the report has to be taken to your guide.")
    raise SystemExit(0 if n_pass == len(checks) else 1)


if __name__ == "__main__":
    main()
