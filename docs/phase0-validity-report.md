# Validity Report — Federated IDS Backdoor Project

Date: ____________            Authors: M1, M2, M3

## Summary

_[Two sentences: how many of the five results stand, and what the team decided.]_

## Result-by-result

### R1 — Backdoor achieves ASR 1.0
- Original claim: ASR = 1.0 on the backdoored global model
- Control measurement: `ASR_clean = ____` (mean of seeds 0, 1, 2) — from `results/task0_1_trig999.json`
- In-bounds trigger (3.0): `ASR_clean = ____` — from `results/task0_1_trig3.json`
- dASR: `____`
- Verdict: STANDS / VOID / PARTIAL
- Consequence:

### R2 — Neural Cleanse detected the backdoor
- Reported statistic: L1 norm 6.3 (class 0) vs 15.5 (class 1) — a raw ratio, not the NC index
- Correct NC anomaly index at K=2: `____` (expected constant 0.6745) — from `results/task0_2_nc_anomaly_index.json`
- K=8 recovery check passed: YES / NO
- Verdict: VOID (statistic is degenerate at K=2; must be redone multi-class)
- Consequence: the Neural Cleanse experiment must run with K ≥ 8 classes.

### R3 — Activation Clustering ARI 1.0
- ARI was scored against ground-truth poison labels the defender does not have, on a 50/50 mix.
- Silhouette + FPR (from `results/task0_3_activation_clustering.json`):

  | Poison ratio | Silhouette | Flagged? | FPR on clean model |
  |---|---|---|---|
  | 1% | | | |
  | 5% | | | |
  | 10% | | | |
  | 0% (clean) | — | | |

- Verdict:
- Consequence:

### R4 — FLTrust failed to block
- Defects found (from `docs/phase0-defense-diff.md`): missing update normalisation (F1), unstratified root slice (F3), no trust separation (F5), ...
- Verdict: VOID — the implementation is not faithful FLTrust; the paper's magnitude defense was absent.
- Consequence: Phase 2 reimplements FLTrust per Cao et al. before any "FLTrust fails" claim.

### R5 — FLAME failed to block
- Defects found: KMeans instead of HDBSCAN (the damaging one), median of kept vs all norms, fixed noise sigma.
- Verdict: VOID — not faithful FLAME.
- Consequence: Phase 2 reimplements FLAME per Nguyen et al.

## Dataset audit

_[Counts from `results/task0_5_dataset_audit.json`: total rows, class balance, duplicate columns/rows, sentinel −1 counts, inf/NaN per column, rows lost to dropna, single-label Destination Ports.]_

## Decision

Reframe: YES / NO — reasoning:

Decision rule:
- `ASR_clean >= 0.9` → current backdoor result is void, **reframe is mandatory**
- `ASR_clean ~= 0` → backdoor is real but still unrealizable (G-12) and still DBA (G-06); **reframe strongly recommended** on novelty grounds

Either way, Phases 1 and 2 are unchanged.

---

**Take this report to your guide.** "We stress-tested our own results and found
four measurement errors before building further on them" reads as rigour.
