# Validity Report — Federated IDS Backdoor Project

Date: 2026-09-03            Authors: M1, M2, M3

## Summary

Of the five results carried into this project, **none stands as originally
stated**: R1 is PARTIAL (the backdoor is real, but the headline ASR of 1.0 was
measured with a trigger that reaches ASR 1.0 on a clean model too) and R2-R5 are
VOID (a degenerate statistic at K=2, a detector scored against labels the
defender does not have, and two defenses that were not faithful to their
papers). Every verdict below comes from a measurement on the real CIC-IDS2017
data, not from re-reading the code.

## Result-by-result

### R1 — Backdoor achieves ASR 1.0
- Original claim: ASR = 1.0 on the backdoored global model
- Measured on the real CIC-IDS2017 arrays (`data/processed/`, 8 families,
  60k-row stratified subsample, 5 rounds, 10 clients, seeds 0/1/2). Clean
  main-task accuracy was ~0.965 across all three seeds, so the models being
  probed are healthy.

  | Trigger | `ASR_clean` per seed | mean `ASR_clean` | mean `ASR_backdoor` | mean dASR |
  |---|---|---|---|---|
  | `999.0` (as reported, out-of-bounds) | 0.0000 / 0.0000 / 1.0000 | **0.3333** | 1.0000 | 0.6667 |
  | `3.0` (in-bounds) | 0.1479 / 0.2636 / 0.2806 | **0.2307** | 0.7955 | 0.5648 |

  Sources: `results/validation/task0_1_trig999_real.json`, `..._trig3_real.json`.
- **The `999.0` control is bimodal**: two seeds give `ASR_clean = 0`, one gives
  `ASR_clean = 1.0`. On that seed a model that has never seen a poisoned sample
  already classifies *every* triggered attack flow as Benign. The stamp pushes
  the row so far outside the training distribution that the network falls back
  to the majority class, which is exactly gap G-01. Three seeds are not enough
  to say how often this happens - the mean of 0.3333 is an average over a
  yes/no event, not a stable rate - so this needs a wider seed sweep before any
  number from it goes in the report.
- The in-bounds `3.0` trigger behaves like a real backdoor: `ASR_clean` is
  consistently non-zero but modest (0.1479 / 0.2636 / 0.2806), `ASR_backdoor` reaches 0.7955, and the
  gap dASR = 0.5648 is stable across seeds.
- Verdict: **PARTIAL**. The backdoor is real - the in-bounds rung shows a
  genuine, reproducible dASR of 0.5648. But the *headline* number is void as
  stated: "ASR = 1.0" was measured with an out-of-bounds stamp that, on at
  least one seed in three, achieves the same 1.0 on a completely clean model.
- Consequence: every ASR in the project is reported as dASR against a
  clean-model control at the same trigger rung (this is now enforced -
  `flids/runner.py` reads `results/baselines/clean_asr.csv` and Gate G2 fails
  any ladder run without a baseline). The `999.0` rung is kept only as the
  unconstrained upper bound; the in-bounds rungs carry the claims.

### R2 — Neural Cleanse detected the backdoor
- Reported statistic: L1 norm 6.3 (class 0) vs 15.5 (class 1) — a raw ratio, not the NC index
- Correct NC anomaly index at K=2: `0.6745` — the predicted constant. It is
  identical for the reported 6.3-vs-15.5 pair, for a no-backdoor 12.0-vs-12.4
  pair, and for a blatant 0.1-vs-900 pair, so at K=2 the index cannot
  distinguish a backdoor from no backdoor at all. Corroborated by CatBack
  (NDSS 2026), which reports the same ~0.67xMAD degeneracy on tabular data.
- K=8 recovery check passed: **YES** — with 8 classes the planted small-norm
  class scores 9.76 against a 0.87 median for the others, well clear
  of Neural Cleanse's anomaly-index threshold of 2.0.
- Source: `results/validation/task0_2_nc_anomaly_index.json` (this task is
  arithmetic on L1 norms, so it does not depend on which dataset was loaded)
- Verdict: VOID (statistic is degenerate at K=2; must be redone multi-class)
- Consequence: the Neural Cleanse experiment must run with K ≥ 8 classes.

### R3 — Activation Clustering ARI 1.0
- ARI was scored against ground-truth poison labels the defender does not have, on a 50/50 mix.
- Re-measured with Chen et al.'s actual decision rule (silhouette > 0.12, not
  ARI) on the real data, at realistic poison ratios, with a clean-model control.
  From `results/validation/task0_3_activation_clustering_real.json`:

  | Poison ratio | Silhouette | Flagged? | Smaller-cluster fraction |
  |---|---|---|---|
  |   1% | 0.993 | **yes** | 0.010 |
  |   5% | 0.993 | **yes** | 0.050 |
  |  10% | 0.993 | **yes** | 0.100 |
  | **0% (clean model)** | 0.323 | **yes - false positive** | 0.431 |

- **The detector flags everything.** At the 0.12 threshold the clean model's
  target-class activations split into two clusters cleanly enough to be called
  poisoned, so the false-positive rate is 100% and the 100% true-positive rate
  below it carries no information. A detector that fires on every model has not
  detected anything.
- The reported ARI of 1.0 measured something else entirely: it was scored
  against ground-truth poison labels the defender does not have, on a
  hand-built 50/50 mix. At a realistic 1% the poisoned points are 1.0% of
  the class, not half of it.
- One thing that *does* track the truth: the smaller cluster's share is
  0.010 / 0.050 / 0.100 at 1% / 5% / 10% poison, against 0.431 on the
  clean model. Cluster *size* separates where the silhouette does not, which is
  the statistic Phase 2's calibrated version should be built on.
- Verdict: **VOID** - ARI against unavailable labels on a 50/50 mix is not the
  Activation Clustering decision rule, and the real rule has a 100% FPR here.
- Consequence: Activation Clustering is re-run in Phase 2 with a threshold
  calibrated on clean models (`flids/defenses/activation_clustering.py`), and
  every detection number is reported with its clean-model FPR beside it.

### R4 — FLTrust failed to block
- Defects found (from `docs/phase0-defense-diff.md`): missing update normalisation (F1), unstratified root slice (F3), no trust separation (F5), ...
- Verdict: VOID — the implementation is not faithful FLTrust; the paper's magnitude defense was absent.
- Consequence: Phase 2 reimplements FLTrust per Cao et al. before any "FLTrust fails" claim.

### R5 — FLAME failed to block
- Defects found: KMeans instead of HDBSCAN (the damaging one), median of kept vs all norms, fixed noise sigma.
- Verdict: VOID — not faithful FLAME.
- Consequence: Phase 2 reimplements FLAME per Nguyen et al.

## Dataset audit

All eight day-CSVs, 2,830,743 raw rows x 79 columns, from
`results/validation/task0_5_dataset_audit.json` (`--data data/raw`).

| Finding | Count | What the pipeline does about it |
|---|---|---|
| Duplicate rows (raw frame) | 308,381 (10.9%) | the pipeline removes 307,078 of them - the small difference is rows already dropped for inf/NaN. Deduplication happens **before** the train/test split; doing it after leaks near-identical flows across the split |
| `Destination Port` values carrying a single label | 52,118 | the column is a near-label proxy and is dropped - but **after** deduplication, since it is the only thing separating one PortScan flow from the next |
| `-1` sentinels, `Init_Win_bytes_forward` | 1,001,189 | kept, not treated as missing: dropping them discarded 50.9% of the dataset, unevenly across classes |
| `-1` sentinels, `Init_Win_bytes_backward` | 1,441,552 | as above |
| `inf` cells (`Flow Bytes/s`, `Flow Packets/s`) | 1,509 / 2,867 | converted to NaN, then the affected rows dropped |
| Rows lost to `dropna` | 2,867 (0.10%) | acceptable loss, counted and reported |
| Duplicate *columns* by name | 0 | none by name, but `Fwd Header Length.1` is a pandas-suffixed copy of `Fwd Header Length` and is dropped explicitly - a name check alone would miss it |

Class balance is severe: BENIGN is 80.3% of rows, and the tail is thin -
Heartbleed 11 rows, Web Attack Sql Injection 21, Infiltration 36 across
the whole dataset. This is why the 14 raw labels are collapsed to 8 families
(`flids/data/labels.py`) and why the stratified subsample used for the runs
above carries a per-class floor: proportional sampling deletes these classes
outright.

The raw label strings also contain a mis-encoded byte: the separator in the
three `Web Attack` labels is 0x96 (a cp1252 en-dash), not an ASCII hyphen, so it
reads as a replacement character under UTF-8. The label map handles it
explicitly; a naive string match on these labels silently drops all three web
attack classes.

**Standing caveat.** Engelen et al. (WTMC 2021) reconstructed and relabelled
more than 20% of CIC-IDS2017 flows. Every number in this project is therefore
reported against the original release with that limitation stated; moving to a
corrected release (Improved CIC-IDS2017 / LYCOS-IDS2017) and reporting both is
the recommended follow-up.

## Decision

Reframe: YES — reasoning: every branch of the decision rule recommends a reframe, and a 30-seed sweep shows the `999.0` rung lands in the "mandatory" branch on about 1 seed in 6 and in the "recommended" branch on the rest. Recorded 2026-09-13 as the team's decision; **still to be signed off by the guide.**

Decision rule (`phase-0-validity-triage.md`, Task 0.1):
- `ASR_clean >= 0.9` → current backdoor result is void, **reframe is mandatory**
- `0.3 < ASR_clean < 0.9` → partially confounded, dASR is the only meaningful number; **reframe strongly recommended**
- `ASR_clean ~= 0` → backdoor is real but still unrealizable (G-12) and still DBA (G-06); **reframe recommended** on novelty grounds

**The n=3 measurement landed in neither branch cleanly** (mean 0.333 at `999.0`,
bimodal 0 / 0 / 1), so the sweep was widened to 30 seeds before answering.
`python -m scripts.baselines.clean_asr --processed --seeds ...`, 20-round clean
FedAvg on the same 60k subsample shape; `results/baselines/clean_asr_summary.csv`:

| rung | seeds | mean `ASR_clean` | seeds with `ASR_clean >= 0.9` | 95% CI on that rate | seeds with `ASR_clean ~= 0` |
|---|---|---|---|---|---|
| `oob_999` (`999.0` stamp) | 30 | 0.173 ± 0.371 | **5 / 30** | 0.06 – 0.35 | 24 / 30 |
| `inbounds_any` | 30 | 0.190 ± 0.058 | 0 / 30 | 0.00 – 0.12 | 0 / 30 |
| `inbounds_free` | 30 | 0.199 ± 0.062 | 0 / 30 | 0.00 – 0.12 | 0 / 30 |

What that settles:

- **The `999.0` stamp is a lottery, not a rate.** 29 of 30 seeds are all or
  nothing (the exception is 0.197). On a clean model the stamp pushes every flow
  so far outside the data that a single class takes ≥ 98% of them in 29 of 30
  seeds — DoS in 17, **Benign in 5**, DDoS in 3, Infiltration in 3, PortScan in
  2 (`top_class` in `clean_asr.csv`). Which class wins is decided by how the
  network extrapolates, not by any backdoor. "ASR = 1.0" at this rung partly
  measures whether Benign happened to win, which it does about one time in six.
- **The rule's answer does not depend on the seed.** Per seed the rung lands in
  "mandatory" 5 times and "recommended" 24 times; every branch recommends a
  reframe. The strength of the recommendation is what varies, not the answer.
- **The in-bounds rungs are sound.** A stable, non-zero clean baseline (≈ 0.19)
  that never fires, so dASR means the same thing at every seed. They carry the
  quantitative claims.

**What the reframe is.** The one this project has already made: the headline
moves off "backdoor ASR = 1.0 with a `999.0` stamp" to whether faithful defenses
can identify the malicious clients (Phase 2, `docs/phase2-baselines.md`), with
the in-bounds rungs carrying any quantitative backdoor claim and `oob_999` kept
only as the unconstrained upper-bound control. It does **not** reopen the
problem-space attack, which is out of scope for this project.

Phases 1 and 2 are unchanged in method. What the answer constrains is reporting:
wherever `oob_999` appears, dASR is degenerate on the seeds whose clean model
already fires (seeds 2 and 4 of the five Phase 2 campaign seeds), so raw ASR is
quoted beside it.

---

**Take this report to your guide.** "We stress-tested our own results and found
four measurement errors before building further on them" reads as rigour.
