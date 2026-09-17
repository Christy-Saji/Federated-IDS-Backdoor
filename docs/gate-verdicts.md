# Gate verdicts

Each phase plan ends in a gate. This file is the record of whether it is open or
closed. **A gate with no verdict here is an open gate** — the previous handoff
lost a week to "the scripts exist, so presumably they pass", which is not the
same thing as having run them.

Re-record a verdict whenever the thing it depends on changes. A verdict is
pinned to a commit and a machine, not to the project.

---

## Gate G0 — validity report + reframe decision

**Status: DECISION RECORDED — `Reframe: YES`. Awaiting the guide's sign-off.**

```
python -m scripts.gates.gate_g0        ->  G0: 16/16 mechanical checks pass
```

`docs/phase0-validity-report.md` is written, four of the five original headline
results were measured VOID or PARTIAL on real CIC-IDS2017, and the reframe line
is now answered. At n = 3 the measurement sat between branches of the decision
rule (mean 0.333 at `999.0`, bimodal 0 / 0 / 1), so the clean-model sweep was
widened to **30 seeds** before answering (`results/baselines/clean_asr_summary.csv`):

| rung | seeds | mean `ASR_clean` | seeds ≥ 0.9 ("mandatory") | 95% CI | seeds ≈ 0 ("recommended") |
|---|---|---|---|---|---|
| `oob_999` | 30 | 0.173 | **5 / 30** | 0.06 – 0.35 | 24 / 30 |
| `inbounds_any` | 30 | 0.190 ± 0.058 | 0 / 30 | 0.00 – 0.12 | 0 / 30 |
| `inbounds_free` | 30 | 0.199 ± 0.062 | 0 / 30 | 0.00 – 0.12 | 0 / 30 |

On a clean model the `999.0` stamp sends ≥ 98% of flows to **one** class in
29 of 30 seeds: DoS 17, Benign 5, DDoS 3, Infiltration 3, PortScan 2. Whether
it is Benign is a per-seed lottery (about 1 in 6), which is why n = 3 looked
bimodal. **Every branch of the rule recommends a reframe**; the seed only
changes how strongly. The reframe is the one already made — the headline is
client-level detection with faithful defenses, the in-bounds rungs carry
quantitative claims, `oob_999` is only the upper-bound control. It does not
reopen the out-of-scope problem-space work.

**What G0 constrains in Phase 2:** `oob_999` dASR is structurally 0 on seeds
whose clean model fires — seeds 2 and 4 of the five campaign seeds — so raw ASR
is quoted beside it everywhere, and `oob_999` carries no durability conclusion.

Still needed: take the report to the guide. The mechanical gate cannot check that.

---

## Gate G1 — byte-identical `summary.json` on three machines

**Status: PASSES on machine 1 of 3.** Cross-machine check outstanding.

```
python -m scripts.gates.gate_g1        ->  G1: 7/7 checks pass
```

| check | result |
|---|---|
| runner executes `configs/clean_fedavg.yaml` | pass |
| re-run refuses to overwrite | pass |
| `summary.json` reproducible bit-for-bit | pass — digest `511f566fc9f5bc59` |
| `env.json` populated (versions, commit, seed) | pass |
| partition figures for all four alphas | pass |
| `perturbability.csv` covers every feature, no blanks | pass — `missing=[] blanks=[]` |
| multi-class macro-F1 sane (0.5–0.999) | pass — 0.9103 |

The perturbability check is stronger than it was at the last handoff: with
`data/processed/feature_names.json` present it now validates coverage against
the **real** 76 CIC-IDS2017 feature names rather than the synthetic `fNN`
placeholders.

### The reference number the other two machines must reproduce

Run `python -m flids.runner --config configs/clean_fedavg.yaml` and compare
`results/9863521c42a9/summary.json`:

```json
"final": {
  "round": 19,
  "accuracy": 0.98575,
  "macro_f1": 0.9102735922586803,
  "removed_clients": []
}
```

sha256 of the sorted `final` block, first 16 hex: **`511f566fc9f5bc59`**

Recorded on:

| | |
|---|---|
| machine | `Windows-10-10.0.26200-SP0` |
| commit | `b0db73dbf85a3ea54551283e7cfd5d953dfba1f7` |
| python | 3.11.9 |
| numpy / scipy / sklearn | 2.4.6 / 1.17.1 / 1.9.0 |

**If machine 2 or 3 gets a different digest, compare `env.json` first** —
version drift is the likely cause, which is why `requirements.txt` is pinned.
Note this reference run is on the *synthetic* generator on purpose: G1 must be
checkable on a machine that has not downloaded the 2.3 GB dataset.

**Core count is not a risk for this digest — checked.** OpenBLAS partitions
matrix multiplies by thread count, and on the real data the trained weights are
*not* bit-identical across thread counts. The G1 digest was re-run at
`OPENBLAS_NUM_THREADS` = 1, 4, 8, 12 and the default (16 on machine 1): all five
give `511f566fc9f5bc59`. A teammate's laptop with a different core count should
still match. `env.json` now records `openblas_num_threads` for every new run.

Machine 2: _not yet run._
Machine 3: _not yet run._

---

## Gate G2 — on `oob_999` with correct defenses, at least one defense reduces dASR meaningfully

**Status: CLOSED — the "all defenses fail = finding" branch.** Phase 2's own
text names this outcome explicitly as an acceptable close: *"if all defenses
still fail at dASR = 1.0, that is now a genuine finding, not a bug"* — and with
faithful implementations, that is what real CIC-IDS2017 shows.

```
python -m scripts.gates.gate_g2      ->  G2: 7/7 mechanical checks pass
```

| check | result |
|---|---|
| every ladder attack run reports dASR | pass |
| `clean_asr.csv` covers all 3 rungs | pass |
| Neural Cleanse has a calibrated threshold + FPR | pass |
| Activation Clustering has a clean-FPR row | pass |
| FLAME zero-attacker check passes | pass |
| FLTrust performs the norm_g0/norm_gi step | pass |
| detection AUC table exists for all aggregators | pass |

### Prevention: none of the four defenses lower the attack

**Raw ASR = 1.000 for every aggregator at every seed (25/25 runs, seeds 0–4).**
Quote that, not dASR: at seeds 2 and 4 `oob_999`'s *clean-model* baseline is
1.0, so dASR is structurally 0.000 there for every arm (G0 above).

Real CIC-IDS2017, 60k subsample, α=0.5, 4/10 malicious, `oob_999`, 20 rounds,
mean ± std over seeds 0–4 (`scripts.baselines.prevention_report`):

| aggregator | accuracy | macro-F1 | raw ASR | attackers excluded, total over 5 runs (of 400 client-rounds) | honest wrongly excluded (of 600) |
|---|---|---|---|---|---|
| fedavg (no defense) | 0.972 ± 0.004 | 0.595 ± 0.063 | **1.000** | — | — |
| fltrust | 0.964 ± 0.003 | 0.548 ± 0.035 | **1.000** | 35 | 40 |
| flame | 0.972 ± 0.001 | 0.621 ± 0.022 | **1.000** | **0** | 0 |
| fltrust+flame | 0.964 ± 0.003 | 0.548 ± 0.035 | **1.000** | 35 | 40 |
| gradnorm_scorer | 0.972 ± 0.004 | 0.595 ± 0.063 | **1.000** | 22 *(flagged only)* | 54 *(flagged only)* |

**In no round of any defended run were all four attackers excluded at once.**
An earlier version of this table said "0 honest clients rejected" for FLTrust;
that column read only the final round. `fltrust+flame` equals `fltrust` because
FLAME's pre-filter removes no one; `gradnorm_scorer` equals `fedavg` because it
is detection-only by design.

**Macro-F1 ≈ 0.6 is a base-model weakness, not the attack:** WebAttack and Bot
have F1 ≈ 0 in every run *and* on clean models (147 and 142 training rows of
60k). Excluding Infiltration only lifts macro-F1 to 0.65. See
`docs/phase2-baselines.md` §1.

FLAME's own noise parameter (σ = λ·S) *can* suppress the backdoor — swept
λ ∈ {0.001 … 1.0} and dASR only drops once λ = 1.0, where accuracy collapses to
0.265 (far worse than predicting "Benign" for everything, which scores ≈0.80 on the test split). **There is no
setting at which FLAME both works and leaves a usable model.**

### Detection is the finding this project actually has — see `docs/phase2-baselines.md`

| detector (client-level) | raw AUC, mean ± std (n=5) | orientation | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|---|
| FLAME cosine | **0.227 ± 0.128** | **inverted, 5/5** | 0.075 | 0.300 | 0.106 | 0.229 | 0.423 |
| FLTrust trust | 0.597 ± 0.111 | **flips** (inverted at s3) | 0.748 | 0.652 | 0.636 | 0.425 | 0.526 |
| FLTrust+FLAME | 0.597 ± 0.111 | identical to FLTrust | 0.748 | 0.652 | 0.636 | 0.425 | 0.526 |
| GradNorm MAD | 0.524 ± 0.152 | **flips** (inverted at s2) | 0.601 | 0.564 | 0.246 | 0.699 | 0.510 |

**FLAME: the only stable direction — below chance in all five seeds.** It ranks
the attackers as the least suspicious clients because their trigger objective
is easy and they converge to consensus fastest. Strength varies: two seeds
invert near-perfectly (9 and 6 of 20 rounds separate all four attackers), seed 4
only weakly. Quote the direction and the range, not 0.925 (seed 0's
sign-corrected figure).

**FLTrust: the n = 3 "stable detector" verdict did not survive n = 5.** Seed 3
inverts. Weak, orientation not guaranteed, and it prevents nothing.

**GradNorm: no claim available.** Bouncing across chance is noise.

**The guard ablation now rules out our own code on three seeds.** Across
`readmit_tol_mult` ∈ {3, 2, 1, 0.5, 0} and seeds 0–2, FLAME rejects **zero
attackers at every setting** while honest rejections climb to 33, and every
setting that rejects anyone fails the zero-attacker check
(`docs/phase0-defense-diff.md`).

Model-level detectors (is *this model* backdoored, no client visibility) — both
re-measured after fixing three validity problems (AC never saw a poisoned row;
both FPRs were in-sample; neither verified the backdoor):

| detector (model-level) | result |
|---|---|
| Activation Clustering, `oob_999` | **flags 20/20 poisoned models at every ratio (down to 0.08% of the class), 0/5 clean; AUC 1.00; suspect cluster = the poison** |
| Activation Clustering, `inbounds_free` | **0/20 flagged, AUC 0.44–0.48** — blind to the realizable trigger although it works (ASR 0.35–0.53 vs clean 0.20) |
| Neural Cleanse, `oob_999` | AUC 0.55 on 10 out-of-sample pairs, every backdoor verified (ASR 1.00) — chance; the range clamp cannot express a `999.0` trigger |
| Neural Cleanse, `inbounds_free` | **AUC 0.85, 9/10 pairs ordered correctly** — but TPR = FPR = 0 at the calibrated threshold (13.55, set by one clean outlier at 20.07), and it names Benign on every model, clean or not |

The earlier "AC silhouette flat at 0.352–0.354" row was the bug, not a result.

### Durability — in-distribution rungs persist; `oob_999` is uninterpretable

`scripts.baselines.durability`, attacker exits at round 20, 100 rounds total,
no defense. dASR at round 100, n=3:

| rung | s0 | s1 | s2 | decays? |
|---|---|---|---|---|
| `inbounds_any` | 0.228 | 0.272 | 0.218 | **never, 3/3** |
| `inbounds_free` | 0.218 | 0.192 | 0.215 | **never, 3/3** |
| `oob_999` | 0.000 | **1.000** | 0.000 | **no claim** |

Both in-distribution rungs are tight across seeds and fully persistent — the
backdoor survives all 80 post-exit rounds. The rung realizable in feature space
is the one that lasts, and the one Activation Clustering cannot see.

`oob_999` supports no durability claim either way: seed 0 decays, seed 1 never
does, and seed 2 is degenerate (clean-model ASR already 1.0). The 30-seed sweep
under G0 shows that degeneracy is a ~1-in-6 event, not a one-off.
