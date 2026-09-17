# Phase 2 — Faithful Baselines

Reimplement every attack, detector, and defense so it matches its paper, and
measure everything with controls. After this phase "defense X fails against
attack Y" is a statement about the defense, not a bug.

Entry condition: Gate G1 cleared. Everything here runs on the synthetic fallback
with no args. For the real thing, preprocess once and then use the cache:

```
python -m scripts.preprocessing.preprocess --data data/raw   # writes data/processed/
python -m scripts.baselines.clean_asr --processed            # 60k stratified rows
python -m scripts.baselines.clean_asr --processed --subsample 0   # full split
```

`--data` still works and re-runs the whole preprocessing contract on every
invocation; `--processed` reads the cached arrays and is what the baseline
sweep should use. The subsample is stratified with a per-class floor so the
rare families (Infiltration: 27 training rows) survive the draw.

## What changed in `flids/`

| File | Change |
|---|---|
| `flids/data/triggers.py` | **new** — the 4-rung trigger ladder (`oob_999`, `inbounds_any`, `inbounds_free`, `problemspace`), `feature_stats`, `apply_trigger` |
| `flids/attacks/badnets.py` | `stamp_trigger` / `poison_split` / `evaluate_backdoor` take an optional trigger `spec` + train `stats`; the Phase 0 bare-`value` path still works |
| `flids/models/mlp.py` | `set_params` now **copies** — it previously aliased the global vector so `fit` mutated it in place (broke FLTrust, silently degraded FedAvg) |
| `flids/fl/server.py` | resolves the trigger rung, carves a stratified FLTrust root set, logs per-client defense scores + detection AUC to `metrics.jsonl` |
| `flids/runner.py` | reports `dasr_final` / `dasr_peak` / `dasr_by_round` when `results/baselines/clean_asr.csv` has a matching `(trigger, seed)` row |
| `flids/fl/aggregators/fltrust.py` | **faithful** — ReLU(cos) trust + the missing normalise-to-`‖g0‖` step; root-set server update |
| `flids/fl/aggregators/flame.py` | **faithful** — HDBSCAN (`sklearn.cluster.HDBSCAN`, no new dep) on model-cosine distance, clip to median of **all** norms, adaptive noise `σ = λ·S` |
| `flids/fl/aggregators/gradnorm.py` | adds `gradnorm_scores` + `GradNormScorer` (MAD norm anomaly, detection-only) |
| `flids/fl/aggregators/combined.py` | **new** — `fltrust+flame` (FLAME filters, FLTrust weights survivors) |
| `flids/defenses/neural_cleanse.py` | **new** — per-class, sigmoid mask, range-clamped pattern, Adam, λ schedule, MAD anomaly index |
| `flids/defenses/activation_clustering.py` | **new** — silhouette rule, threshold calibrated on clean models, optional exclusionary reclassification |

Config schema: `attack.trigger` now takes `{name, target_label, source_class}`;
`federated.aggregator` gains `gradnorm_scorer` and `fltrust+flame`. The old
`attack.trigger.value` still works as a fallback.

## Scripts (run from the repo root as modules; all write to `results/`)

| Command | Deliverable | Output |
|---|---|---|
| `python -m scripts.baselines.run_all_real` | **the whole campaign in dependency order** (`--seed N`, `--only`, `--skip`) | everything below |
| `python -m scripts.baselines.check_triggers` | Task 2.1 sanity check | stdout |
| `python -m scripts.baselines.clean_asr` | Task 2.2 — clean-model ASR per rung × seed; merges by `(trigger, seed)`, never truncates | `results/baselines/clean_asr.csv`, `clean_asr_summary.csv` |
| `python -m scripts.baselines.durability` | Task 2.3 — dASR-vs-round, attacker exits round 20 | `results/figures/durability*.png` |
| `python -m scripts.baselines.flame_zero_attacker` | Task 2.7 sanity check (run before any FLAME attack) | exit code |
| `python -m scripts.baselines.detection_auc` | Tasks 2.6–2.9 — per-aggregator AUC + FPR (script-helper attack, not the campaign's — superseded by `detection_report`) | `results/baselines/detection_auc*.csv` |
| `python -m scripts.baselines.detection_report` | **the detection table** — per-seed AUC, orientation, P@k, removals; reads `results/` | `results/baselines/detection_report.csv` |
| `python -m scripts.baselines.prevention_report` | prevention across seeds + per-class F1; reads `results/` | `results/baselines/prevention_report.csv` |
| `python -m scripts.baselines.flame_guard_ablation` | is FLAME's "rejects nobody" ours or the paper's? (`--seed N`) | `results/baselines/flame_guard_ablation*.csv` |
| `python -m scripts.baselines.nc_calibrate` | Task 2.4 step 3 — NC null distribution + p95 threshold (seeds 0–9) | `results/calibration/nc_null.csv` |
| `python -m scripts.baselines.nc_roc` | Task 2.4 step 4 — NC ROC on out-of-sample seeds 100+, backdoor verified (`--trigger`) | `results/baselines/nc_roc*.csv` |
| `python -m scripts.baselines.activation_clustering` | Task 2.5 — AC across poison ratios, on the rows the federation trained on (`--trigger`) | `results/baselines/activation_clustering*.csv` |
| `python -m scripts.gates.gate_g2` | G2 mechanical checklist | stdout |

## Recommended order

`run_all_real` encodes it; use that rather than running steps by hand. For a
seed sweep of the campaign arms:

```
python -m scripts.baselines.run_all_real                  # seed 0, everything (~2-3 h)
python -m scripts.baselines.run_all_real --seed 3 --only clean_asr flame_zero attack fltrust flame combined gradnorm
python -m scripts.baselines.detection_report
python -m scripts.baselines.prevention_report
python -m scripts.gates.gate_g2
```

Set `OPENBLAS_NUM_THREADS=1` when running several of these in parallel. It is
roughly 2x faster per process even alone (OpenBLAS oversubscribes on these small
matrices), and `env.json` records the setting. Real-data weights are not
bit-identical across thread counts, although every summary number checked so far
has been.

## Known limits (synthetic data)

- `oob_999` fires at ASR ≈ 1.0 on a **clean** synthetic model, so its dASR ≈ 0
  there. That is the point of G-01 — on real CIC-IDS2017 the OOD trigger has a
  real dASR. Use `--data` for any number that goes in the report.
- Neural Cleanse and Activation Clustering barely separate on synthetic
  activations. Task 2.4 budgets 3–5 days of NC tuning on real data; if it still
  fails to separate, that is a citable finding (cf. CatBack, NDSS 2026), not a
  broken run.
- `results/` run_ids created before the `MLP.set_params` fix are stale —
  regenerate them.

---

# Results — real CIC-IDS2017

Everything below is the 60k stratified subsample, α = 0.5, 10 clients, 4
malicious (clients 0–3), `oob_999` unless stated, 20 rounds. Client-level
detection and prevention are **n = 5** (seeds 0–4); durability is n = 3;
model-level detectors use 5–10 seeds each, as stated. Gate G2 closed 7/7
(`docs/gate-verdicts.md`).

**This project is about backdoor *detection*, not prevention.** The prevention
table is reported first because it is the shorter story and because it sets up
why the detection numbers are the interesting ones.

## 1. Prevention — no defense reduces the attack, at any seed

`python -m scripts.baselines.prevention_report` (reads `results/`, trains nothing).

**Raw ASR = 1.000 for every aggregator at every seed** (25/25 campaign runs).

| aggregator (n = 5) | accuracy | macro-F1 | macro-F1 w/o Infiltration | raw ASR | dASR |
|---|---|---|---|---|---|
| fedavg (no defense) | 0.972 ± 0.004 | 0.595 ± 0.063 | 0.648 ± 0.047 | **1.000** | 0.600 ± 0.490 |
| fltrust | 0.964 ± 0.003 | 0.548 ± 0.035 | 0.627 ± 0.040 | **1.000** | 0.600 ± 0.490 |
| flame | 0.972 ± 0.001 | 0.621 ± 0.022 | 0.689 ± 0.026 | **1.000** | 0.600 ± 0.490 |
| fltrust+flame | 0.964 ± 0.003 | 0.548 ± 0.035 | 0.627 ± 0.040 | **1.000** | 0.600 ± 0.490 |
| gradnorm_scorer | 0.972 ± 0.004 | 0.595 ± 0.063 | 0.648 ± 0.047 | **1.000** | 0.600 ± 0.490 |

Two rows duplicate others **by design**, not by accident:
`gradnorm_scorer` is detection-only (it scores clients but aggregates with plain
FedAvg), and `fltrust+flame` uses FLAME only as a pre-filter before FLTrust —
FLAME removes nobody in any campaign run, so the combined arm reduces exactly to
FLTrust.

Phase 2's own fallback clause names this outcome as a legitimate close: with
faithful implementations it is a statement about the defenses, not a bug.

> **Quote raw ASR here, not dASR, and say why.** dASR is 1.000 at seeds 0, 1, 3
> and **0.000 at seeds 2 and 4 for every aggregator** — not because anything
> defended, but because `oob_999`'s *clean-model* ASR is already 1.0 at those
> seeds (`clean_asr.csv`), so the subtraction removes the whole signal. That is
> the ± 0.490.

### `oob_999` is a lottery on a clean model — the 30-seed sweep

The seed-2 anomaly first looked like a one-off; it is not. `clean_asr` over 30
seeds (`results/baselines/clean_asr_summary.csv`):

| rung | seeds | mean clean ASR | seeds where it is ≥ 0.9 | 95% CI | where a clean model sends stamped flows |
|---|---|---|---|---|---|
| `oob_999` | 30 | 0.173 | **5 / 30** | 0.06 – 0.35 | one class takes ≥ 98% in 29/30 seeds: DoS 17, Benign 5, DDoS 3, Infiltration 3, PortScan 2 |
| `inbounds_any` | 30 | 0.190 ± 0.058 | 0 / 30 | 0.00 – 0.12 | spread; plurality DoS |
| `inbounds_free` | 30 | 0.199 ± 0.062 | 0 / 30 | 0.00 – 0.12 | spread; plurality DoS |

The `999.0` stamp pushes a flow so far outside the data that the network's
extrapolation, not any backdoor, picks one class for all of them. Whether that
class is Benign (the target) is roughly a one-in-six event per seed. G-01's
subtraction handles it correctly, but it means:

1. **Prefer `inbounds_any` / `inbounds_free` for anything quantitative.** Their
   clean baselines are stable (≈ 0.19) and never degenerate.
2. This is what answers Gate G0: `docs/phase0-validity-report.md` now records
   `Reframe: YES` on this evidence.

### Per-class F1 — the base IDS misses three families

Accuracy 0.97 hides that macro-F1 is ~0.6. The per-class F1 (mean over the 25
campaign runs) says why:

| Benign | DoS | DDoS | PortScan | BruteForce | WebAttack | Bot | Infiltration |
|---|---|---|---|---|---|---|---|
| 0.981 | 0.936 | 0.991 | 0.989 | 0.623 | **0.000** | **0.015** | 0.118 |

**WebAttack and Bot are essentially never detected** — those test flows are
classified Benign. This is not the backdoor: two *clean* FedAvg models (seeds 0,
1) show the same pattern (WebAttack 0.000 / 0.000, Bot 0.000 / 0.000,
BruteForce 0.000 / 0.827). The cause is data: 147 WebAttack, 142 Bot and 27
Infiltration training rows out of 60k, split non-IID across 10 clients, 20
rounds, no class weighting. Excluding Infiltration lifts macro-F1 only from
0.595 to 0.648 — **Infiltration is not the main drag; WebAttack and Bot are.**
This bears on HANDOFF open question 2: dropping or merging Infiltration alone
would not fix macro-F1.

FLAME's noise parameter (σ = λ·S) was swept to check whether the defense has
*any* working operating point (seed 0):

| λ | accuracy | macro-F1 | ASR |
|---|---|---|---|
| 0.001 (paper default) | 0.918 | 0.560 | 1.000 |
| 0.05 | 0.925 | 0.634 | 1.000 |
| 0.1 | 0.758 | 0.294 | 1.000 |
| 0.3 | 0.694 | 0.159 | 1.000 |
| 1.0 | 0.265 | 0.123 | 0.001 |

The backdoor dies only at λ = 1.0, where accuracy (0.265) is worse than a
model that always predicts Benign (≈0.80 of the test split is Benign). **There is no λ at which FLAME both
works and leaves a usable model.**

## 2. Detection, client-level — *which client is lying?*

`python -m scripts.baselines.detection_report` (trains nothing; reads the
per-round `client_scores` every aggregator already logs).

### 2a. Does the server actually remove the attackers? No.

The question an operator asks first. 4 attackers × 20 rounds = 80 chances per
run to exclude an attacker; 6 honest × 20 = 120 chances to wrongly exclude an
honest client.

| detector | attackers excluded, s0 / s1 / s2 / s3 / s4 (of 80) | honest wrongly excluded (of 120) | rounds with all 4 attackers out | final ASR |
|---|---|---|---|---|
| FLAME | **0 / 0 / 0 / 0 / 0** | 0 / 0 / 0 / 0 / 0 | 0 in every run | 1.000 ×5 |
| FLTrust (trust clipped to 0) | 2 / 23 / 5 / 2 / 3 | 2 / 5 / 16 / 14 / 3 | 0 in every run | 1.000 ×5 |
| GradNorm scorer *(flags only, by design)* | 2 / 12 / 1 / 3 / 4 | 6 / 12 / 13 / 15 / 8 | 0 in every run | 1.000 ×5 |

**In 15 defended runs, no defense ever had all four attackers out in the same
round**, FLAME never excluded anyone, and FLTrust excluded at least as many
honest client-rounds as malicious ones in 4 of 5 seeds (more in two, equal in
two) — only seed 1 leans the right way. Final-round `removed_clients` is `[]`
for most of these runs, which is why an earlier prevention table reported "0
honest clients rejected" — that column counted the last round only.

### 2b. Do the scores at least rank the attackers?

Per seed, n = 5 (chance: AUC 0.5, precision@4 = 0.40):

| detector | seed | raw AUC | orientation | corrected AUC | precision@4 | perfect rounds |
|---|---|---|---|---|---|---|
| FLAME cosine | 0 | 0.075 | inverted | 0.925 | 0.82 | **9 / 20** |
| FLAME cosine | 1 | 0.300 | inverted | 0.700 | 0.56 | 0 / 20 |
| FLAME cosine | 2 | 0.106 | inverted | 0.894 | 0.79 | **6 / 20** |
| FLAME cosine | 3 | 0.229 | inverted | 0.771 | 0.61 | 0 / 20 |
| FLAME cosine | 4 | 0.423 | inverted | 0.590 | 0.36 | 0 / 20 |
| FLTrust trust | 0 | 0.748 | as-published | 0.748 | 0.62 | 0 / 20 |
| FLTrust trust | 1 | 0.652 | as-published | 0.692 | 0.53 | 0 / 20 |
| FLTrust trust | 2 | 0.636 | as-published | 0.645 | 0.62 | 0 / 20 |
| FLTrust trust | 3 | 0.425 | **inverted** | 0.619 | 0.44 | 0 / 20 |
| FLTrust trust | 4 | 0.526 | as-published | 0.655 | 0.46 | 0 / 20 |
| GradNorm MAD | 0 | 0.601 | as-published | 0.684 | 0.50 | 1 / 20 |
| GradNorm MAD | 1 | 0.564 | as-published | 0.614 | 0.45 | 0 / 20 |
| GradNorm MAD | 2 | 0.246 | **inverted** | 0.758 | 0.62 | 0 / 20 |
| GradNorm MAD | 3 | 0.699 | as-published | 0.743 | 0.61 | 1 / 20 |
| GradNorm MAD | 4 | 0.510 | as-published | 0.635 | 0.41 | 0 / 20 |

Across seeds:

| detector | raw AUC mean ± std (n = 5) | orientation | verdict |
|---|---|---|---|
| FLAME cosine | **0.227 ± 0.128** | **inverted, 5/5** | direction solid; strength varies from near-perfect to near-chance |
| FLTrust trust | 0.597 ± 0.111 | as-published 4/5, **inverted 1/5** | weak; **no stable directional claim** |
| FLTrust+FLAME | 0.597 ± 0.111 | identical to FLTrust | reduces to FLTrust |
| GradNorm MAD | 0.524 ± 0.152 | as-published 4/5, **inverted 1/5** | **no claim available** |

**FLAME — the one directional finding, and a claim about the design
assumption.** Below chance in all five seeds, never once as-published. Two
seeds invert near-perfectly (6 and 9 of 20 rounds separate all four attackers),
two moderately (0.229, 0.300), and seed 4 only weakly (0.423). Defensible
sentence: *"FLAME's cosine score is below chance on tabular intrusion data in
all five seeds (raw AUC 0.227 ± 0.128); the attackers look the least
suspicious clients, strongly so in two seeds of five."* Do **not** quote 0.925
bare. Five of five below chance has a one-sided sign-test p of 0.03 —
suggestive, not conclusive, and worth saying exactly that way.

**FLTrust — the n = 3 verdict did not survive n = 5.** At three seeds it looked
"stable and boring" (0.679 ± 0.049). Seed 3 inverts (0.425) and seed 4 is near
chance (0.526): 0.597 ± 0.111. It is a weak detector whose orientation is not
guaranteed, and it detects without preventing. Even when the ranking is right,
trust-weighting scales a malicious update down rather than removing it — after
normalising onto ‖g₀‖ the four attackers still contributed 0.426 / 1.511 =
**28% of every aggregated update** at seed 0.

**GradNorm — no claim available.** Inverted at seed 2, as-published elsewhere,
mean at chance. Consistently below chance is a finding; bouncing across chance
is noise. `detection_report` prints `ORIENTATION NOT STABLE` for exactly this.

> The corrected AUC and precision@4 columns choose each run's orientation
> *after* seeing which clients were malicious. They are upper bounds on what an
> operator could get, not deployable numbers. For FLAME the orientation is the
> same in every seed, so a fixed "flip the score" rule would reach them. For
> FLTrust and GradNorm no fixed rule would.

### Why FLAME's score is inverted

The mechanism, measured directly (cosine similarity between update deltas,
seed 0):

| round | cos(mal, hon) | cos(hon, hon) | cos(mal, mal) | ‖update‖ mal ÷ hon |
|---|---|---|---|---|
| 0 | +0.464 | +0.622 | +0.710 | 1.10 |
| 1 | +0.360 | +0.328 | +0.379 | 0.82 |
| 2 | +0.254 | +0.257 | +0.334 | 0.64 |

FLAME assumes a malicious update is an **outlier**. On tabular IDS data the
poisoned objective (three columns → "Benign") is trivially easier than the real
8-class task, so the attackers converge first, their gradient on the poisoned
rows collapses, and their models end up **closest to consensus** — the tightest
cluster, not the furthest one. Final-round model-cosine distances (×10⁻⁵,
seed 0): malicious `[2.31, 2.01, 1.92, 1.93]`, honest `[1.98, 3.64, 2.07, 5.78,
2.86, 2.64]`.

**This is FLAME's, not ours — now on three seeds.** The `3×median` re-admit
guard in `flids/fl/aggregators/flame.py` is a local addition, so it was ablated
(`scripts.baselines.flame_guard_ablation`, table in
`docs/phase0-defense-diff.md`). Across seeds 0–2 and five guard settings, **not
one attacker is rejected at any setting**, while honest rejections climb to 33;
detection AUC stays below chance throughout. Every setting tight enough to
reject anyone rejects honest clients in a federation with **no attacker at
all** — re-creating G-05. No clustering threshold fixes a sign error.

## 3. Detection, model-level — *is this model backdoored at all?*

Both detectors were re-measured after three validity problems in the first
version were fixed (details in each script's docstring):

- **Activation Clustering never saw a poisoned row.** It clustered the clean
  global training split, which contains no stamped row at any poison ratio, so
  its flat silhouette measured the input, not the defense. It now clusters the
  target-class rows the federation actually trained on.
- **Both FPRs were in-sample.** The clean models used for the false-positive
  row were the same models the threshold was calibrated on. Calibration and
  evaluation now use disjoint seeds.
- **Neither verified the backdoor.** Every model now reports its ASR, and the
  attack matches the campaign (`source_class: null`; `poison_ratio: 0.5` for
  NC) instead of the script helper's binary-era `source_class=2`.

### Activation Clustering — perfect on `oob_999`, blind on the realizable rung

Threshold = p95 of 10 clean silhouettes (seeds 100–109) = 0.408; 5 models per
ratio (seeds 0–4). "Poisoned share" is the stamped rows as a fraction of the
target class.

| rung | poison ratio | poisoned share | silhouette | flagged | AUC vs clean | poison in suspect cluster | ASR |
|---|---|---|---|---|---|---|---|
| — | 0 (clean) | 0 | 0.348 ± 0.034 | **0 / 5 (FPR 0)** | — | — | — |
| `oob_999` | 1% | 0.08% | 0.991 | **5 / 5** | **1.00** | **100%** | 1.000 |
| `oob_999` | 5% | 0.42% | 0.991 | 5 / 5 | 1.00 | 100% | 1.000 |
| `oob_999` | 10% | 0.85% | 0.991 | 5 / 5 | 1.00 | 100% | 1.000 |
| `oob_999` | 30% | 2.5% | 0.991 | 5 / 5 | 1.00 | 100% | 1.000 |
| `inbounds_free` | 1% | 0.08% | 0.345 | 0 / 5 | 0.44 | 24% | 0.354 |
| `inbounds_free` | 5% | 0.42% | 0.350 | 0 / 5 | 0.48 | 19% | 0.469 |
| `inbounds_free` | 10% | 0.85% | 0.350 | 0 / 5 | 0.48 | 19% | 0.496 |
| `inbounds_free` | 30% | 2.5% | 0.349 | 0 / 5 | 0.48 | 19% | 0.528 |

(Clean-model ASR at seeds 0–4: 0.400 for `oob_999` — the lottery above — and
0.199 for `inbounds_free`.)

**This reverses the earlier write-up.** The first version reported AC's
silhouette as "flat at 0.352–0.354 across poison ratios — the cleaner negative
result". That was the bug. Shown the poisoned rows, AC separates `oob_999`
perfectly at every ratio down to 0.08% of the class, and the suspect cluster is
exactly the poison. **On the realizable in-distribution trigger it is at
chance** (AUC 0.44–0.48, silhouette identical to clean, suspect cluster not the
poison) even though that backdoor works (ASR 0.35–0.53 against a clean 0.20).

So AC finds a trigger that is out of distribution — which is what makes it
unrealizable — and misses the one built from attacker-controlled features. That is the
ladder's argument in one table. Caveat: AC needs the training rows, which in
federated learning live on the clients; this measures whether the rows reveal
the backdoor, not whether a server could run it.

### Neural Cleanse

Threshold = p95 of 10 clean models (seeds 0–9) = **13.55**; ROC on 10
clean/backdoored pairs at seeds 100–109, disjoint from calibration. Attack
`source_class: null`, `poison_ratio: 0.5`, 15 rounds.

| rung | backdoor ASR vs clean (mean) | AUC | backdoored > clean twin | TPR / FPR at 13.55 | class NC names |
|---|---|---|---|---|---|
| `oob_999` | 1.000 vs 0.007 — implanted 10/10 | **0.55** | 4 / 10 | 0.00 / 0.00 | Benign, 20/20 models |
| `inbounds_free` | 0.615 vs 0.211 — dASR 0.40 | **0.85** | **9 / 10** | 0.00 / 0.00 | Benign, 20/20 models |

The mirror image of Activation Clustering:

- **On `oob_999` NC is at chance, and structurally so.** The pattern is clamped
  to the observed feature range (the plan requires it, or every class looks
  backdoored), so the true `999.0` stamp is a value NC is not allowed to express.
  It is looking for a trigger it cannot represent.
- **On the realizable rung the anomaly index does rank backdoored models
  higher** — AUC 0.85, 9 of 10 pairs in the right order (one-sided sign test
  p ≈ 0.01). That is the first model-level signal on the in-distribution trigger
  anywhere in this project. Ten pairs is still a wide interval.
- **Operationally it flags nothing.** The calibrated threshold is set by the
  null's tail: 9 of 10 clean models score 2.2–5.6 and one scores 20.07, and with
  n = 10 the p95 sits almost on that outlier. At 13.55 TPR = FPR = 0. Choosing a
  lower threshold from the ROC set would be fitting to the test data, so no such
  number is reported.
- **Its class attribution is uninformative.** NC names Benign on every model,
  clean or backdoored. Benign is 82% of the training data, so the smallest mask
  that pushes flows into it is naturally small. With an attack that also targets
  Benign, "NC found the target class" carries no information.

Against the earlier write-up ("AUC 0.375 on 4 seeds, too few to call"): that
version reused its calibration models, never checked the backdoor worked, and
used a single-family attack. The comparable `oob_999` number is now 0.55 on 10
out-of-sample pairs — still chance, now for a stated reason.

## 4. Durability — the in-distribution rungs persist; `oob_999` is uninterpretable

`scripts.baselines.durability`: attacker poisons rounds 1–20, then stays in the
federation and trains honestly for 80 more. No defense. dASR at round 100:

| rung | seed 0 | seed 1 | seed 2 | decays after attacker exits? |
|---|---|---|---|---|
| `inbounds_any` | 0.228 | 0.272 | 0.218 | **never, all 3 seeds** |
| `inbounds_free` | 0.218 | 0.192 | 0.215 | **never, all 3 seeds** |
| `oob_999` | 0.000 (lifespan 17) | **1.000 (never)** | 0.000 (degenerate) | **no claim available** |

**The in-distribution rungs are the result.** Tight across seeds and fully
persistent: the backdoor survives 80 rounds of honest training after the
attacker stops reinforcing it. The rung that is realizable in feature space
(`inbounds_free`) is also the one that lasts — and, from §3, the one Activation
Clustering cannot see.

**`oob_999` supports no durability claim in either direction.** Seed 0 decays,
seed 1 never does, and seed 2 is degenerate: its clean-model ASR is already
1.0, so dASR = 0 every round and nothing is measured. The 30-seed sweep in §1
shows that happens on about one seed in six, which is reason enough to keep
`oob_999` out of durability conclusions altogether.

## Caveats on every number above

- **Seeds.** Prevention and client-level detection n = 5; durability n = 3; the
  guard ablation n = 3; AC 10 calibration + 5 per ratio; NC 10 calibration + 10
  ROC pairs; the clean-ASR sweep n = 30. The FLAME λ sweep and the
  cosine-mechanism table are seed 0 only.
- **Five seeds moved one conclusion.** FLTrust went from "stable detector" at
  n = 3 to "orientation not stable" at n = 5. The FLAME direction held (5/5) but
  its mean weakened from 0.160 to 0.227. Treat every n ≤ 5 claim here as
  provisional in exactly that way.
- **Upper-bound columns.** Corrected AUC and precision@k pick orientation after
  seeing the labels (see §2b).
- **Base-model weakness.** WebAttack and Bot are near F1 0 even without an
  attack (§1); "accuracy 0.97" must not be quoted without that.
- **60k stratified subsample**, not the full 1.89M-row split.
- **Original CIC-IDS2017.** Engelen et al. (WTMC 2021) relabelled more than 20%
  of it; that caveat belongs wherever these numbers are reported.
