# Identifying the attackers — the output-concentration detector

This is the one detector in the project that is **ours**, not a re-implementation
from a paper. It exists because the honest result of Phase 2 is negative — no
published defense identifies the malicious clients — and the natural next
question is *can anything?*

Short answer: **yes, for the realizable trigger; no, for the extreme one — and
the reason why is the interesting part.**

## The idea

Every defense in Phase 2 looks at the client update as a whole and asks "is this
an outlier?" On this data the answer is no: the attackers converge to consensus
first (poisoning is an easier objective than the real 8-class task), so they are
the *tightest* cluster, which is why FLAME's score comes out inverted.

Instead of the whole update, look at the one thing an attacker **must** do that
an honest client need not. To make a triggered flow land on the target class,
the attacker has to rewrite the network's **final layer** so those inputs route
to that single class. So the detector ignores the bulk of the update and
measures only how concentrated each client's final-layer change is:

```
e_c   = ||ΔW_out[:, c]||² + Δb_out[c]²          energy pushed toward class c
score = max_c  e_c / Σ_c e_c                     peak share on any one class
```

An honest client on non-IID traffic spreads its final-layer change across the
several families it happens to hold; a targeted attacker piles it onto one.
`flids/fl/aggregators/outconc.py`, aggregator name `outconc_scorer`. Like
`gradnorm_scorer` it aggregates with plain FedAvg and only *reports* the score,
so its ranking AUC is comparable to the other defenses with no aggregation
confound.

## How it was measured (so the number is trustworthy)

The signal was chosen by looking at seeds 0–4. **Every number below is from the
held-out seeds 5–9, which the detector was never tuned on**, produced by

```
python -m scripts.baselines.outconc_eval --processed --seeds 5 6 7 8 9
```

Detector definition was frozen before those seeds were generated: score as
above, run-averaged over the 20 rounds, higher = more suspicious.

## Results — held-out seeds 5–9

| trigger | attackers | detection AUC (mean [min–max]) | precision@k | honest false-positive rate |
|---|---|---|---|---|
| `inbounds_free` (realizable) | 4 of 10 | **0.767 [0.38–1.00]** | 0.65 | **0.00** |
| `inbounds_free` (realizable) | 2 of 10 | **0.875 [0.62–1.00]** | 0.60 | 0.00 |
| `oob_999` (extreme) | 4 of 10 | 0.475 [0.21–0.71] | 0.45 | 0.00 |
| none (attacker-free) | 0 | — | — | **0 / 50** |

Read against the Phase 2 detectors on the same client-scoring task
(`docs/phase2-baselines.md` §2): FLAME **0.227 ± 0.128 inverted**, FLTrust
**0.597 ± 0.111 (flips)**, GradNorm **0.524 ± 0.152 (flips)**. On the realizable
trigger the output-concentration score is the only one that is both above chance
and correctly oriented on every seed.

## What it does and does not show

- **It identifies attackers on the realizable trigger** — the same trigger FLAME
  ranks backwards and Activation Clustering (model-level) cannot see. That is the
  trigger that matters, because it is the one built only from attacker-controlled
  features and the one that persists after the attacker leaves.
- **It cannot detect the extreme `999` trigger** (AUC ≈ chance), and this is a
  *robust* negative, confirmed on held-out seeds: the extreme trigger is so easy
  that it barely moves the final layer and leaves no fingerprint. The easier the
  backdoor, the less there is to detect.
- **It is a ranking aid, not an automatic filter.** The attackers rank high but
  are not extreme MAD-outliers, so a fixed z ≥ 3.5 cutoff flags almost none of
  them (and, usefully, none of the honest clients — FPR 0). Used as intended —
  rank the clients, inspect the top few — it surfaces the attackers; used as an
  automatic remover it does little. This matches the project's standing
  distinction between *detection* and *prevention*.
- **Scope.** n = 5 held-out seeds; one dataset; 2 and 4 attackers of 10; feature
  space only. It is a promising, reproducible result, not a solved problem.

## Where to see it

- **Live:** `python -m flids.dashboard`, tab 1, aggregator `outconc_scorer`,
  trigger `inbounds_free`. Its detection AUC reads the right way up; switch to
  `flame` on the same trigger and the tile flips to INVERTED. On `oob_999` the
  detector drops to chance, like everything else.
- **Reproducible table:** `results/baselines/outconc_eval.csv`.
