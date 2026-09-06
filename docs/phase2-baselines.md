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
| `python -m scripts.baselines.check_triggers` | Task 2.1 sanity check | stdout |
| `python -m scripts.baselines.clean_asr` | Task 2.2 — clean-model ASR per rung × seed | `results/baselines/clean_asr.csv` |
| `python -m scripts.baselines.durability` | Task 2.3 — dASR-vs-round, attacker exits round 20 | `results/figures/durability.png` |
| `python -m scripts.baselines.flame_zero_attacker` | Task 2.7 sanity check (run before any FLAME attack) | exit code |
| `python -m scripts.baselines.detection_auc` | Tasks 2.6–2.9 — per-aggregator detection AUC + FPR | `results/baselines/detection_auc.csv` |
| `python -m scripts.baselines.nc_calibrate` | Task 2.4 step 3 — NC null distribution + p95 threshold | `results/calibration/nc_null.csv` |
| `python -m scripts.baselines.nc_roc` | Task 2.4 step 4 — NC ROC/AUC, TPR, FPR | `results/baselines/nc_roc.csv` |
| `python -m scripts.baselines.activation_clustering` | Task 2.5 — AC TPR/FPR across poison ratios | `results/baselines/activation_clustering.csv` |
| `python -m scripts.gates.gate_g2` | G2 mechanical checklist | stdout |

## Recommended order

```
python -m scripts.baselines.check_triggers
python -m scripts.baselines.clean_asr             # needed before any dASR is reported
python -m flids.runner --config configs/badnets_oob999.yaml
python -m flids.runner --config configs/badnets_fltrust.yaml
python -m flids.runner --config configs/badnets_flame.yaml
python -m flids.runner --config configs/badnets_fltrust_flame.yaml
python -m scripts.baselines.flame_zero_attacker
python -m scripts.baselines.detection_auc
python -m scripts.baselines.durability
python -m scripts.baselines.nc_calibrate          # slow: trains ~10 models
python -m scripts.baselines.nc_roc
python -m scripts.baselines.activation_clustering
python -m scripts.gates.gate_g2
```

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
