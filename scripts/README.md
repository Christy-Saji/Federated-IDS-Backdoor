# scripts/

Runnable entrypoints for the project. Everything is a module under the `scripts`
package — run it from the **repo root** with `-m`, never as a file path:

```
python -m scripts.<group>.<name> [--processed | --data path/to/cicids2017]
```

Every script stays runnable with no arguments (synthetic CIC-IDS2017-shaped
fallback) and writes machine-readable output under `results/`.

## Choosing a dataset

Three sources, in precedence order:

| Flag | Source | When |
|---|---|---|
| `--processed [DIR]` | cached arrays in `data/processed/` | **the normal real-data path** |
| `--data PATH` | raw CIC-IDS2017 CSV or directory | first run, or Task 0.5's raw-frame audit |
| *(none)* | synthetic, same 77-feature shape | smoke tests, CI |

`--data` re-runs the entire preprocessing contract (load, dedupe, split, fit the
quantile transformer) on every invocation — minutes per script on 2.8M rows.
`--processed` reads what `scripts.preprocessing.preprocess` already wrote.

`--subsample N` (default 60000) draws a class-stratified subset from
`--processed`; `--subsample 0` uses the full 1.89M-row split. The models are
pure numpy, so the full split is only practical for a final run. The draw keeps
`min(count, 100)` rows of every class before sharing out the remainder, because
proportional allocation alone would round Infiltration's 27 training rows to
zero and silently make it a 7-class problem.

| Group | Purpose | Modules |
|---|---|---|
| `scripts.preprocessing` | build `data/processed/`, partition figures | `preprocess`, `partition_figures` |
| `scripts.validation` | Phase 0 validity triage (G0) | `run_all`, `clean_control`, `nc_anomaly_index`, `activation_clustering`, `dataset_audit` |
| `scripts.baselines` | Phase 2 faithful-baseline measurements (G2) | `run_all_real`, `check_triggers`, `clean_asr`, `durability`, `detection_auc`, **`detection_report`**, **`prevention_report`**, `flame_zero_attacker`, `flame_guard_ablation`, `nc_calibrate`, `nc_roc`, `activation_clustering` |
| `scripts.gates` | one-machine gate checkers, and the presentation-day demo check | `gate_g0`, `gate_g1`, `gate_g2`, `preflight` |

## Running the Phase 2 campaign

```
python -m scripts.baselines.run_all_real          # ~5-6 h on the 60k subsample; NC + AC are most of it
python -m scripts.gates.gate_g2
```

`run_all_real` exists because the Phase 2 scripts have a dependency order that
is easy to get wrong *silently*: `clean_asr` must precede every attack run (the
runner turns ASR into dASR by looking up `results/baselines/clean_asr.csv`, and
writes a `note_dasr` instead if the row is missing), `flame_zero_attacker` must
pass before any FLAME arm, and `nc_calibrate` must precede `nc_roc` (which
otherwise falls back to the paper's default threshold of 2.0 without saying so).
Each step is its own process, so one failure is reported and the campaign
continues — read the summary table it prints at the end.

`detection_report` is the one to look at first: this project is about backdoor
**detection**, so the headline is how well each defense's per-client score
identifies the attackers, not whether it lowers dASR. It trains nothing — every
aggregator already logs `client_scores` per round — and it reports three things
`detection_auc.csv` does not: sign-corrected AUC (an AUC of 0.08 is a *perfect
inverse* ranking, not an absent signal), precision@k, and the per-round curve
that a 20-round mean hides.

`prevention_report` is its counterpart for the other question: accuracy,
macro-F1, raw ASR and dASR per aggregator as mean ± std over every seed in
`results/`, plus per-class F1 re-scored from each run's saved model. Both read
`results/` only, so re-run them after any new seed.

**Seed sweeps.** `run_all_real --seed N` gives each seed its own run_ids and its
own output files. `clean_asr` merges rows into `clean_asr.csv` by
`(trigger, seed)` rather than rewriting it (the runner reads that file for
every seed it has ever run), and takes a lock, so parallel `--seeds` sweeps
compose. Set `OPENBLAS_NUM_THREADS=1` when running several processes at once:
it is about 2x faster per process even alone, and `env.json` records it.

`scripts/_common.py` holds the shared helpers: result-dir paths, `add_data_arg`
(the three flags above), `load_data` / `resolve_dataset`, `train_model`,
`backdoor_attack`. Scripts should take their dataset from those rather than
calling `load_dataset` / `synthetic_dataset` directly, so a new source only has
to be wired in once.

Phase-by-phase run order and context: `docs/phase1-foundation.md`,
`docs/phase2-baselines.md`. The phase plans themselves are the `phase-*.md` files
at the repo root.
