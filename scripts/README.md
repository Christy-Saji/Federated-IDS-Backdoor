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
| `scripts.baselines` | Phase 2 faithful-baseline measurements (G2) | `check_triggers`, `clean_asr`, `durability`, `detection_auc`, `flame_zero_attacker`, `nc_calibrate`, `nc_roc`, `activation_clustering` |
| `scripts.gates` | one-machine gate checkers | `gate_g0`, `gate_g1`, `gate_g2` |

`scripts/_common.py` holds the shared helpers: result-dir paths, `add_data_arg`
(the three flags above), `load_data` / `resolve_dataset`, `train_model`,
`backdoor_attack`. Scripts should take their dataset from those rather than
calling `load_dataset` / `synthetic_dataset` directly, so a new source only has
to be wired in once.

Phase-by-phase run order and context: `docs/phase1-foundation.md`,
`docs/phase2-baselines.md`. The phase plans themselves are the `phase-*.md` files
at the repo root.
