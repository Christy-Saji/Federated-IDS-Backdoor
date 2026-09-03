# scripts/

Runnable entrypoints for the project. Everything is a module under the `scripts`
package — run it from the **repo root** with `-m`, never as a file path:

```
python -m scripts.<group>.<name> [--data path/to/cicids2017]
```

Every script stays runnable with no arguments (synthetic CIC-IDS2017-shaped
fallback) and writes machine-readable output under `results/`.

| Group | Purpose | Modules |
|---|---|---|
| `scripts.preprocessing` | build `data/processed/`, partition figures | `preprocess`, `partition_figures` |
| `scripts.validation` | Phase 0 validity triage (G0) | `run_all`, `clean_control`, `nc_anomaly_index`, `activation_clustering`, `dataset_audit` |
| `scripts.baselines` | Phase 2 faithful-baseline measurements (G2) | `check_triggers`, `clean_asr`, `durability`, `detection_auc`, `flame_zero_attacker`, `nc_calibrate`, `nc_roc`, `activation_clustering` |
| `scripts.gates` | one-machine gate checkers | `gate_g1`, `gate_g2` |

`scripts/_common.py` holds the shared helpers (result-dir paths, `load_data`,
`train_model`, `backdoor_attack`).

Phase-by-phase run order and context: `docs/phase1-foundation.md`,
`docs/phase2-baselines.md`. The phase plans themselves are the `phase-*.md` files
at the repo root.
