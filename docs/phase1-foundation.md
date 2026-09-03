# Phase 1 — Foundation Rebuild (implementation)

Builds the substrate every later experiment runs on. Nothing here is a result;
everything here is what makes the later results trustworthy. See
`../phase-1-foundation-rebuild.md` for the reasoning.

## What was built

| Task | Location | Notes |
|---|---|---|
| 1.1 Corrected dataset + preprocessing | `flids/data/loaders.py` | One data contract. `QuantileTransformer(output_distribution='normal')` fit on train only; sentinels→NaN; dedupe **before** split; `save_processed` / `load_processed` |
| 1.2 Multi-class label map | `flids/data/labels.py` | 14 raw labels → 8 families; `\x96` en-dash handled; Heartbleed→DoS, Infiltration kept but flagged rare; `--binary` ablation flag |
| 1.3 Dirichlet partitioning | `flids/data/partition.py` | `dirichlet_partition(y, n_clients, alpha)`, `alpha=inf` = IID; `plot_partitions` for the four figures |
| 1.4 Package scaffold + runner | `flids/runner.py`, `configs/*.yaml` | `run_id = sha256(canonical_json(config))[:12]`; writes `results/<run_id>/{config.yaml,metrics.jsonl,summary.json,model_final.npz,env.json}`; refuses overwrite |
| 1.5 Determinism | `flids/utils/seeding.py` | `set_all_seeds`; partition / poison / init seeded separately. **No torch** in this project, so the CUDA knobs from the plan don't apply |
| 1.6 Models | `flids/models/` | `mlp.py` (ported from Phase 0), `tabtransformer.py` (pure-numpy, forward+backward), `registry.py` |
| 1.7 Metrics | `flids/eval/metrics.py` | `delta_asr`, `main_task_accuracy` (acc + macro-F1), `detection_auc`, `defense_fpr`, `backdoor_lifespan`, `summarise_seeds` |
| 1.8 Perturbability table | `flids/data/perturbability.csv` + `.py` | one row per CIC-IDS2017 feature, free/partial/fixed + justification; 22 `free` (expected 20–25) |

FLTrust and FLAME aggregators were scaffolded in Phase 1 and are now faithful
reimplementations (Phase 2) — see `docs/phase2-baselines.md`.

## Run

Run from the repo root:

```
# preprocessing (Task 1.1/1.2)
.venv/Scripts/python.exe -m scripts.preprocessing.preprocess --data <cicids_csv_or_dir>
.venv/Scripts/python.exe -m scripts.preprocessing.preprocess              # synthetic

# partition figures (Task 1.3)
.venv/Scripts/python.exe -m scripts.preprocessing.partition_figures

# one experiment
.venv/Scripts/python.exe -m flids.runner --config configs/clean_fedavg.yaml

# Gate G1 checklist (what can be verified on one machine)
.venv/Scripts/python.exe -m scripts.gates.gate_g1
```

## Gate G1

`python -m flids.runner --config configs/clean_fedavg.yaml` must produce a
**byte-identical `summary.json` on all three machines**. `gate_g1.py` checks
single-machine reproducibility, the no-overwrite guarantee, `env.json`, the
partition figures, perturbability coverage, and a sane macro-F1. The
cross-machine check is manual: run it on each machine and `diff` the
`summary.json` `final` block.

## Caveats

- **No dataset yet.** Every script falls back to a synthetic CIC-IDS2017-shaped
  generator. Synthetic multi-class is *optimistic* (macro-F1 ≈ 0.99); on the
  real 8-class problem expect noticeably lower — that is correct, not a
  regression.
- Perturbability coverage is checked against real feature names only once
  `data/processed/feature_names.json` exists; until then only well-formedness
  is checked.
- The numpy TabTransformer is correct but slow relative to a torch build; keep
  `d_model=32` and round counts modest.
