# CLAUDE.md

Guidance for working in this repository.

## What this project is

Final-year project: **backdoor attacks and defenses in federated learning for
network intrusion detection (CIC-IDS2017)**. The plan is in `README.md` +
`phase-0..6-*.md`. Gap IDs (G-01 … G-12) referenced in the plans come from a
technical-plan document that is not in this repo yet.

Phases are gated — each ends with a checkable condition (G0 … G5) that must hold
before the next phase starts. Do not start phase N+1 work while phase N's gate
is open.

## Environment

- **Python dependencies go in a project-local venv only.** Create/use `.venv/`
  at the repo root; never `pip install` into the system interpreter.
  - `python -m venv .venv`
  - `./.venv/Scripts/python.exe -m pip install -r requirements.txt`  (Windows)
- Platform is Windows; the shell is PowerShell. A Bash tool is also available.
- Runtime deps: numpy, pandas, scikit-learn, scipy. **No torch** — the models in
  `flids/` are pure numpy on purpose, so any team machine can reproduce a run
  without a GPU stack (Phase 1's G1 needs byte-identical runs on 3 machines).

## Layout

```
flids/                        research package (no torch) - the library
  data/  loaders.py           the one data contract: QuantileTransformer, dedupe-before-split
         labels.py            14 raw CIC-IDS2017 labels -> 8 families (+ binary flag)
         partition.py         dirichlet_partition (alpha=inf = IID) + partition figures
         triggers.py          the 4-rung trigger ladder (oob_999 .. problemspace)
         perturbability.csv   Task 1.8 table: every feature free/partial/fixed + why
  models/ mlp.py              pure-numpy MLP (set_params copies - see Phase 2 note below)
          tabtransformer.py   pure-numpy TabTransformer, forward + hand-written backward
          registry.py         build_model("mlp"|"tabtransformer", ...)
  fl/    server.py            deterministic FedAvg loop (FederatedServer)
         client.py            local training + poisoning window
         aggregators/         fedavg, gradnorm, fltrust, flame, fltrust+flame, gradnorm_scorer
  attacks/ badnets.py         trigger stamping, evaluate_model (acc), evaluate_backdoor (ASR)
  defenses/ neural_cleanse.py  multi-class, range-clamped, MAD anomaly index
           activation_clustering.py  silhouette rule, clean-model calibration
  eval/  metrics.py           delta_asr, main_task_accuracy, detection_auc, defense_fpr, backdoor_lifespan
  utils/ seeding.py           set_all_seeds, env_info
  runner.py                   `python -m flids.runner --config configs/x.yaml`
  model.py, backdoor.py       back-compat shims for Phase 0 imports
configs/                      one YAML per experimental condition
results/<run_id>/             append-only; run_id = sha256(canonical_json(config))[:12]; never overwritten
results/{baselines,calibration,figures,validation}/   script outputs (not per-run)
scripts/                      runnable entrypoints, one package per concern - run with `-m`:
  preprocessing/              preprocess, partition_figures
  validation/                 Phase 0 triage (run_all + the individual tasks)
  baselines/                  Phase 2 measurement: clean_asr, durability, detection_auc,
                              neural-cleanse calibrate/roc, activation_clustering, flame_zero_attacker
  gates/                      gate_g1, gate_g2
  _common.py                  shared helpers (paths, load_data, train_model)
docs/                         prose deliverables: phase0-validity-report, phase0-defense-diff,
                              phase1-foundation, phase2-baselines
phase-0..6-*.md               the phase plans (specs) - kept at repo root
```

Scripts are run from the repo root as modules, e.g.
`python -m scripts.baselines.clean_asr` / `python -m scripts.gates.gate_g1`.

Git is initialised; `env.json` records the commit once there is one.

Ownership (from `README.md`): **M1** attack/infra `flids/attacks`,`flids/fl`,`runner.py` ·
**M2** model/detection `flids/models`,`flids/defenses`,preprocessing ·
**M3** aggregation/eval `flids/fl/aggregators`,`flids/eval`,dashboard. Members
meet only at `runner.py` and the YAML config schema — keep it that way.

## Phases

**Phase 0 — validity triage.** `python -m scripts.validation.run_all` (+ `--data`).
Fill `docs/phase0-validity-report.md` and take it to the guide. Phase 0 measures;
it does not fix. Always 3 seeds. Note: Phase 0's synthetic-demo numbers moved
when Phase 1 switched the synthetic generator to QuantileTransformer — the
real-data path (`--data`) and the MLP itself are unchanged, so the validity
verdicts still come from a real-data run.

**Phase 1 — foundation rebuild.** `python -m scripts.preprocessing.preprocess`,
`python -m scripts.preprocessing.partition_figures`,
`python -m scripts.gates.gate_g1`. Gate G1 =
`python -m flids.runner --config configs/clean_fedavg.yaml` produces a
byte-identical `summary.json` on all three machines. `gate_g1` checks everything
verifiable on one machine (7/7 on synthetic data).

**Phase 2 — faithful baselines.** FLTrust and FLAME are now faithful
reimplementations (`flids/fl/aggregators/`); Neural Cleanse and Activation
Clustering live in `flids/defenses/`. See `docs/phase2-baselines.md` for the run
order and `docs/phase0-defense-diff.md` for the paper-vs-code closeout. **Phase 2
found and fixed an aliasing bug in `MLP.set_params` (it returned views the
optimiser then mutated in place); `results/` run_ids created before that fix are
stale and must be regenerated.**

## Conventions

- Match the style of the file you're editing; keep comment density similar.
- `scripts/` modules take `--data` (optional), stay runnable with no args
  (synthetic fallback), and write machine-readable output under `results/`
  (`baselines/`, `calibration/`, `figures/`, `validation/`).
- When a phase plan quotes a formula (NC anomaly index, FLTrust normalisation,
  FLAME clustering), implement it exactly as the cited paper defines it.
