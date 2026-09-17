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

**If you are resuming this project in a fresh session, read `HANDOFF.md` first.**
It records which committed runs are stale, how to regenerate the dataset (it is
not in the repo), and what is actually left to do.

## Scope — feature-space only

The project is scoped to **feature-space** backdoor attacks. Phases 0–2 are the
deliverable, not a stepping stone toward something larger.

The **problem-space** attack — crafting real packets, shaping traffic through a
Kali VM, round-tripping through CICFlowMeter — is **out of scope and will not be
built.** This contradicts the repo's own documents, so be ready for it:
`phase-3-realizable-trigger.md` calls that work "the novel contribution" and says
"the whole project rests on this phase", and `docs/everything-explained.md`
Part 9 lists it under "what is still missing" as "the actual contribution". Those
framings are aspirational and outdated relative to what is actually being
delivered.

Consequently: `phase-3-*.md` through `phase-6-*.md` are **not** the live plan. Do
not treat the problem-space attack as remaining scope, as a gap, or as a reason
to call the project incomplete when assessing it. Do not propose building the
pcap/CICFlowMeter pipeline, Kali traffic shaping, or DBA-style distributed
problem-space attacks unless the project owner reopens that scope explicitly.

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
  dashboard/                  the viva demo: `python -m flids.dashboard`
         engine.py            datasets/models/live sims; never writes to results/
         server.py            stdlib http.server + SSE - no new dependencies
         static/              hand-written HTML/CSS/JS, no CDN (must work offline)
  runner.py                   `python -m flids.runner --config configs/x.yaml`
  model.py, backdoor.py       back-compat shims for Phase 0 imports
configs/                      one YAML per experimental condition; `*_real.yaml`
                              = the same condition on data/processed/
results/<run_id>/             append-only; run_id = sha256(canonical_json(config))[:12]; never overwritten
results/{baselines,calibration,figures,validation}/   script outputs (not per-run)
scripts/                      runnable entrypoints, one package per concern - run with `-m`:
  preprocessing/              preprocess, partition_figures
  validation/                 Phase 0 triage (run_all + the individual tasks)
  baselines/                  Phase 2 measurement: clean_asr, durability, detection_auc,
                              neural-cleanse calibrate/roc, activation_clustering, flame_zero_attacker
                              run_all_real = the whole campaign in dependency order
  gates/                      gate_g1, gate_g2
  _common.py                  shared helpers (paths, load_data, train_model)
docs/                         prose deliverables: phase0-validity-report, phase0-defense-diff,
                              phase1-foundation, phase2-baselines, gate-verdicts, demo-script
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
verdicts still come from a real-data run. The reframe line is answered
(`Reframe: YES`, from a 30-seed `scripts.baselines.clean_asr` sweep); G0 is
16/16 and waits only on the guide. "Reframe" means the detection-first framing
already adopted, **not** the out-of-scope problem-space plan.

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
stale and must be regenerated.** The whole real-data campaign is one command:
`python -m scripts.baselines.run_all_real` (`--seed N` for a sweep). Gate verdicts are recorded in
`docs/gate-verdicts.md` — a gate with no recorded verdict is an open gate.

Report numbers from `scripts.baselines.detection_report` and
`scripts.baselines.prevention_report` (they read `results/` across every seed),
not from `detection_auc.csv`. Two measurement rules learned the hard way:

- **A model-level detector must be shown the poisoned rows.** Activation
  Clustering once clustered the clean global split and "failed" for that reason.
  Calibrate on seeds disjoint from the ones you evaluate, and record each
  model's ASR so "backdoored" is verified rather than assumed.
- **`clean_asr.csv` is merged, never rewritten** — every recorded run's dASR
  reads it. When running several processes in parallel set
  `OPENBLAS_NUM_THREADS=1` (faster; real-data weights are not bit-identical
  across thread counts, and `env.json` records the setting).

## The demo

`python -m flids.dashboard` serves a local page for the viva: a live federation
you can watch train and poison, a backdoor inspector that flips one real flow
through a *trained* model from `results/`, and the recorded-run table. The
walkthrough and its fallbacks are in `docs/demo-script.md`.

Two rules the dashboard must keep:

- **No new dependencies and no CDN.** It is stdlib `http.server` plus
  hand-written HTML/CSS/JS. `requirements.txt` is pinned because Gate G1 wants
  byte-identical runs on three machines, and the page has to open on a laptop
  with no network in front of a judge.
- **It never writes to `results/`.** Live simulations live in memory and are
  discarded. A demo that could manufacture a run_id would make `results/`
  untrustworthy.

## Conventions

- Match the style of the file you're editing; keep comment density similar.
- `scripts/` modules take their dataset from `scripts._common` (`add_data_arg`
  + `load_data` / `resolve_dataset`), never by calling `load_dataset` /
  `synthetic_dataset` directly. Three sources, in precedence order:
  `--processed [DIR]` (cached `data/processed/` arrays — the normal real-data
  path, `--subsample N` draws a class-stratified subset, 0 = full split),
  `--data PATH` (raw CSVs, re-runs the whole preprocessing contract), and no
  flag at all (synthetic fallback, still binary for Phase 0 compatibility).
  All of them write machine-readable output under `results/`
  (`baselines/`, `calibration/`, `figures/`, `validation/`); real-data triage
  results are written as `*_real.json` so they never overwrite synthetic ones.
- **Nothing may assume 2 classes.** `MLP`, `evaluate_model` and
  `evaluate_backdoor` all default to `n_classes=2`; pass `data.n_classes`
  explicitly. Likewise prefer "every class except the target"
  (`source_classes=None`) over the binary `ATTACK` constant — on the 8-family
  data `ATTACK == 1` silently means DoS alone.
- When a phase plan quotes a formula (NC anomaly index, FLTrust normalisation,
  FLAME clustering), implement it exactly as the cited paper defines it.
