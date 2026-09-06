# Handoff — Federated IDS Backdoor (CIC-IDS2017)

Generated: 2026-09-06
Branch: `main` (clean, synced with `origin/main` at `fa4877f`)
Repo: https://github.com/Christy-Saji/Federated-IDS-Backdoor

**Read this first, then `CLAUDE.md`, then `docs/phase2-baselines.md`.** The single
thing most likely to waste your day is at the top of "Current state": most of the
committed `results/` are stale and must not be quoted.

## Goal

Final-year project on **backdoor attacks and defenses in federated learning for
network intrusion detection**, using CIC-IDS2017. Work is organised into gated
phases (`phase-0..6-*.md`); each ends in a checkable condition that must hold
before the next begins.

**Scope note that overrides the phase plans:** the project is scoped to
**feature-space attacks only**. The problem-space work described in
`phase-3-realizable-trigger.md`, `phase-4-distributed-attack.md`, and
`docs/everything-explained.md` — crafting real packets, shaping traffic through
a Kali VM, round-tripping through CICFlowMeter — is **out of scope and not
required**. Those documents describe it as "the actual contribution" and list it
under "what is still missing"; that framing is outdated. Treat Phases 0–2 as the
deliverable, not as a stepping stone. Do not start building the pcap/CICFlowMeter
pipeline unless Christy reopens that scope.

## Current state

### Done & verified

- **Phase 0 — validity triage.** Complete, run on real CIC-IDS2017 data. Four of
  the five original headline results were measured to be VOID or PARTIAL; see
  `docs/phase0-validity-report.md` for the per-result verdicts. Outputs are in
  `results/validation/` (`*_real.json` = real data, unsuffixed = synthetic).
- **Phase 1 — foundation rebuild.** Preprocessing contract, Dirichlet
  partitioning, deterministic FedAvg loop, seeding, and the `runner.py` +
  `results/<run_id>/` schema. Partition figures in `results/partition_viz/`.
- **Phase 2 — faithful baselines.** FLTrust and FLAME reimplemented per their
  papers (`flids/fl/aggregators/`), Neural Cleanse and Activation Clustering
  rebuilt (`flids/defenses/`). Paper-vs-code closeout in
  `docs/phase0-defense-diff.md`.
- **Repo published.** 199 files tracked. Code, configs, docs, phase plans and
  `results/` are all on GitHub.

### Done but NOT verified

- **Gate G1 and Gate G2 were not re-run when this handoff was written.** The
  scripts exist (`scripts/gates/gate_g1.py`, `gate_g2.py`) but no pass/fail
  verdict is recorded anywhere in `docs/`. Re-run both yourself before assuming
  either gate is closed. G1 additionally requires a byte-identical
  `summary.json` on three machines — **you are machine #2**, so running it is
  real progress, not a formality. Record the hash you get.

### Known-bad / in progress

**12 of the 16 committed runs in `results/` are stale.** They were produced
before the `MLP.set_params` aliasing fix (commit `2a834f8`, 2026-09-03 16:43),
where `set_params` returned views that the optimiser then mutated in place. That
broke FLTrust and silently degraded FedAvg. Their numbers are wrong.

**Valid (post-fix) — safe to use:**

| run_id | config | data |
|---|---|---|
| `3d12e09900b7` | `smoke_real` | real |
| `9863521c42a9` | `clean_fedavg_dir0.5_s0` | synthetic |
| `9ae88a6e7216` | `clean_fedavg_real_dir0.5_s0` | real |
| `0c274a4a6be4` | `badnets_oob999_real_dir0.5_s0` | real |

**Stale (pre-fix) — regenerate before quoting:**

| run_id | config |
|---|---|
| `df0118d4a910` | `clean_fedavg_binary_dir0.5_s0` |
| `7603dbb4e2b8` | `clean_tabtransformer_dir0.5_s0` |
| `b903b572258b` | `badnets_gradnorm_s0` |
| `5cd1fde4d76e` | `badnets_fedavg_dir0.5_s0` |
| `0648b0608eca` | `durability_inbounds_any_s0` |
| `da02b4754066` | `durability_inbounds_free_s0` |
| `8632522691c5` | `durability_oob_999_s0` |
| `bf1be1d981c0` | `badnets_fltrust_s0` |
| `ff374e6b23de` | `badnets_oob999_fedavg_s0` |
| `0b0c197018b8` | `badnets_flame_s0` |
| `90cb2f8297c2` | `badnets_fltrust_flame_s0` |
| `f5df5a72c1c6` | `badnets_gradnorm_scorer_s0` |

**This bites you mechanically, not just statistically.** `run_id` is
`sha256(canonical_json(config))[:12]` and `results/` is append-only and never
overwritten — so re-running any of those configs targets a directory that
already exists holding stale contents. Delete the stale directory first.

Two aborted runs were also committed with no `summary.json`: `614315db9f31` and
`d5d569565863` (the latter is empty). Both are safe to delete.

## Decisions made this session

- **Decision:** `data/raw/` and `data/processed/` stay git-ignored; `results/` is
  now committed. **Why:** `data/` is 2.3 GB and five individual files exceed
  GitHub's 100 MB per-file hard limit (`X_train.npy` 1.1 GB, `X_test.npy` 383 MB,
  Wednesday 225 MB, Monday 177 MB, Tuesday 135 MB) — the push would have been
  rejected outright. `results/` is only 7.6 MB across 106 files. **Alternatives
  considered:** Git LFS (rejected — adds a dependency and a quota for files that
  are deterministically regenerable); committing nothing (rejected — you would
  lose the record of what has already been run).
- **Decision:** the data is regenerated on your machine rather than transferred.
  **Why:** the preprocessing contract is deterministic by design, and that
  determinism is exactly what Gate G1 tests. If your regenerated arrays do not
  reproduce the committed numbers, that is a finding worth reporting, not an
  inconvenience to work around.

## Dead ends / things already tried

- **Tried:** staging all of `data/` (2.3 GB) for the initial commit by clearing
  the ignore rules. **Result:** caught and unstaged before committing;
  `git log --all -- data/` confirms no `data/` blob ever entered history.
  **Why it did not work:** the per-file 100 MB limit above. Do not re-attempt
  this — if you find yourself about to `git add data/`, that is the mistake.

## Open questions / unresolved

1. **The reframe decision in `docs/phase0-validity-report.md` is still blank.**
   The line literally reads `Reframe: YES / NO — reasoning:`. The measurement
   lands in neither branch of the stated decision rule: mean `ASR_clean` is 0.333
   at the `999.0` rung and 0.231 in-bounds — far short of the 0.9 "reframe is
   mandatory" threshold, but nowhere near 0 either. The `999.0` figure is also
   bimodal across seeds (0 / 0 / 1), so it is an average over a yes/no event
   rather than a rate. **A wider seed sweep should settle this before the line is
   answered.** This needs the guide's input; it is not yours to decide alone.
   Phases 1 and 2 are unaffected either way.
2. **Whether the stale runs should be deleted from the repo or kept with a
   warning** was raised and left unsettled. This document is currently the
   warning.
3. **Standing caveat for anything written up:** Engelen et al. (WTMC 2021)
   reconstructed and relabelled more than 20% of CIC-IDS2017 flows. Every number
   here is against the original release, and that limitation must be stated
   wherever a number is reported. Moving to a corrected release (Improved
   CIC-IDS2017 / LYCOS-IDS2017) and reporting both is the recommended follow-up,
   not something already done.

## Next steps

1. **Set up.** Clone, then:
   ```
   python -m venv .venv
   ./.venv/Scripts/python.exe -m pip install -r requirements.txt
   ```
   Project-local venv only — never `pip install` into the system interpreter.

2. **Get the data.** Download CIC-IDS2017 from the Canadian Institute for
   Cybersecurity: https://www.unb.ca/cic/datasets/ids-2017.html. Take
   `MachineLearningCSV.zip` and unzip the 8 CSVs into `data/raw/`, so you have
   `Monday-WorkingHours.pcap_ISCX.csv` through
   `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`.

3. **Regenerate `data/processed/`** (writes roughly 1.5 GB):
   ```
   python -m scripts.preprocessing.preprocess --data data/raw
   ```
   Check the printed class-counts table against the generated
   `data/processed/preprocessing_report.md`. Infiltration should have 27 training
   rows, and Heartbleed should be folded into the DoS family.

4. **Verify you can reproduce a known-good run.** Re-run `clean_fedavg_real` and
   confirm you land on run_id `9ae88a6e7216` with a matching `summary.json`. This
   is the machine-#2 half of Gate G1. If it does not match, stop and compare
   `env.json` — library version drift is the likely cause.

5. **Clear the stale runs, then re-run them.** Delete the 12 stale run_ids listed
   above plus `614315db9f31` and `d5d569565863`, then:
   ```
   python -m scripts.baselines.check_triggers
   python -m scripts.baselines.clean_asr --processed    # needed before any dASR
   python -m flids.runner --config configs/badnets_oob999.yaml
   python -m flids.runner --config configs/badnets_fltrust.yaml
   python -m flids.runner --config configs/badnets_flame.yaml
   python -m flids.runner --config configs/badnets_fltrust_flame.yaml
   python -m scripts.baselines.flame_zero_attacker
   python -m scripts.baselines.detection_auc
   python -m scripts.baselines.durability
   python -m scripts.baselines.nc_calibrate      # slow: trains ~10 models
   python -m scripts.baselines.nc_roc
   python -m scripts.baselines.activation_clustering
   ```

6. **Run both gates and record the verdicts** in `docs/`:
   ```
   python -m scripts.gates.gate_g1
   python -m scripts.gates.gate_g2
   ```

7. **Coordinate with Christy on the reframe line** before anything goes to the
   guide.

## Repo state at handoff time

- Branch: `main`, tracking `origin/main`; both at `fa4877f`
- Uncommitted changes: none — working tree clean
- Stashes: none
- Last commits:
  - `fa4877f` results as well
  - `225c1e3` initital commit
  - `2a834f8` Fix two preprocessing interactions that silently corrupted the real dataset
  - `c36e6e5` Phase 0-2: validity triage, foundation rebuild, faithful baselines
- Not in the repo: `data/raw/`, `data/processed/`, `.venv/` — all git-ignored

## Environment / setup notes

- **Windows + PowerShell** is the reference platform. Paths in the docs use
  `./.venv/Scripts/python.exe`; adjust to `.venv/bin/python` on macOS or Linux.
- **No torch, deliberately.** Every model in `flids/` is pure numpy so that any
  team machine can reproduce a run without a GPU stack — Gate G1 needs
  byte-identical runs on three machines. Do not introduce a deep-learning
  dependency.
- Runtime deps are numpy, pandas, scikit-learn, scipy, matplotlib and PyYAML,
  pinned in `requirements.txt`. **Version drift will break byte-identical
  reproducibility**, so install from the pins rather than from latest.
- Scripts run from the repo root as modules:
  `python -m scripts.baselines.clean_asr`.
- Scripts take their dataset from `scripts._common`, never by calling
  `load_dataset` directly. Three sources, in precedence order: `--processed [DIR]`
  (cached arrays — the normal real-data path; `--subsample N` draws a
  class-stratified subset, `0` means the full split), `--data PATH` (raw CSVs,
  re-runs the whole preprocessing contract on every invocation), and no flag at
  all (synthetic fallback, still binary, for Phase 0 compatibility).
- **Nothing may assume 2 classes.** `MLP`, `evaluate_model` and
  `evaluate_backdoor` all default to `n_classes=2` — pass `data.n_classes`
  explicitly. Prefer `source_classes=None` ("every class except the target") over
  the binary `ATTACK` constant; on the 8-family data `ATTACK == 1` silently means
  DoS alone.
- Ownership, so you do not collide: **M1** attacks / FL / runner · **M2** models /
  defenses / preprocessing · **M3** aggregators / eval / dashboard. Members meet
  only at `runner.py` and the YAML config schema — keep it that way.
