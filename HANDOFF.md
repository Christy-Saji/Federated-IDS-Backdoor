# Handoff — Federated IDS Backdoor (CIC-IDS2017)

Generated: 2026-09-13 (supersedes the 2026-09-12 handoff)
Branch: `main`
Repo: https://github.com/Christy-Saji/Federated-IDS-Backdoor

**Read this first, then `CLAUDE.md`, then `docs/gate-verdicts.md`.**

> **Nothing from 2026-09-12 or 2026-09-13 is committed yet.** The dashboard, the
> seed 1–4 runs, the new configs and scripts, and every doc change below are in
> the working tree only. Commit before anyone else pulls.

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
required**. Those documents describe it as "the actual contribution"; that
framing is outdated. Treat Phases 0–2 as the deliverable. Do not start building
the pcap/CICFlowMeter pipeline unless Christy reopens that scope.

## Current state

### Done & verified

- **Phase 0 — validity triage. Gate G0: `Reframe: YES` recorded, 16/16
  mechanical checks; guide sign-off pending.** The blank reframe line was
  settled with a 30-seed clean-model sweep: the `999.0` stamp sends every flow
  to one arbitrary class on a clean model, Benign in 5/30 seeds (95% CI
  0.06–0.35). Every branch of the decision rule recommends a reframe. See
  `docs/phase0-validity-report.md` § Decision.
- **Phase 1 — foundation rebuild.** Gate G1 7/7 on machine 1, reference digest
  `511f566fc9f5bc59`. Verified identical at 1/4/8/12/16 OpenBLAS threads, so a
  different core count on machines 2–3 should not break it.
- **Phase 2 — faithful baselines. Gate G2 CLOSED, 7/7.** Campaign arms at
  **5 seeds** (0–4), durability at 3, guard ablation at 3, model-level detectors
  re-measured after a validity fix (below). Write-up: `docs/phase2-baselines.md`.
- **The demo.** `python -m flids.dashboard`, four tabs: live federation,
  backdoor inspector, recorded runs, and **Compare defenses** (all five
  defenses side by side on the recorded runs, replayable round by round).
  Presentation-day check: `python -m scripts.gates.preflight` must say READY.
  Word-for-word walkthrough in `docs/demo-script.md`, verified against the
  live page; night-before summary in `docs/presentation-brief.md`.

### The framing that matters — this is a detection project

The deliverable is **whether the server can identify the compromised clients**.
Answer on real CIC-IDS2017: **no defense does.**

- **Decision level.** Over 15 defended runs (3 defenses × 5 seeds), no round
  ever excluded all four attackers. FLAME excluded no one at all. FLTrust
  excluded at least as many honest client-rounds as malicious in 4 of 5 seeds.
  Raw ASR is 1.000 in 25/25 runs.
- **Score level.** FLAME's cosine score is **inverted in all five seeds** (raw
  AUC 0.227 ± 0.128): it ranks the attackers as the *least* suspicious, because
  the poisoned objective is easy and they converge to consensus first. This is
  the one directional finding; strength varies from near-perfect (2 seeds) to
  near-chance (seed 4). FLTrust (0.597 ± 0.111) and GradNorm (0.524 ± 0.152)
  **flip orientation between seeds** — no claim. FLTrust looked "stable" at
  n = 3; seed 3 overturned that.
- **Our own code is ruled out** for FLAME on three seeds: at every re-admit guard
  setting it rejects zero attackers (`docs/phase0-defense-diff.md`).
- **Model level.** Activation Clustering flags the `oob_999` backdoor perfectly
  (AUC 1.00 at every poison ratio, 0 false positives) and is at chance on the
  realizable `inbounds_free` trigger. Neural Cleanse is the mirror image: chance
  on `oob_999` (AUC 0.55; its range clamp cannot express 999), AUC 0.85 on
  `inbounds_free`, but it flags nothing at its calibrated threshold and names
  Benign for every model, clean or backdoored.

Use `scripts.baselines.detection_report` and `scripts.baselines.prevention_report`
for anything that goes in the report — not `detection_auc.csv`, which runs its
own differently-configured attack (script-helper `source_class=2`, poison 0.3).

### Fixed in the 2026-09-13 pass — know these before touching the baselines

1. **Activation Clustering never saw a poisoned row.** The script clustered
   `ds.X_train[y == target]`, the clean global split, so its "silhouette flat
   across poison ratios" was a property of the input. It now clusters the rows
   the federation trained on (`Client._training_data`), and the result reversed
   for `oob_999`.
2. **NC and AC false-positive rates were in-sample.** Both scored the same clean
   models they calibrated on. Calibration and evaluation now use disjoint seeds
   (NC: calibrate 0–9, ROC 100–109; AC: calibrate 100–109, evaluate 0–4).
3. **Neither model-level script verified the backdoor.** Both now record ASR per
   model and use the campaign's attack (`source_class: null`).
4. **`clean_asr` truncated its CSV.** `--seeds 7` would have deleted the seed 0–2
   rows every recorded run's dASR depends on. It now merges by
   `(trigger, seed)` under a lock. Seeds 0–2 were re-run and are byte-identical.
5. **`run_name` kept `_s0` under `--seed N`.** The runner now rewrites the
   suffix (it is not hashed, so no run_id moved). The existing seed 1/2 run
   directories still say `_s0` on disk (append-only); the dashboard corrects the
   label when it displays them.
6. **The prevention table's "honest clients rejected: 0"** read only the final
   round. `detection_report` now counts removals over every round.
7. **`flame_guard_ablation` had no CSV and no seed option.** It now writes
   `flame_guard_ablation[_sN].csv`. The old seed-0 table (AUC 0.225) did not
   reproduce (0.033 at both 1 and 16 threads) and was replaced.

### Still open

- **G0 guide sign-off.** The answer is recorded; the guide has not seen it.
- **Gate G1 cross-machine.** Machines 2 and 3 have not run it. Record digests in
  `docs/gate-verdicts.md`.
- **Base-model weakness to state in the write-up.** WebAttack and Bot have
  F1 ≈ 0 in every run *and* on clean models (≈145 training rows each of 60k,
  non-IID). Macro-F1 ≈ 0.6 is mostly those two, not Infiltration.
- **Thin seeds where they remain.** Durability and the guard ablation are n = 3;
  the FLAME λ sweep and the cosine-mechanism table are seed 0 only. FLAME's
  inversion is 5/5 (sign test p = 0.03) — suggestive, not conclusive.

## Decisions made

- **`source_class: null` on every `_real` attack config** and now in the NC/AC
  scripts too: the clean baseline is measured over every non-target family, so
  a single-source attack would subtract a baseline from a different population.
  The older synthetic configs (`badnets_fltrust.yaml` and friends) still say
  `source_class: 2` — kept so their run_ids stay stable; do not compare them.
- **`configs/clean_fedavg.yaml` stays synthetic.** It is the G1 reference run.
- **`Reframe: YES` means the reframe already made**, not a return to the
  problem-space plan: detection is the headline, in-bounds rungs carry the
  quantitative claims, `oob_999` is only the upper-bound control.
- **The data contract was not changed for Infiltration.** Dropping or merging it
  would re-id every run; `prevention_report` reports macro-F1 with and without it
  instead, and the difference (0.60 → 0.65) is too small to justify the churn.
- **`OPENBLAS_NUM_THREADS=1` for new parallel runs.** About 2x faster per
  process. Real-data weights differ bit-for-bit across thread counts, though
  every summary number compared so far matched; `env.json` records the setting.
- **`data/raw/` and `data/processed/` stay git-ignored; `results/` is
  committed.** Git LFS was rejected — a dependency and a quota for files that
  are deterministically regenerable.
- **The dashboard adds no dependencies** and never writes to `results/`.

## Dead ends / things already tried

- **Staging all of `data/` (2.3 GB).** Caught before committing. If you find
  yourself about to `git add data/`, that is the mistake.
- **Treating the `-1` sentinels in `Init_Win_bytes_*` as missing.** Dropped
  50.9% of the dataset unevenly across classes.
- **Dropping `Destination Port` before deduplication.** Collapsed PortScan by
  98.8%. It is now a dedupe key, dropped immediately after.
- **Reading AC's flat silhouette as a detector result.** It was the input (fix 1
  above). If a model-level detector does not respond to poisoning, first check
  that the poisoned rows are actually in what it is looking at.

## Open questions

1. **Infiltration** (27 train rows vs a rare threshold of 50). Numbers now
   exist: excluding it lifts macro-F1 by ~0.05. WebAttack and Bot are the bigger
   problem, and nothing about them is decided.
2. **Standing caveat:** Engelen et al. (WTMC 2021) relabelled more than 20% of
   CIC-IDS2017. State it wherever a number is reported; a corrected release
   (Improved CIC-IDS2017 / LYCOS-IDS2017) is the recommended follow-up.

## Next steps

1. **Commit** the working tree (see the note at the top).

2. **Set up** (fresh machine):
   ```
   python -m venv .venv
   ./.venv/Scripts/python.exe -m pip install -r requirements.txt
   ```
   Project-local venv only.

3. **Get the data.** `MachineLearningCSV.zip` from
   https://www.unb.ca/cic/datasets/ids-2017.html into `data/raw/`, then
   `python -m scripts.preprocessing.preprocess --data data/raw`.
   Infiltration should have 27 training rows; Heartbleed folds into DoS.

4. **Close the cross-machine half of G1** on machines 2 and 3:
   `python -m scripts.gates.gate_g1`, compare against `511f566fc9f5bc59`.

5. **Take `docs/phase0-validity-report.md` to the guide** with the reframe
   answer, and `docs/phase2-baselines.md` with the detection result.

6. **Reproduce or extend** (optional):
   ```
   python -m scripts.baselines.run_all_real --seed 5 --only clean_asr flame_zero attack fltrust flame combined gradnorm
   python -m scripts.baselines.detection_report
   python -m scripts.baselines.prevention_report
   python -m scripts.gates.gate_g0
   python -m scripts.gates.gate_g2
   ```

## Environment notes

- **Windows + PowerShell** is the reference platform.
- **No torch, deliberately.** Pure numpy.
- **Install from the pins.** Version drift breaks byte-identical reproducibility.
- Scripts run from the repo root as modules.
- **Nothing may assume 2 classes.** Pass `data.n_classes` explicitly; prefer
  `source_classes=None` over the binary `ATTACK` constant.
- Ownership: **M1** attacks / FL / runner · **M2** models / defenses /
  preprocessing · **M3** aggregators / eval / dashboard.
