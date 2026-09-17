# Federated IDS Backdoor — CIC-IDS2017

Backdoor attacks and **detection** in federated learning for network intrusion
detection. Pure numpy, no torch, deterministic by seed.

The question is not "can a defense stop the backdoor" (none of the four tested
do) but "can the server identify the compromised clients". The result under
investigation is that **FLAME's cosine-similarity score sits below chance on
tabular intrusion data** — it ranks the attackers as the *least* suspicious
clients, because the poisoned objective is easy enough that they converge to
consensus fastest, so they end up nearest the centroid rather than furthest
from it.

**Strength of this claim (n = 5):** raw AUC **0.227 ± 0.128** — below chance in
all five seeds, never the published orientation. Two seeds invert
near-perfectly (6 and 9 of 20 rounds separate all four attackers); the others
are weaker, one only just below chance. Quote the direction and the range, not
one seed.

**No defense removes the attackers.** Across 15 defended runs no round ever
excluded all four; FLAME excluded no one at all; the backdoor succeeds in 25/25
runs. FLTrust and the gradient-norm scorer both **flip orientation between
seeds** (0.597 ± 0.111 and 0.524 ± 0.152) and support no directional claim.
Model-level, Activation Clustering catches the out-of-distribution `999.0`
trigger perfectly but is blind to the realizable in-bounds one. See
`docs/phase2-baselines.md`.

**New here?** Read `HANDOFF.md` first — it records what is done, what is
verified, and what the next person should actually run. `CLAUDE.md` is the
working guidance.

## Quick start

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt      # Windows

# the data is not in the repo (2.3 GB); regenerate it once
./.venv/Scripts/python.exe -m scripts.preprocessing.preprocess --data data/raw

# one experiment -> results/<run_id>/
./.venv/Scripts/python.exe -m flids.runner --config configs/clean_fedavg_real.yaml

# the whole Phase 2 campaign, in dependency order (~5-6 h; NC + AC are most of it)
./.venv/Scripts/python.exe -m scripts.baselines.run_all_real

# the demo - preflight first on presentation day, it must say READY
./.venv/Scripts/python.exe -m scripts.gates.preflight
./.venv/Scripts/python.exe -m flids.dashboard
```

## Scope — feature-space only

The project is scoped to **feature-space** backdoor attacks. Phases 0–2 are the
deliverable, not a stepping stone.

The **problem-space** attack — crafting real packets, shaping traffic through a
Kali VM, round-tripping through CICFlowMeter — is **out of scope and will not be
built**, so `phase-3-*.md` through `phase-6-*.md` are not live plans. Those
documents call that work "the novel contribution"; that framing is outdated and
they are kept only as a record of the original plan. Do not treat the
problem-space attack as remaining scope or as a reason to call the project
incomplete.

## Phase plans and gates

Each plan ends in a **gate** — a checkable condition that must hold before the
next phase starts. Verdicts live in `docs/gate-verdicts.md`; a gate with no
recorded verdict is an open gate.

| Phase | File | Gate | State |
|---|---|---|---|
| 0 | [phase-0-validity-triage.md](phase-0-validity-triage.md) | G0 — validity report + reframe decision | `Reframe: YES` recorded (30-seed sweep), 16/16 — **guide sign-off pending** |
| 1 | [phase-1-foundation-rebuild.md](phase-1-foundation-rebuild.md) | G1 — byte-identical run on 3 machines | see `docs/gate-verdicts.md` |
| 2 | [phase-2-faithful-baselines.md](phase-2-faithful-baselines.md) | G2 — a defense that measurably works | see `docs/gate-verdicts.md` |
| 3–6 | `phase-3..6-*.md` | — | **out of scope** (see above) |

## The demo

`python -m flids.dashboard` opens a local page with four views: a live
federation you can watch train and be poisoned, a backdoor inspector that runs
one real CIC-IDS2017 flow through a *trained* model from `results/` before and
after the trigger is stamped, the recorded-run table, and a side-by-side
comparison of all five defenses on the recorded runs, replayable round by round.

It adds no dependencies (stdlib `http.server`, hand-written HTML/CSS/JS, no
CDN) and never writes to `results/`. The viva walkthrough, the questions to
expect, and the fallbacks if something breaks are in `docs/demo-script.md`; the
night-before summary for presenters is `docs/presentation-brief.md`.

## Owner shorthand

- **M1** — Attack & infrastructure. Owns `flids/attacks/`, `flids/fl/`, `runner.py`.
- **M2** — Model & detection. Owns `flids/models/`, `flids/defenses/`, preprocessing.
- **M3** — Defense & evaluation. Owns `flids/fl/aggregators/`, `flids/eval/`, the dashboard.

Members meet only at `runner.py` and the YAML config schema. Keep it that way.

## Repository layout

- `flids/` — the research library (pure numpy, no torch); `flids/dashboard/` is the demo
- `configs/` — one YAML per experimental condition; `*_real.yaml` runs it on `data/processed/`
- `scripts/` — runnable entrypoints, one package per concern; run with
  `python -m scripts.<group>.<name>`
- `docs/` — prose deliverables, gate verdicts, and the demo script
- `results/` — append-only run outputs, **committed**; `run_id` is the SHA-256 of the
  resolved config, and the runner refuses to overwrite an existing one
- `data/` — git-ignored (2.3 GB, and five files exceed GitHub's 100 MB limit);
  deterministically regenerable from the raw CSVs
- `phase-*.md` — the phase plans · `CLAUDE.md` — working guidance · `HANDOFF.md` — pickup notes

## Standing caveat

Engelen et al. (WTMC 2021) reconstructed and relabelled more than 20% of
CIC-IDS2017. Every number here is against the **original** release, and that
limitation belongs wherever a number is reported. Moving to a corrected release
(Improved CIC-IDS2017 / LYCOS-IDS2017) and reporting both is the recommended
follow-up, not something already done.
