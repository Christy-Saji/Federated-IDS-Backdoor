# Phase Plans — Federated IDS Backdoor Project

Seven self-contained phase plans. Each is written so the owner can pick it up and work
from it without re-reading the others, and each ends with a **gate** — a checkable
condition that must hold before the next phase starts.

Read `../FL-IDS-Backdoor-Technical-Plan.md` once for the reasoning behind all of this.
The gap IDs referenced throughout (G-01 … G-12) are defined there.

| Phase | File | Duration | Owner focus | Gate |
|---|---|---|---|---|
| 0 | [phase-0-validity-triage.md](phase-0-validity-triage.md) | 4 days | All three | G0 — validity report + reframe decision |
| 1 | [phase-1-foundation-rebuild.md](phase-1-foundation-rebuild.md) | 2 weeks | All three | G1 — byte-identical run on 3 machines |
| 2 | [phase-2-faithful-baselines.md](phase-2-faithful-baselines.md) | 3 weeks | Parallel tracks | G2 — a defense that actually works |
| 3 | [phase-3-realizable-trigger.md](phase-3-realizable-trigger.md) | 4 weeks | M1 lead | G3 — a pcap carrying the trigger |
| 4 | [phase-4-distributed-attack.md](phase-4-distributed-attack.md) | 3 weeks | M1 lead | G4 — measurable distributed advantage |
| 5 | [phase-5-evaluation-campaign.md](phase-5-evaluation-campaign.md) | 3 weeks | M3 lead | G5 — every claim traces to a run_id |
| 6 | [phase-6-writing-and-defense.md](phase-6-writing-and-defense.md) | 2 weeks | All three | Submission |

**Total: ~18 weeks.**

## The four things that must not slip

1. **Phase 0 finishes in week 1.** Everything downstream depends on knowing whether the
   current results are real.
2. **The perturbability table exists before Phase 3.** The trigger design is meaningless
   without it.
3. **The runner and results schema exist before Phase 2.** 192 runs cannot be managed in
   notebooks.
4. **Traffic shaping is timeboxed with the fallback pre-agreed**, so nobody has to make
   that call under pressure in week 14.

## Owner shorthand

- **M1** — Attack & infrastructure. Owns `flids/attacks/`, `flids/fl/`, the VM lab, `runner.py`.
- **M2** — Model & detection. Owns `flids/models/`, `flids/defenses/`, preprocessing.
- **M3** — Defense & evaluation. Owns `flids/fl/aggregators/`, `flids/eval/`, the dashboard.

Members meet only at `runner.py` and the YAML config schema. Keep it that way.

## Repository layout

- `flids/` — the research library (pure numpy, no torch)
- `configs/` — one YAML per experimental condition
- `scripts/` — runnable entrypoints, one package per concern; run with
  `python -m scripts.<group>.<name>` (see `scripts/README.md`)
- `docs/` — prose deliverables and per-phase run notes
- `results/` — append-only run outputs (git-ignored, regenerated)
- `phase-*.md` — the phase plans above · `CLAUDE.md` — working guidance
