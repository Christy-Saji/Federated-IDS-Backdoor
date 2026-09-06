# Phase 0 — Validity Triage (implementation)

Re-measures the five existing results before anything is built on them. See
`../phase-0-validity-triage.md` for the reasoning and the Gate G0 template.

## Run

Run from the repo root, as modules:

```
.venv/Scripts/python.exe -m scripts.validation.run_all               # synthetic
.venv/Scripts/python.exe -m scripts.validation.run_all --processed   # real, cached
.venv/Scripts/python.exe -m scripts.validation.run_all --data data/raw
```

`--processed` reads the arrays written by `scripts.preprocessing.preprocess`
and takes a stratified subsample (60k train rows by default, `--subsample 0`
for the full 1.89M-row split). The subsample keeps every class: proportional
allocation alone would round Infiltration's 27 training rows to zero and turn
the 8-class problem into a 7-class one.

Task 0.5 audits the *raw* frame, so it needs `--data` and is skipped under
`--processed` alone.

Real-data runs write `*_real.json` alongside the synthetic files rather than
overwriting them, and every Task 0.1 JSON now records `data_source` — a
synthetic number can no longer be mistaken for a real one in the report.

Or individually (each writes JSON/CSV to `results/`):

| Task | Script | What it produces |
|---|---|---|
| 0.1 Clean-model ASR control | `scripts.validation.clean_control` | `ASR_clean`, `ASR_backdoor`, `dASR` for 3 seeds, at trigger `999.0` and in-bounds `3.0` |
| 0.2 Neural Cleanse anomaly index | `scripts.validation.nc_anomaly_index` | Proof the NC index is a constant `0.6745` at K=2; recovery at K=8 |
| 0.3 Activation Clustering | `scripts.validation.activation_clustering` | Silhouette + clean-model FPR at poison ratios 1/5/10% |
| 0.4 Defense diff | `docs/phase0-defense-diff.md` | **Manual** paper-vs-code checklist (needs real FLTrust/FLAME code) |
| 0.5 Dataset audit | `scripts.validation.dataset_audit` | Duplicate cols/rows, sentinels, label-leak ports, inf/NaN, dropna losses |

## Important

* **No `--data` → synthetic CIC-IDS2017-like data** (77 features, same MLP
  shape). The synthetic run exercises the code and shows the *shape* of each
  result; the verdicts in `validity_report.md` must come from a real-data run on
  your actual seeds. On synthetic data the clean-model ASR is ~0 by
  construction — that is not evidence about the real backdoor.
* Phase 0 **measures only**. Do not fix the FLTrust/FLAME defects here — Phase 2
  rebuilds them faithfully.
* Fill `validity_report.md` from `results/` and take it to your guide.

## Deliverable → Gate G0

`validity_report.md`, completed, with a STANDS / VOID / PARTIAL verdict for each
of R1–R5 and a reframe decision.
