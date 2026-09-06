# Phase 1 — Foundation Rebuild (implementation)

Builds the substrate every later experiment runs on. Nothing here is a result;
everything here is what makes the later results trustworthy. See
`../phase-1-foundation-rebuild.md` for the reasoning.

## What was built

| Task | Location | Notes |
|---|---|---|
| 1.1 Corrected dataset + preprocessing | `flids/data/loaders.py` | One data contract. `QuantileTransformer(output_distribution='normal')` fit on train only; -1 sentinels kept (`sentinel_policy`); dedupe **before** split and **before** dropping `Destination Port`; `save_processed` / `load_processed` |
| 1.2 Multi-class label map | `flids/data/labels.py` | 14 raw labels → 8 families; `\x96` en-dash handled; Heartbleed→DoS, Infiltration kept but flagged rare; `--binary` ablation flag |
| 1.3 Dirichlet partitioning | `flids/data/partition.py` | `dirichlet_partition(y, n_clients, alpha)`, `alpha=inf` = IID; `plot_partitions` for the four figures |
| 1.4 Package scaffold + runner | `flids/runner.py`, `configs/*.yaml` | `run_id = sha256(canonical_json(config))[:12]`; writes `results/<run_id>/{config.yaml,metrics.jsonl,summary.json,model_final.npz,env.json}`; refuses overwrite |
| 1.5 Determinism | `flids/utils/seeding.py` | `set_all_seeds`; partition / poison / init seeded separately. **No torch** in this project, so the CUDA knobs from the plan don't apply |
| 1.6 Models | `flids/models/` | `mlp.py` (ported from Phase 0), `tabtransformer.py` (pure-numpy, forward+backward), `registry.py` |
| 1.7 Metrics | `flids/eval/metrics.py` | `delta_asr`, `main_task_accuracy` (acc + macro-F1), `detection_auc`, `defense_fpr`, `backdoor_lifespan`, `summarise_seeds` |
| 1.8 Perturbability table | `flids/data/perturbability.csv` + `.py` | one row per CIC-IDS2017 feature, free/partial/fixed + justification; 22 `free` (expected 20–25) |

FLTrust and FLAME aggregators were scaffolded in Phase 1 and are now faithful
reimplementations (Phase 2) — see `docs/phase2-baselines.md`.

`load_processed` also takes `subsample=N`, a class-stratified draw used by every
script through `--processed --subsample N`. The models are pure numpy and the
real train split is 1.89M x 76 float64 (1.1 GB), so the full split is only
practical for a final run. The draw keeps `min(count, 100)` rows of every class
before sharing out the remainder: proportional allocation alone rounds
Infiltration's 27 training rows to zero and turns the 8-class problem into a
7-class one without saying so.

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

## Real-data preprocessing decisions (found by measurement)

The first run of the pipeline over the real CIC-IDS2017 CSVs exposed two
interactions between individually-correct rules. Both are recorded here because
both silently changed the dataset rather than failing loudly, and both move
every downstream number.

### Destination Port must be dropped *after* deduplication, not before

`Destination Port` is a near-label proxy (port 80 ~ web attack, 21 ~ FTP brute)
so it must not reach the model. It is also the only feature distinguishing one
PortScan flow from the next: a port scan is by definition the same flow repeated
across ports. Dropping it first made those rows byte-identical, and
dedupe-before-split then deleted them.

Measured on `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`:

| | rows | unique after dedupe | lost |
|---|---|---|---|
| Destination Port dropped first | 158,740 | 1,892 | 98.8% |
| Destination Port kept for dedupe | 158,740 | 90,630 | 42.9% |

The fix is ordering, not a rule change: `DEDUPE_KEY_COLUMNS` holds the column
back from the early drop so it can serve as a dedupe key, and it is dropped
immediately afterwards (`drop_columns_post_dedupe`). Two flows identical
*including* the port are genuine duplicates; identical flows to *different*
ports are distinct events.

**Stated limitation.** The surviving 90,630 PortScan rows still contain
feature-identical vectors that can straddle the train/test split once the port
is removed. They are retained deliberately, as distinct observations. Any
PortScan-specific accuracy should be read with that in mind.

### The -1 sentinels are kept, not treated as missing

`Init_Win_bytes_forward` / `_backward` use -1 to mean "no window observed" -
a real property of a flow, not a corrupt reading. Converting it to NaN and
dropping the affected rows discarded **1,441,552 rows, 50.9% of the dataset**,
and did so unevenly across classes (58.0% of Benign, 29.5% of DoS, 0.0% of
PortScan), which distorts the class balance as well as the volume.

That conversion existed for `StandardScaler`, which would have scaled -1 as a
real byte count. Task 1.1 replaced it with `QuantileTransformer`, which bounds
-1 into the bottom quantile, so the reason is gone. `sentinel_policy="keep"` is
now the default; `sentinel_policy="nan"` restores the old behaviour and exists
only to keep the table below reproducible.

### Before / after, full pipeline over all 8 CSVs

Both runs start from the same 2,830,743 raw rows.

| step | old contract | current |
|---|---|---|
| rows after inf/NaN drop | 1,388,204 | 2,827,876 |
| removed by dedupe | 277,000 | 307,078 |
| rows into the split | 1,111,204 | 2,520,798 |
| train / test | 833,403 / 277,801 | 1,890,598 / 630,200 |
| features | 76 | 76 |

Train-split class counts:

| class | old contract | current |
|---|---|---|
| Benign | 628,486 | 1,571,292 |
| DoS | 132,877 | 145,317 |
| DDoS | 61,117 | 96,010 |
| PortScan | 1,419 | **68,021** |
| BruteForce | 6,806 | 6,863 |
| WebAttack | 1,595 | 1,607 |
| Bot | 1,078 | 1,461 |
| Infiltration | 25 | 27 |

### Label separator normalisation

The Web Attack labels carry a Windows-1252 en-dash (`0x96`) in the original
CSVs, `U+FFFD` in mirrors transcoded to UTF-8 with `errors='replace'`, and a
plain ASCII hyphen in some corrected releases. `map_labels` folds every variant
before lookup, so a new mirror cannot present as an unmapped label. This was
caught rather than absorbed because `map_labels` raises on unknown labels
instead of dropping them - had it dropped, WebAttack would simply have been
empty in every result downstream.

---

## Caveats

- **Infiltration cannot support its own class.** 36 rows in the raw data, 27 in
  the train split, against a `RARE_SUPPORT_THRESHOLD` of 50. The 8-class problem
  is effectively 7-class plus a token family. Task 1.2's decision to keep family
  7 but flag it rare should be revisited against merging or dropping it.
- Synthetic multi-class is *optimistic* (macro-F1 approx. 0.91-0.99); on the
  real 8-class problem expect noticeably lower - that is correct, not a
  regression. Runs still fall back to the synthetic generator when a config
  sets no `path` / `processed_dir`.
- The numpy TabTransformer is correct but slow relative to a torch build; keep
  `d_model=32` and round counts modest.
- Raw CSVs live in `data/raw/` (git-ignored, ~844 MiB, fetched by
  `scripts/preprocessing/fetch_cicids2017.sh`). The UNB link is gated behind a registration form, so
  the script pulls the 8 original `MachineLearningCSV` files from the
  `c01dsnap/CIC-IDS2017` mirror; file sizes and the full 15-label distribution
  were verified against the canonical dataset.
