# Backdoor Attacks and Defenses in Federated Learning for Network Intrusion Detection

Study of **feature-space backdoor attacks** against federated intrusion-detection systems trained on
[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), and how well
server-side and model-level defenses catch them.

Everything is pure NumPy (no PyTorch, no GPU), deterministic by seed, and
reproducible from a clean checkout. Includes an offline interactive dashboard
for watching a federation train and get poisoned.

## Key findings

The central lesson is that **a defense result means nothing without the trigger
it was tested against.** The same five aggregators give opposite answers on two
trigger types ("rungs"), so every number below names its rung.

| | `oob_999` (out-of-range control) | `inbounds_free` (realizable trigger) |
|---|---|---|
| Trigger | stamps `999.0` on fixed columns | in-distribution values on attacker-controllable columns only |
| Any defense removes all attackers? | **No** — 0 of 25 defended runs excluded all four attackers | **Yes** — FLTrust excludes all four together in 12 rounds |
| Backdoor success (raw ASR) | 1.000 in 25/25 runs | FLTrust 0.372 vs FedAvg 0.553, lower in 5/5 seeds |
| FLTrust detection AUC | 0.597 ± 0.111, orientation flips between seeds | **0.863 ± 0.080**, correctly oriented 5/5 |
| FLAME detection AUC | **0.227 ± 0.128, below chance in 5/5 seeds** | 0.597 ± 0.109, no stable inversion |
| Model-level (Activation Clustering) | detects it perfectly | blind to it |

(Five seeds per cell, mean ± std, original CIC-IDS2017 release.)

- **FLAME's cosine score is inverted on the easy trigger.** It ranks attackers
  as the *least* suspicious clients: a `999.0` objective is so easy to learn
  that they converge to consensus fastest and land nearest the centroid instead
  of furthest from it.
- **That inversion is a property of the easy trigger, not of FLAME.** A harder
  in-distribution objective dissolves it, as the mechanism predicts.
- **The trigger that defeats every defense is the one an attacker cannot
  build.** The realizable one is weaker (dASR ≈ 0.36) and partly catchable at
  the client level by FLTrust, while model-level detection points the opposite
  way.

Full tables, seeds and caveats: [docs/phase2-baselines.md](docs/phase2-baselines.md) §0.

## What's in the box

- **Data pipeline** — 14 raw CIC-IDS2017 labels mapped to 8 attack families,
  de-duplicated *before* splitting, quantile-normalised, with IID and Dirichlet
  non-IID client partitioning.
- **Models** — a NumPy MLP and a NumPy TabTransformer with a hand-written
  backward pass.
- **Federated learning** — a deterministic FedAvg loop with a configurable
  malicious-client fraction and poisoning window.
- **Trigger ladder** — four feature-space triggers from `oob_999` up to a
  realizable, per-feature-perturbability-aware trigger.
- **Aggregators / defenses** — FedAvg, gradient-norm scorer, FLTrust, FLAME and
  FLTrust+FLAME (faithful reimplementations); Neural Cleanse and Activation
  Clustering at the model level.
- **Metrics** — ΔASR, main-task accuracy, detection AUC, defense false-positive
  rate, backdoor lifespan.
- **Dashboard** — a local, dependency-free web app (see below).
- **Tests** — 67 stdlib `unittest` tests covering the bugs that previously
  failed silently.

## Getting started

Requires Python 3.10+. Dependencies are numpy, pandas, scikit-learn and scipy,
pinned in `requirements.txt`.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt      # Windows
# Linux/macOS: ./.venv/bin/python
```

### Get the data

The dataset (about 2.3 GB) is not in the repository. Download
`MachineLearningCSV.zip` from the
[CIC-IDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html) into
`data/raw/`, then build the processed arrays once:

```bash
./.venv/Scripts/python.exe -m scripts.preprocessing.preprocess --data data/raw
```

### Run an experiment

```bash
# one condition -> results/<run_id>/
./.venv/Scripts/python.exe -m flids.runner --config configs/clean_fedavg_real.yaml

# the full baseline campaign, both trigger rungs (many hours; --seed N for a sweep)
./.venv/Scripts/python.exe -m scripts.baselines.run_all_real
```

Each run writes `config.yaml`, `env.json`, `metrics.jsonl`, `summary.json` and
the final model to `results/<run_id>/`. The `run_id` is the SHA-256 of the
resolved config, and the runner refuses to overwrite an existing one.

### Reproduce the reported numbers

```bash
./.venv/Scripts/python.exe -m scripts.baselines.detection_report
./.venv/Scripts/python.exe -m scripts.baselines.prevention_report
```

Both read every recorded run and group by `(trigger, aggregator)`.

### Run the tests

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -t .
```

## The dashboard

```bash
./.venv/Scripts/python.exe -m flids.dashboard      # http://127.0.0.1:8765
```

Options: `--port N`, `--no-browser`, `--processed DIR`.

Four views:

1. **Live federation** — watch clients train and get poisoned round by round,
   choosing the aggregator, trigger rung, number of malicious clients, rounds
   and Dirichlet α.
2. **Recorded runs** — every run saved in `results/`.
3. **Compare defenses** — all five defenses side by side on recorded runs,
   replayable round by round, with a trigger-rung selector.
4. **Backdoor inspector** — pushes one real CIC-IDS2017 flow through a trained
   model before and after the trigger is stamped.

It uses only the standard library plus hand-written HTML/CSS/JS, with no CDN, so
it works fully offline. It never writes to `results/`. A presenter's
walkthrough is in [docs/demo-script.md](docs/demo-script.md).

## Repository layout

```
flids/            the research library (pure numpy)
  data/           loaders, label mapping, partitioning, trigger ladder
  models/         MLP, TabTransformer, registry
  fl/             server, client, aggregators
  attacks/        trigger stamping, ASR evaluation
  defenses/       Neural Cleanse, Activation Clustering
  eval/           metrics
  dashboard/      the demo app
configs/          one YAML per experimental condition (*_real.yaml = real data)
scripts/          runnable entrypoints: preprocessing, validation, baselines, gates
tests/            unit tests
docs/             write-ups: validity report, baselines, gate verdicts, demo script
results/          append-only recorded runs (committed)
data/             git-ignored; regenerate with the preprocessing step
```

## Documentation

- [docs/phase2-baselines.md](docs/phase2-baselines.md) — the main results and how they were measured
- [docs/phase0-validity-report.md](docs/phase0-validity-report.md) — validity checks on the attack setup
- [docs/phase0-defense-diff.md](docs/phase0-defense-diff.md) — each defense compared with its paper
- [docs/gate-verdicts.md](docs/gate-verdicts.md) — verification checks and their outcomes
- [docs/demo-script.md](docs/demo-script.md) — walkthrough of the dashboard

## Scope and limitations

- **Feature-space only.** The attacks perturb flow features directly. Crafting
  real packets and round-tripping them through CICFlowMeter (a problem-space
  attack) is out of scope.
- **Original CIC-IDS2017.** Engelen et al. (WTMC 2021) found more than 20% of
  the dataset was mislabeled or reconstructed. All numbers here are on the
  original release; repeating the study on a corrected release (Improved
  CIC-IDS2017 / LYCOS-IDS2017) is the natural follow-up.
- **Reproducibility.** Runs are deterministic per seed and the reference run's
  `summary.json` is byte-identical across thread counts on the same machine.
  Trained weights are bit-identical only at the same BLAS thread count.
- **Not yet explored.** All recorded runs use Dirichlet α = 0.5 and the MLP; the
  TabTransformer is implemented and tested but has no recorded federated runs.
