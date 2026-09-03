# Phase 5 — Evaluation Campaign

**Duration:** 3 weeks · **Owner:** M3 lead, M2 support · **Entry condition:** Gate G4 cleared

---

## Objective

Run the full experimental matrix, produce every figure and table the report needs, and make sure **every claim traces back to a reproducible run**. Plus the cross-dataset, adversarial-robustness, SHAP, and dashboard deliverables that round out the three members' sections.

This is where the config-driven runner from Phase 1 pays off. Without it, this phase is impossible.

---

## The matrix

```
4 triggers  x  4 defenses  x  4 alpha values  x  3 seeds  =  192 runs
```

| Axis | Values |
|---|---|
| Trigger | `oob_999`, `inbounds_any`, `inbounds_free`, `problemspace` |
| Defense | none, FLTrust, FLAME, FLTrust+FLAME |
| Heterogeneity | `alpha in {0.1, 0.5, 1.0, inf}` |
| Seed | 0, 1, 2 |

At ~10 min/run that is ~32 GPU-hours — parallelizable across the three machines plus Colab. Generate the configs programmatically:

```python
# scripts/gen_configs.py
import itertools, yaml, hashlib, json
for trig, defense, alpha, seed in itertools.product(TRIGGERS, DEFENSES, ALPHAS, SEEDS):
    cfg = build_config(trig, defense, alpha, seed)
    run_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    yaml.safe_dump(cfg, open(f"configs/gen/{run_id}.yaml", "w"))
```

### If the schedule tightens — the prioritized subset

Define this **before** launching, so triage under pressure is not needed later. Run in this order:

1. All 4 triggers × {none, FLTrust+FLAME} × `alpha=0.5` × 3 seeds  (**32 runs — the headline Claim 2 table**)
2. Add the other two defenses at `alpha=0.5`  (+32)
3. Add `alpha in {0.1, 1.0, inf}` for the headline trigger×defense cells  (the heterogeneity story)
4. Everything else, time permitting

Rows 1–2 alone are a complete, defensible result set. The rest is depth.

---

## Task 5.1 — Full ablation matrix · M3 · 1 week

Launch the runs, collect `results/`, build the master results table.

```python
# notebooks/aggregate.py  — READS results/, never writes to it
import glob, json, pandas as pd
rows = [json.load(open(f)) for f in glob.glob("results/*/summary.json")]
df = pd.DataFrame(rows)
master = (df.groupby(["trigger", "defense", "alpha"])
            .agg(dASR_mean=("delta_asr", "mean"),
                 dASR_std =("delta_asr", "std"),
                 mta_mean =("macro_f1", "mean"),
                 auc_mean =("detection_auc", "mean"))
            .round(3))
```

### The headline table — the paper's centrepiece

| Trigger | Realizable | Clean ASR | dASR none | dASR FLTrust+FLAME | Lifespan |
|---|---|---|---|---|---|
| oob_999 | No | | | | |
| inbounds_any | No | | | | |
| inbounds_free | feature-space | | | | |
| problemspace | **Yes** | | | | |

The gap between the top row and the bottom row **is** Claim 2. Every cell is mean ± std over 3 seeds.

**Deliverable:** master results CSV + the headline table.

---

## Task 5.2 — Durability curves · M3 · 3 days

For every attack variant: `dASR` vs round, attacker exit at round 20 marked, one panel per trigger, lines coloured by defense.

Headline metric per curve: **lifespan = rounds until `dASR` < 50% of peak**.

**Deliverable:** durability figure (the strongest single plot in the presentation) + a lifespan summary table.

---

## Task 5.3 — UNSW-NB15 cross-dataset replication · M2 · 1 week

Rerun the **headline conditions only** (not all 192) on UNSW-NB15 to show the results are not a CIC-IDS2017 artifact.

### Steps

1. Preprocess UNSW-NB15 through the same pipeline. Its feature set differs — build a UNSW-specific perturbability table (smaller effort, the categories are the same).
2. Run: 4 triggers × {none, FLTrust+FLAME} × `alpha=0.5` × 3 seeds.
3. Compare the Claim 2 gap across datasets. Consistent direction across two datasets is much stronger than one.

**Bonus — cross-dataset trigger transfer (adjacent win, high signal).** Does a trigger optimized on CIC-IDS2017 still work on UNSW-NB15? CatBack shows tabular triggers transfer across model families; nobody has tested cross-*dataset* transfer for NIDS. One extra experiment, potentially a standalone finding.

**Deliverable:** UNSW-NB15 headline table + cross-dataset comparison.

---

## Task 5.4 — Adversarial robustness (FGSM/PGD) · M2 · 3 days

Stress-test the classifier itself, separate from the backdoor story. Use IBM ART.

```python
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

clf = PyTorchClassifier(model=model, loss=criterion,
                        input_shape=(n_features,), nb_classes=8)
for eps in [0.05, 0.1, 0.2, 0.3]:
    X_adv = FastGradientMethod(clf, eps=eps).generate(X_test)
    acc   = (clf.predict(X_adv).argmax(1) == y_test).mean()
    print(f"FGSM eps={eps}: acc={acc:.3f}")
```

**Realism caveat to state:** FGSM/PGD perturb all features freely, including `fixed` ones, so this is a *feature-space* robustness bound — an upper limit on the attacker, not a realizable evasion. This is consistent with the project's problem-space thesis, and worth one honest sentence rather than overclaiming.

**Deliverable:** accuracy-vs-epsilon curves for FGSM and PGD, with the realism caveat noted.

---

## Task 5.5 — SHAP on clean vs backdoored models · M3 · 3 days

More than an explainability deliverable — it is a lightweight fourth **detector** (adjacent win, high signal).

### Steps

1. Compute SHAP attributions on the clean global model and on the backdoored global model.
2. Compare. **If the trigger features show anomalous attribution in the backdoored model, that is a detection signal** — a defender could flag it.
3. Report both the standard explainability view (top features driving attack classification) and the clean-vs-backdoored differential.

**Privacy caveat for the limitations section:** SHAP on a federated model can leak information about local data distributions, since attributions reflect the data the model was trained on. This is an active research area (Federated SHAP; SHAP entropy regularization). One honest sentence.

**Deliverable:** SHAP summary plots (clean + backdoored) + the differential analysis.

---

## Task 5.6 — Streamlit dashboard · M3 · 4 days

Reads `results/`, visualises the federated process. Runs **locally** (Colab cannot host Streamlit well).

Panels:
- Live/replayed FL rounds — accuracy and `dASR` per round
- Per-client trust scores / anomaly scores each round, malicious clients highlighted
- Which clients were filtered by FLAME each round
- The Claim 2 table, interactive by trigger and defense
- Durability curve with the attacker-exit marker

The dashboard **reads `results/`; it does not train anything.** Keep all training in the runner.

**Deliverable:** working dashboard + a 90-second demo script for the presentation.

---

## Task 5.7 — Statistical treatment · M3 · 2 days

- Every headline number: mean ± std over 3 seeds.
- Comparative claims ("`inbounds_free` `dASR` < `oob_999` `dASR`") get a paired test across seeds, with the p-value or effect size reported.
- Do not over-claim significance with 3 seeds — report effect sizes and be honest about the small n. Examiners respect calibrated confidence more than inflated significance.

**Deliverable:** a stats appendix backing every comparative claim in the report.

---

## Gate G5

**Every claim in the report traces to a `results/<run_id>/` directory containing its config.**

Checklist:

- [ ] Headline Claim 2 table complete, mean ± std
- [ ] Durability curves for every variant
- [ ] UNSW-NB15 replication of headline conditions
- [ ] FGSM/PGD robustness curves, realism caveat stated
- [ ] SHAP clean-vs-backdoored differential
- [ ] Dashboard reads `results/` and runs locally
- [ ] Every figure's underlying `run_id`s are listed in a manifest, so any number regenerates on demand

---

## Common failure modes in this phase

- **Notebooks that train.** All training is `runner.py`. Notebooks only read `results/`. Breaking this reintroduces the reproducibility problem Phase 1 solved.
- **Running the full 192 before the subset.** Run the 32-run headline subset first — it is the whole result. Depth comes after.
- **Overclaiming FGSM/PGD as realizable evasion.** It is a feature-space bound. Say so.
- **Reporting p-values from 3 seeds as if n were large.** Effect sizes, honest confidence.
- **Losing the config-to-figure mapping.** Keep a manifest linking each figure to its `run_id`s. Gate G5 depends on it.

---

**Previous:** [Phase 4 — Distributed & Adaptive Attack](phase-4-distributed-attack.md) · **Next:** [Phase 6 — Writing & Defense Prep](phase-6-writing-and-defense.md)
