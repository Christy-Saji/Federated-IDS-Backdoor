# Phase 2 — Faithful Baselines

**Duration:** 3 weeks · **Owners:** three parallel tracks · **Entry condition:** Gate G1 cleared

---

## Objective

Reimplement every attack, detector, and defense so it **matches its paper**, and measure everything **with controls**.

After this phase, any statement of the form "defense X fails against attack Y" is a statement about defense X, not about a bug. That distinction is the whole point.

---

## Why this phase exists

Right now the ablation matrix is heading toward this:

| Attack | No defense | FLTrust | FLAME | Both |
|---|---|---|---|---|
| Standard backdoor | 1.0 | 1.0 | 1.0 | 1.0 |
| Distributed | 1.0 | 1.0 | 1.0 | 1.0 |

Every cell 1.0 means **no differential**, so there is nothing for any attack to be better at. The causes are G-01 (uncontrolled metric), G-04 and G-05 (defenses that are not the real defenses), and G-12 (a trigger so far out of distribution it overwhelms everything). This phase fixes the first three.

The three tracks below run **in parallel** and only meet at `runner.py`.

---

# Track M1 — Attack baselines

## Task 2.1 — BadNets across the trigger ladder · 1 week

Implement all four rungs behind one interface, so switching is a config change.

```python
# flids/data/triggers.py
TRIGGERS = {
    "oob_999":       dict(features=[0, 1, 2],     value="999.0",    realizable=False),
    "inbounds_any":  dict(features="top3_shap",   value="p85",      realizable=False),
    "inbounds_free": dict(features="top3_free",   value="p85",      realizable="feature-space"),
    "problemspace":  dict(features="top3_free",   value="from_pcap", realizable=True),  # Phase 3
}

def apply_trigger(X, spec, stats):
    """stats holds per-feature percentiles from the TRAINING set only."""
    Xt = X.copy()
    idx = resolve_features(spec["features"], stats)
    if spec["value"] == "999.0":
        Xt[:, idx] = 999.0
    elif spec["value"] == "p85":
        Xt[:, idx] = stats["p85"][idx]          # in-distribution, unremarkable
    return Xt
```

`p85` is the 85th percentile of that feature's observed distribution — unremarkable, in-distribution, and reachable by an attacker who pads packets or adds delay.

**Keep `oob_999` in the ladder deliberately.** It is the current result, it becomes the unconstrained upper bound, and the gap between it and `problemspace` is the project's Claim 2.

### Targeted, not untargeted

With multi-class labels the backdoor becomes targeted: poisoned samples come from one **source class** (say DDoS) and are relabelled to **Benign**. Record both.

```yaml
attack:
  type: badnets
  source_class: 2          # DDoS
  target_class: 0          # Benign
  poison_ratio: 0.3
  malicious_clients: [0]
```

## Task 2.2 — Clean-model controls for every rung · 3 days

For **each** trigger rung and **each** seed, train a clean FedAvg model and measure ASR on it. This is the baseline table every later number subtracts from.

```
results/baselines/clean_asr.csv
trigger,seed,asr_clean
oob_999,0,____
oob_999,1,____
inbounds_any,0,____
...
```

Nothing in the project reports raw ASR again. Only `dASR = ASR_model - ASR_clean`.

## Task 2.3 — Durability protocol · 4 days

The standard evaluation axis in this literature (G-09), and currently absent.

```yaml
federated:
  rounds: 100
attack:
  attack_window: [1, 20]      # attacker participates rounds 1-20, then leaves
```

Implementation: after round 20, the malicious client trains **honestly** on unpoisoned data (it stays in the federation but stops attacking). Log `dASR` every round to `metrics.jsonl`.

Then compute:

```python
lifespan = backdoor_lifespan(asr_by_round, attack_end_round=20, threshold=0.5)
# rounds after exit until dASR falls below 50% of its peak
```

**Deliverable:** a `dASR` vs round curve, one line per trigger rung, attacker exit marked. This is the single best plot in the final presentation.

### M1 deliverables

- [ ] `flids/attacks/badnets.py` with all four rungs
- [ ] `results/baselines/clean_asr.csv` — every rung × 3 seeds
- [ ] `dASR` table for all rungs, no defense
- [ ] Durability curves, all rungs
- [ ] A short note on whether `oob_999` and `inbounds_*` behave differently at all

---

# Track M2 — Detection baselines

## Task 2.4 — Neural Cleanse, multi-class and calibrated · 1.5 weeks

The single highest-value fix in the project (G-02).

### Steps

1. Run trigger reverse-engineering **per class**, K=8. For each target class `t`, optimise a mask `m` and pattern `p` minimising

   ```
   loss = CE(f(x * (1-m) + p * m), t) + lambda * ||m||_1
   ```

   For tabular data, `m` is a per-feature vector in `[0,1]` and `p` is a per-feature value. **Clamp `p` to the observed feature range** — otherwise the optimiser escapes to absurd values and every class looks backdoored.

2. Compute the real anomaly index:

```python
def nc_anomaly_index(l1_norms):
    L   = np.asarray(l1_norms, float)
    dev = np.abs(L - np.median(L))
    mad = np.median(dev) * 1.4826
    return dev / mad if mad > 0 else np.full_like(dev, np.inf)
```

3. **Calibrate.** Train ~10 independently-seeded **clean** models, run Neural Cleanse on each, collect the null distribution of the max anomaly index, set the threshold at the 95th percentile.

```
results/calibration/nc_null.csv
clean_model_seed,max_anomaly_index
0,____
1,____
...
threshold_p95 = ____        # compare against the paper's default of 2
```

4. Report **ROC/AUC, TPR, and FPR** across backdoored and clean models — not a single anecdote.

### Tuning notes for tabular data

Neural Cleanse was designed for images and does not transfer cleanly. Expect to spend 3–5 days here.

- Optimise on a **subset** (5k samples) — full-dataset optimisation is needlessly slow
- Use Adam, `lr` around `1e-2` on the mask, with `lambda` on an increasing schedule
- Search `lambda` over a log grid; a single fixed value gives either an all-ones mask or an all-zeros mask
- Sigmoid-parameterise the mask (`m = sigmoid(w)`) to keep it in `[0,1]` without projection
- If every class returns a near-identical norm, `lambda` is too high

CatBack (NDSS 2026) reports Neural Cleanse failing entirely on tabular data. **If yours also fails to separate, that is a legitimate, citable finding** — but only once it is correctly implemented and calibrated. Report it as a finding, not as a broken run.

## Task 2.5 — Activation Clustering with the real decision rule · 4 days

1. Extract penultimate-layer activations for **all** training samples of the target class.
2. PCA to 10 components, KMeans `k=2`, compute silhouette.
3. Decision rule: `flagged = silhouette > threshold` (Chen et al. use ~0.10–0.15). **Calibrate the threshold on clean models** rather than accepting the paper's default.
4. Run at poison ratios 1%, 5%, 10%, 30% — and at 0% for the false-positive rate.
5. Optionally implement **exclusionary reclassification**: remove the suspect cluster, retrain, and check whether the removed samples get reclassified to their source class. Stronger evidence than silhouette alone, and it identifies the source class for free.

**Deliverable table:**

| Poison ratio | Silhouette | Flagged | TPR | FPR |
|---|---|---|---|---|
| 0% (clean) | | — | — | |
| 1% | | | | |
| 5% | | | | |
| 10% | | | | |
| 30% | | | | |

### M2 deliverables

- [ ] `flids/defenses/neural_cleanse.py`, multi-class, range-clamped
- [ ] `results/calibration/nc_null.csv` + calibrated threshold
- [ ] Neural Cleanse ROC/AUC across clean and backdoored models
- [ ] `flids/defenses/activation_clustering.py` with silhouette decision rule
- [ ] Activation Clustering TPR/FPR table across poison ratios

---

# Track M3 — Defense baselines

## Task 2.6 — FLTrust, faithfully · 1 week

Fix G-04. The missing piece is the normalization step.

```python
# flids/fl/aggregators/fltrust.py
def fltrust_aggregate(global_params, client_updates, server_update):
    """Cao et al., NDSS 2022, Eq. 4."""
    g0      = flatten(server_update)
    norm_g0 = np.linalg.norm(g0)

    trust, normalised = [], []
    for u in client_updates:
        gi      = flatten(u)
        norm_gi = np.linalg.norm(gi)
        if norm_gi == 0:
            trust.append(0.0); normalised.append(gi); continue

        ts = max(0.0, float(np.dot(gi, g0) / (norm_gi * norm_g0)))   # ReLU(cos)
        trust.append(ts)

        # ---- THE STEP THAT WAS MISSING ----
        # rescale onto the server update's hypersphere BEFORE weighting
        normalised.append(gi * (norm_g0 / norm_gi))

    total = sum(trust)
    if total == 0:
        return global_params, trust                 # reject the round entirely
    agg = sum(t * n for t, n in zip(trust, normalised)) / total
    return apply(global_params, unflatten(agg)), trust
```

The paper is explicit that normalization is what "limits the impact of malicious local model updates with large magnitudes." Without it a client with trust 0.06 still injects at full magnitude — exactly the failure previously observed.

### Also fix the root dataset

```python
from sklearn.model_selection import train_test_split
# class-balanced, held out, never given to any client
_, root_idx = train_test_split(np.arange(len(y_train)), test_size=1000,
                               stratify=y_train, random_state=0)
```

Small (500–1000), clean, **stratified**. Not `X_train[-500:]`.

### Report the right metric

```python
detection_auc(trust_scores, is_malicious)   # per round
```

Trust-score AUC as a malicious-client classifier is far more informative than final ASR, and it is what makes M3's section rigorous. Previously all ten clients scored 0.455–0.496, which is AUC ~0.5 — no separation. Note whether normalization changes that.

## Task 2.7 — FLAME, faithfully · 1 week

Fix G-05. `pip install hdbscan`.

```python
# flids/fl/aggregators/flame.py
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

def flame_aggregate(global_params, client_updates, lambda_noise=0.001):
    """Nguyen et al., USENIX Security 2022."""
    U = np.stack([flatten(u) for u in client_updates])
    N = len(U)

    # --- 1. cluster on COSINE DISTANCE with HDBSCAN ---
    D = cosine_distances(U).astype(np.float64)
    labels = hdbscan.HDBSCAN(min_cluster_size=N // 2 + 1,
                             min_samples=1,
                             metric='precomputed',
                             allow_single_cluster=True).fit_predict(D)
    kept = np.where(labels != -1)[0]        # -1 = noise = rejected
    if len(kept) == 0:
        kept = np.arange(N)

    # --- 2. clip to the median norm of ALL updates (not just kept) ---
    norms = np.linalg.norm(U, axis=1)
    S     = float(np.median(norms))
    clipped = np.stack([U[i] * min(1.0, S / norms[i]) for i in kept])

    # --- 3. average, then add adaptive noise derived from the clip bound ---
    agg   = clipped.mean(axis=0)
    sigma = lambda_noise * S
    agg  += np.random.normal(0, sigma, agg.shape)

    removed = [i for i in range(N) if i not in set(kept)]
    return apply(global_params, unflatten(agg)), removed
```

`min_cluster_size = N//2 + 1` with `allow_single_cluster=True` is the key: **at most one cluster can form**, so when all clients are benign nobody is rejected. KMeans always returns two non-empty clusters, which is why the old version rejected honest clients every round.

### Sanity check before trusting it

Run FLAME with **zero** malicious clients for 10 rounds. `removed` should be empty or near-empty every round. If honest clients are being rejected with no attacker present, the parameters are wrong — fix that before running any attack.

## Task 2.8 — Gradient-norm anomaly scorer · 2 days

The cheap baseline defense. Flag clients whose update L2 norm deviates from the median by more than `k` MADs.

```python
def gradnorm_scores(client_updates, k=2.0):
    norms = np.array([np.linalg.norm(flatten(u)) for u in client_updates])
    dev   = np.abs(norms - np.median(norms))
    mad   = np.median(dev) * 1.4826
    scores = dev / mad if mad > 0 else np.zeros_like(dev)
    return scores, scores > k
```

Report its detection AUC alongside FLTrust and FLAME. It is expected to do poorly against norm-matched attacks — that is a useful contrast, and it is nearly free to include.

## Task 2.9 — Combined aggregator · 2 days

`fltrust+flame`: FLAME filters first, FLTrust then trust-weights the survivors. Verify the composition order is logged, and that both stages' per-client scores land in `metrics.jsonl`.

### M3 deliverables

- [ ] `fltrust.py` with normalization + stratified root set
- [ ] `flame.py` with HDBSCAN + all-norm clipping + adaptive noise
- [ ] `gradnorm.py`
- [ ] `fltrust+flame.py`
- [ ] FLAME zero-attacker sanity check passing
- [ ] Detection AUC + FPR per round for all four aggregators
- [ ] Comparison against the Phase 0 defect list: which reported failures were bugs, which were real?

---

## Gate G2

**On the `oob_999` trigger with correct defenses, at least one defense reduces `dASR` meaningfully.**

Checklist:

- [ ] Every ASR in `results/` is a `dASR` with a recorded clean baseline
- [ ] Neural Cleanse produces a real anomaly index with a calibrated threshold and reported FPR
- [ ] Activation Clustering uses silhouette, with FPR on clean models
- [ ] FLTrust normalises; FLAME uses HDBSCAN and passes the zero-attacker check
- [ ] Every defense reports detection AUC, not just "was the client removed"
- [ ] The Phase 0 defect list is fully closed out, item by item

### If all defenses still fail at dASR = 1.0

**That is now a genuine finding, not a bug.** With faithful implementations it becomes a statement about the defenses — worth writing up as a contribution in its own right, and consistent with what CatBack reports for tabular data. Record it, then continue to Phase 3, where the constraint ladder is the real story anyway.

---

## Common failure modes in this phase

- **Comparing defenses across different seeds or partitions.** Fix `alpha` and seed when comparing aggregators. Vary one thing at a time.
- **Reporting "client removed: True/False" as the defense metric.** Binary removal hides everything. AUC over the continuous score is the metric.
- **Skipping the FLAME zero-attacker check.** It catches the single most common misconfiguration in five minutes.
- **Giving up on Neural Cleanse when it fails to separate.** Failure on tabular data is a documented, publishable result — but only from a correct implementation with a calibrated threshold.
- **Letting the three tracks diverge on the config schema.** Any schema change goes through all three members.

---

**Previous:** [Phase 1 — Foundation Rebuild](phase-1-foundation-rebuild.md) · **Next:** [Phase 3 — The Realizable Trigger](phase-3-realizable-trigger.md)
